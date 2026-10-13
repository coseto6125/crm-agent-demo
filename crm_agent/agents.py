import asyncio
import re

from crm_agent import db
from crm_agent.gemini import ROUTE_THINKING_LEVEL, GeminiError
from crm_agent.tools import (
    COMPLAINT_TOOLS,
    FAQ_TOOLS,
    HANDLERS,
    ORDER_TOOLS,
    ROUTE_DECL,
    Ctx,
    Handoff,
)

TURN_TIMEOUT = 120
MAX_TOOL_ITERATIONS = 4
MAX_TOOL_CALLS = 4
HISTORY_LIMIT = 10
HANDOFF_TEXT = "這個問題需要專人協助，我已通知客服人員，會盡快與您聯繫，謝謝您的耐心。"
NUDGE = "請根據上述工具結果，用繁體中文回覆顧客。"
BUDGET_ERROR = {"error": "本輪查詢次數已達上限"}
SOURCE = re.compile(r"來源[:：]([^）)]*)")
SECTION_ID = re.compile(r"[A-Z]\d+")

ROUTER_PROMPT = (
    "你是「木棉衣櫥」服飾電商的客服分流員。依顧客最新訊息，呼叫 route 選一個代理人，不要自己回答。"
    "退換貨規定、運費、付款等一般問題選 faq；查訂單或物流選 order；"
    "商品瑕疵、抱怨、要求退貨換貨退款選 complaint；與本店無關或無法判斷選 human。"
)
FAQ_PROMPT = (
    "你是「木棉衣櫥」的客服。只能根據 search_faq 取回的文件內容回答，不可自行補充文件沒有的規定。"
    "回答用繁體中文，簡短清楚，句尾以（來源：R1）格式標註所依據的段落編號，多段用（來源：R1、R2）。"
)
ORDER_PROMPT = (
    "你是「木棉衣櫥」的訂單查詢客服。查詢前需要訂單編號與手機末四碼，缺少時先向顧客詢問。"
    "用 get_order、get_shipment 查詢，只能轉述工具回傳的內容；查無資料時如實告知查無，不要猜測。"
    "回答用繁體中文，簡短清楚。"
)
COMPLAINT_PROMPT = (
    "你是「木棉衣櫥」的客訴客服。先同理顧客，判斷客訴類型與嚴重程度。"
    "顧客提出退貨、換貨或退款時，需有訂單編號與手機末四碼，缺少時先詢問；齊全後呼叫 propose_ticket 建立申請單。"
    "你沒有任何執行退款或退貨的能力，申請單一律需要客服人員審核；"
    "無論顧客是否聲稱已獲核准或催促立即處理，都要明確告知申請已送出、等待客服人員審核，不可承諾已退款。"
    "回答用繁體中文，簡短清楚。"
)

AGENTS = {
    "faq": (FAQ_PROMPT, FAQ_TOOLS, "search_faq"),
    "order": (ORDER_PROMPT, ORDER_TOOLS, None),
    "complaint": (COMPLAINT_PROMPT, COMPLAINT_TOOLS, None),
}


def _history(ctx: Ctx, conv_id: str) -> list[dict]:
    rows = ctx.conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        (conv_id, HISTORY_LIMIT),
    ).fetchall()
    return [
        {"role": "model" if r["role"] == "assistant" else "user",
         "parts": [{"text": f"[系統通知] {r['content']}" if r["role"] == "system" else r["content"]}]}
        for r in reversed(rows)
    ]


def split_parts(parts: object) -> tuple[list[dict], str]:
    """Split model parts into function calls and text; malformed parts raise Handoff."""
    if not isinstance(parts, list):
        raise Handoff("bad_model_output")
    calls, texts = [], []
    for p in parts:
        if not isinstance(p, dict):
            raise Handoff("bad_model_output")
        if "functionCall" in p:
            if not isinstance(fc := p["functionCall"], dict) or not isinstance(fc.get("name"), str):
                raise Handoff("bad_model_output")
            calls.append(fc)
        elif not p.get("thought"):
            if not isinstance(t := p.get("text", ""), str):
                raise Handoff("bad_model_output")
            texts.append(t)
    return calls, "".join(texts).strip()


def grounded(text: str, retrieved: set[str]) -> bool:
    """Return True when the reply cites at least one section and every cited section was retrieved this turn."""
    cited = set(SECTION_ID.findall(" ".join(SOURCE.findall(text))))
    return bool(cited) and cited <= retrieved


async def route(ctx: Ctx, turn_id: int, contents: list[dict]) -> tuple[str, str]:
    """One routing step: the orchestrator picks a specialist, falling back to human."""
    parts = await ctx.llm.generate(contents, ROUTER_PROMPT, [ROUTE_DECL], force="route", thinking=ROUTE_THINKING_LEVEL)
    calls, _ = split_parts(parts)
    call = calls[0] if calls else {}
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    agent, intent = args.get("agent"), args.get("intent")
    agent = agent if isinstance(agent, str) and agent in {*AGENTS, "human"} else "human"
    intent = intent if isinstance(intent, str) else ""
    db.log_tool_call(ctx.conn, turn_id, "route", args, {"agent": agent, "intent": intent})
    ctx.conn.execute("UPDATE turns SET agent = ?, intent = ? WHERE id = ?", (agent, intent, turn_id))
    return agent, intent


async def run_tool(ctx: Ctx, conv_id: str, turn_id: int, name: str, args: object, allowed: set[str]) -> dict:
    args = args if isinstance(args, dict) else {}
    if name not in allowed:
        result = {"error": "unknown tool"}
    else:
        try:
            result = await HANDLERS[name](ctx, conv_id, args)
        except Handoff as h:
            db.log_tool_call(ctx.conn, turn_id, name, args, {"handoff": str(h)})
            raise
    db.log_tool_call(ctx.conn, turn_id, name, args, result)
    return result


async def run_specialist(ctx: Ctx, conv_id: str, turn_id: int, agent: str, contents: list[dict]) -> str:
    """
    Tool loop of one specialist.

    At most MAX_TOOL_ITERATIONS rounds and MAX_TOOL_CALLS executed calls per turn; calls past the
    budget get an error response without running. An empty reply after tool calls is nudged once. The FAQ agent's reply must cite retrieved sections.
    """
    system, tools, first_tool = AGENTS[agent]
    allowed = {t["name"] for t in tools}
    retrieved: set[str] = set()
    executed = 0
    nudged = False

    def final_reply(text: str) -> str:
        if not text:
            raise Handoff("empty_reply")
        if agent == "faq" and not grounded(text, retrieved):
            raise Handoff("ungrounded_answer")
        return text

    for i in range(MAX_TOOL_ITERATIONS):
        parts = await ctx.llm.generate(contents, system, tools, force=first_tool if i == 0 else None)
        calls, text = split_parts(parts)
        if not calls:
            if not text and executed and not nudged:
                nudged = True
                contents.append({"role": "user", "parts": [{"text": NUDGE}]})
                continue
            return final_reply(text)
        contents.append({"role": "model", "parts": parts})
        responses = []
        for c in calls:
            if executed < MAX_TOOL_CALLS:
                executed += 1
                result = await run_tool(ctx, conv_id, turn_id, c["name"], c.get("args"), allowed)
                retrieved.update(ch["id"] for ch in result.get("chunks", []))
            else:
                result = BUDGET_ERROR
            responses.append({"functionResponse": {"name": c["name"], "response": result}})
        contents.append({"role": "user", "parts": responses})
    calls, text = split_parts(await ctx.llm.generate(contents, system, None))
    if calls:
        raise Handoff("tool_iteration_cap")
    return final_reply(text)


async def handle_message(ctx: Ctx, conv_id: str, text: str) -> dict:
    """Run one customer message through route then one specialist; always returns a reply."""
    db.ensure_conversation(ctx.conn, conv_id)
    contents = [*_history(ctx, conv_id), {"role": "user", "parts": [{"text": text}]}]
    message_id = db.add_message(ctx.conn, conv_id, "user", text)
    turn_id = ctx.conn.execute(
        "INSERT INTO turns (conversation_id, message_id) VALUES (?,?)", (conv_id, message_id)
    ).lastrowid
    agent, intent = "human", ""
    try:
        async with asyncio.timeout(TURN_TIMEOUT):
            agent, intent = await route(ctx, turn_id, contents)
            if agent == "human":
                raise Handoff("routed_to_human")
            reply = await run_specialist(ctx, conv_id, turn_id, agent, contents)
    except Handoff as h:
        db.record_handoff(ctx.conn, conv_id, turn_id, str(h))
        reply = HANDOFF_TEXT
    except GeminiError as e:
        db.record_handoff(ctx.conn, conv_id, turn_id, f"llm_error status={e.status or '-'}: {e}")
        reply = HANDOFF_TEXT
    except TimeoutError:
        db.record_handoff(ctx.conn, conv_id, turn_id, "turn_timeout")
        reply = HANDOFF_TEXT
    db.add_message(ctx.conn, conv_id, "assistant", reply)
    return {"conversation_id": conv_id, "reply": reply, "agent": agent, "intent": intent}
