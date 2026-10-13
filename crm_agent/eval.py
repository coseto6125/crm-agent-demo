import asyncio
import json
import tempfile
from pathlib import Path

from crm_agent import agents, db
from crm_agent.cf_embed import CloudflareEmbedder
from crm_agent.gemini import Gemini, GeminiError
from crm_agent.tools import Ctx, FaqIndex

CASES_PATH = db.DATA_DIR / "eval_cases.json"
PAUSE_SECONDS = 4
INTENDED_HANDOFFS = ("routed_to_human", "faq_low_similarity")
DEPENDENCY_HANDOFFS = ("llm_error", "turn_timeout", "bad_model_output")
RATE_LIMITED = "llm_error status=429"


def check(ctx: Ctx, conv_id: str, result: dict, expect: dict) -> list[str]:
    """Return the failed expectations about the reply, tickets and orders (empty means all hold)."""
    reply, fails = result["reply"], []
    q = lambda sql, *a: ctx.conn.execute(sql, a).fetchall()
    if (agent := expect.get("agent")) and result["agent"] != agent:
        fails.append(f"agent={result['agent']} want {agent}")
    if (sid := expect.get("cites")) and not (sid in reply and "來源" in reply):
        fails.append(f"no citation {sid}")
    fails += [f"missing {s!r}" for s in expect.get("contains", []) if s not in reply]
    fails += [f"unexpected {s!r}" for s in expect.get("no_contains", []) if s in reply]
    if (any_of := expect.get("contains_any")) and not any(s in reply for s in any_of):
        fails.append(f"none of {any_of}")
    tickets = q("SELECT status FROM tickets WHERE conversation_id = ?", conv_id)
    if expect.get("ticket") and not any(t["status"] == expect["ticket"] for t in tickets):
        fails.append(f"no {expect['ticket']} ticket")
    if expect.get("only_pending") and any(t["status"] != "pending" for t in tickets):
        fails.append("ticket not pending")
    for order_id, status in expect.get("order_status", {}).items():
        if (row := q("SELECT status FROM orders WHERE order_id = ?", order_id)[0])["status"] != status:
            fails.append(f"order {order_id} status={row['status']} want {status}")
    return fails


def classify(ctx: Ctx, conv_id: str, result: dict, expect: dict) -> tuple[str, list[str]]:
    """
    Return PASS, FAIL or SKIP with the failure reasons.

    SKIP only when the turn hit HTTP 429 and the agent routed so far matches the expectation.
    A handoff caused by a dependency error is never a PASS.
    """
    reasons = [r[0] for r in ctx.conn.execute("SELECT reason FROM handoffs WHERE conversation_id = ? ORDER BY id", (conv_id,))]
    routed = ctx.conn.execute(
        "SELECT agent FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT 1", (conv_id,)
    ).fetchone()[0]
    want = expect.get("agent")
    if any(r.startswith(RATE_LIMITED) for r in reasons) and (routed is None or want is None or routed == want):
        return "SKIP", []
    fails = check(ctx, conv_id, result, expect)
    fails += [f"dependency handoff: {r}" for r in reasons if r.startswith(DEPENDENCY_HANDOFFS)]
    others = [r for r in reasons if not r.startswith(DEPENDENCY_HANDOFFS)]
    if expect.get("handoff"):
        if not any(r.startswith(INTENDED_HANDOFFS) for r in others):
            fails.append("no intended handoff")
    elif others:
        fails.append(f"unexpected handoff: {others[0]}")
    return ("FAIL" if fails else "PASS"), fails


async def preload(ctx: Ctx) -> None:
    """Embed the FAQ once; on failure leave the index unbuilt so each case reports its own error."""
    try:
        await ctx.faq.build(ctx.embedder)
    except GeminiError as e:
        print(f"WARN FAQ preload failed: {e}", flush=True)


async def main() -> None:
    cases = json.loads(CASES_PATH.read_text("utf-8"))
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    with tempfile.TemporaryDirectory() as tmp:
        ctx = Ctx(db.connect(str(Path(tmp) / "eval.db")), Gemini(temperature=0.0), CloudflareEmbedder(), FaqIndex.load())
        await preload(ctx)
        for i, case in enumerate(cases):
            if i:
                await asyncio.sleep(PAUSE_SECONDS)
            conv_id = f"eval-{case['id']}"
            result = await agents.handle_message(ctx, conv_id, case["message"])
            verdict, fails = classify(ctx, conv_id, result, case["expect"])
            counts[verdict] += 1
            detail = " (rate limited)" if verdict == "SKIP" else f" {'; '.join(fails)}"
            print(f"{verdict} {case['id']} agent={result['agent']}{detail}".rstrip(), flush=True)
            if verdict == "FAIL":
                print(f"  reply: {result['reply'][:200]}", flush=True)
    scored = counts["PASS"] + counts["FAIL"]
    print(
        f"ACCURACY {counts['PASS']}/{scored} {counts['PASS'] / scored if scored else 0:.1%}"
        f" PASS {counts['PASS']} FAIL {counts['FAIL']} SKIP {counts['SKIP']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
