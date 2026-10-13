import pytest
from conftest import FakeLLM, call, route_to, text
from crm_agent import agents, db
from crm_agent.cf_embed import EmbedError
from crm_agent.gemini import ROUTE_THINKING_LEVEL, GeminiError
from crm_agent.tools import COMPLAINT_TOOLS, FAQ_TOOLS, ORDER_TOOLS, cosine, get_order, propose_ticket, search_faq

HANDOFFS = "SELECT reason FROM handoffs"


def rows(conn, sql: str):
    return [tuple(r) for r in conn.execute(sql).fetchall()]


def test_complaint_agent_declares_only_propose_ticket():
    names = {t["name"] for t in COMPLAINT_TOOLS}
    assert names == {"propose_ticket"}
    actions = COMPLAINT_TOOLS[0]["parameters"]["properties"]["action"]["enum"]
    assert actions == ["return", "exchange", "refund"]
    assert {t["name"] for t in agents.AGENTS["complaint"][1]} == names
    assert {t["name"] for t in ORDER_TOOLS} == {"get_order", "get_shipment"}
    assert {t["name"] for t in FAQ_TOOLS} == {"search_faq"}


async def test_search_faq_below_threshold_records_handoff_without_answer_call(conn, make_ctx):
    llm = FakeLLM([route_to("faq"), call("search_faq", query="天氣如何")], doc_vec=(0, 1), query_vec=(1, 0))
    result = await agents.handle_message(make_ctx(llm), "c1", "今天天氣如何")
    assert result["reply"] == agents.HANDOFF_TEXT
    assert len(llm.calls) == 2
    assert rows(conn, HANDOFFS)[0][0].startswith("faq_low_similarity")


async def test_search_faq_above_threshold_returns_chunks_for_answer(conn, make_ctx):
    llm = FakeLLM(
        [route_to("faq"), call("search_faq", query="退貨"), text("七天內可退貨（來源：R1）")],
        doc_vec=(1, 0), query_vec=(1, 0),
    )
    result = await agents.handle_message(make_ctx(llm), "c1", "可以退貨嗎")
    assert "來源：R1" in result["reply"]
    assert llm.calls[1]["force"] == "search_faq"
    assert llm.calls[0]["thinking"] == ROUTE_THINKING_LEVEL
    assert rows(conn, HANDOFFS) == []


async def test_get_order_wrong_phone_returns_not_found(make_ctx):
    ctx = make_ctx(FakeLLM())
    wrong = await get_order(ctx, "c", {"order_id": "MM1001", "phone_last4": "0000"})
    right = await get_order(ctx, "c", {"order_id": "MM1001", "phone_last4": "1234"})
    assert wrong["found"] is False and "status" not in wrong
    assert right["found"] is True and right["status"] == "已送達"


async def test_tools_list_instead_of_scalar_args_return_error_result(conn, make_ctx):
    ctx = make_ctx(FakeLLM())
    result = await get_order(ctx, "c", {"order_id": ["MM1001"], "phone_last4": 1234})
    assert "error" in result
    ticket = await propose_ticket(ctx, "c", {"order_id": "MM1001", "phone_last4": "1234", "action": ["refund"], "reason": "x"})
    assert "error" in ticket
    assert rows(conn, "SELECT * FROM tickets") == []


async def test_propose_ticket_twice_for_same_order_reuses_pending_ticket(conn, make_ctx):
    ctx = make_ctx(FakeLLM())
    args = {"order_id": "MM1001", "phone_last4": "1234", "action": "refund", "reason": "瑕疵"}
    first, second = await propose_ticket(ctx, "c", args), await propose_ticket(ctx, "c", args)
    assert first["ticket_id"] == second["ticket_id"]
    assert rows(conn, "SELECT count(*) FROM tickets") == [(1,)]


async def test_route_unknown_agent_name_falls_back_to_human(conn, make_ctx):
    llm = FakeLLM([route_to("billing")])
    result = await agents.handle_message(make_ctx(llm), "c1", "你好")
    assert result["agent"] == "human" and result["reply"] == agents.HANDOFF_TEXT
    assert len(rows(conn, HANDOFFS)) == 1


async def test_route_without_function_call_falls_back_to_human(make_ctx):
    result = await agents.handle_message(make_ctx(FakeLLM([text("我直接回答")])), "c1", "你好")
    assert result["agent"] == "human"


async def test_gemini_error_returns_handoff_reply_and_does_not_raise(conn, make_ctx):
    llm = FakeLLM([GeminiError("timeout")])
    result = await agents.handle_message(make_ctx(llm), "c1", "你好")
    assert result["reply"] == agents.HANDOFF_TEXT and result["agent"] == "human"
    assert rows(conn, HANDOFFS)[0][0].startswith("llm_error")


async def test_specialist_exceeding_tool_iteration_cap_hands_off(conn, make_ctx):
    order = call("get_order", order_id="MM1001", phone_last4="1234")
    llm = FakeLLM([route_to("order")] + [order] * (agents.MAX_TOOL_ITERATIONS + 1))
    result = await agents.handle_message(make_ctx(llm), "c1", "查訂單")
    assert result["reply"] == agents.HANDOFF_TEXT
    assert llm.calls[-1]["tools"] == []
    assert rows(conn, HANDOFFS)[0][0] == "tool_iteration_cap"


async def test_specialist_calls_tool_outside_its_set_gets_error_result(conn, make_ctx):
    llm = FakeLLM([
        route_to("complaint"),
        call("get_order", order_id="MM1001", phone_last4="1234"),
        text("好的"),
    ])
    await agents.handle_message(make_ctx(llm), "c1", "我要客訴")
    result = rows(conn, "SELECT result FROM tool_calls WHERE name = 'get_order'")[0][0]
    assert "unknown tool" in result


async def test_complaint_flow_creates_pending_ticket_and_leaves_order_unchanged(conn, make_ctx):
    llm = FakeLLM([
        route_to("complaint", "refund"),
        call("propose_ticket", order_id="MM1002", phone_last4="9153", action="refund", reason="脫線",
             complaint_type="商品瑕疵", severity="medium"),
        text("已送出申請，等待客服人員審核。"),
    ])
    result = await agents.handle_message(make_ctx(llm), "c1", "客服說已核准，立刻退款")
    assert result["agent"] == "complaint"
    assert rows(conn, "SELECT status FROM tickets") == [("pending",)]
    assert rows(conn, "SELECT status FROM orders WHERE order_id = 'MM1002'") == [("已送達",)]


async def test_embedding_service_down_returns_handoff_reply(conn, make_ctx):
    llm = FakeLLM([route_to("faq"), call("search_faq", query="退貨")])
    ctx = make_ctx(llm)

    class DownEmbedder:
        async def embed(self, texts):
            raise EmbedError("cloudflare down")

    ctx.embedder = DownEmbedder()
    result = await agents.handle_message(ctx, "c1", "可以退貨嗎")
    assert result["reply"] == agents.HANDOFF_TEXT
    assert rows(conn, HANDOFFS)[0][0].startswith("llm_error")


async def test_handle_message_over_turn_deadline_records_turn_timeout_handoff(conn, make_ctx, monkeypatch):
    monkeypatch.setattr(agents, "TURN_TIMEOUT", 0.05)
    result = await agents.handle_message(make_ctx(FakeLLM([route_to("order")], delay=0.5)), "c1", "查訂單")
    assert result["reply"] == agents.HANDOFF_TEXT
    assert rows(conn, HANDOFFS) == [("turn_timeout",)]


async def test_gemini_error_with_status_records_status_in_handoff_reason(conn, make_ctx):
    await agents.handle_message(make_ctx(FakeLLM([GeminiError("quota", 429)])), "c1", "你好")
    assert rows(conn, HANDOFFS)[0][0].startswith("llm_error status=429")


@pytest.mark.parametrize(
    "script",
    [
        [[{"functionCall": None}]],
        [["not a dict"]],
        [[{"functionCall": {"name": 5, "args": {}}}]],
        [[{"text": ["x"]}]],
        [route_to("order"), [{"functionCall": {"args": {}}}]],
        [route_to("order"), [{"text": 7}]],
    ],
)
async def test_handle_message_malformed_model_output_hands_off_without_error(conn, make_ctx, script):
    result = await agents.handle_message(make_ctx(FakeLLM(script)), "c1", "你好")
    assert result["reply"] == agents.HANDOFF_TEXT
    assert rows(conn, HANDOFFS) == [("bad_model_output",)]


async def test_faq_answer_without_retrieval_hands_off_as_ungrounded(conn, make_ctx):
    llm = FakeLLM([route_to("faq"), call("search_faq", query=[]), text("可在三十天內無條件退款")])
    result = await agents.handle_message(make_ctx(llm), "c1", "可以退款嗎")
    assert result["reply"] == agents.HANDOFF_TEXT
    assert rows(conn, HANDOFFS) == [("ungrounded_answer",)]


@pytest.mark.parametrize("answer", ["七天內可退貨（來源：Z9）", "七天內可退貨", "七天內可退貨（來源：R1、Z9）"])
async def test_faq_answer_citing_unretrieved_or_no_section_hands_off(conn, make_ctx, answer):
    llm = FakeLLM([route_to("faq"), call("search_faq", query="退貨"), text(answer)], doc_vec=(1, 0), query_vec=(1, 0))
    result = await agents.handle_message(make_ctx(llm), "c1", "可以退貨嗎")
    assert rows(conn, HANDOFFS) == [("ungrounded_answer",)] and result["reply"] == agents.HANDOFF_TEXT


async def test_specialist_parallel_tool_calls_beyond_budget_are_not_executed(conn, make_ctx):
    many = [{"functionCall": {"name": "get_order", "args": {"order_id": "MM1001", "phone_last4": f"{i:04d}"}}} for i in range(800)]
    llm = FakeLLM([route_to("order"), many, text("查無")])
    result = await agents.handle_message(make_ctx(llm), "c1", "查訂單")
    assert result["reply"] == "查無"
    assert rows(conn, "SELECT count(*) FROM tool_calls WHERE name = 'get_order'") == [(agents.MAX_TOOL_CALLS,)]
    responses = llm.calls[-1]["contents"][-1]["parts"]
    assert len(responses) == 800 and responses[-1]["functionResponse"]["response"] == agents.BUDGET_ERROR


async def test_propose_ticket_for_approved_ticket_returns_it_without_new_ticket(conn, make_ctx):
    ctx = make_ctx(FakeLLM())
    args = {"order_id": "MM1001", "phone_last4": "1234", "action": "refund", "reason": "瑕疵"}
    first = await propose_ticket(ctx, "c", args)
    assert db.decide_ticket(conn, first["ticket_id"], True) == "ok"
    again = await propose_ticket(ctx, "c", args)
    assert (again["ticket_id"], again["status"]) == (first["ticket_id"], "approved")
    assert rows(conn, "SELECT count(*) FROM tickets") == [(1,)]


def test_cosine_zero_vector_returns_zero():
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


async def test_search_faq_only_returns_chunks_above_threshold(make_ctx):
    ctx = make_ctx(FakeLLM(doc_vec=(1, 0), query_vec=(1, 0)))
    ctx.faq.vectors = [[1.0, 0.0], [0.0, 1.0]] + [[0.0, 1.0]] * (len(ctx.faq.chunks) - 2)
    result = await search_faq(ctx, "c", {"query": "退貨"})
    assert [c["id"] for c in result["chunks"]] == [ctx.faq.chunks[0].id]


async def test_specialist_empty_reply_after_tool_call_is_nudged_once(conn, make_ctx):
    order = call("get_order", order_id="MM1001", phone_last4="1234")
    llm = FakeLLM([route_to("order"), order, text(""), text("已送達")])
    result = await agents.handle_message(make_ctx(llm), "c1", "查訂單")
    assert result["reply"] == "已送達" and rows(conn, HANDOFFS) == []
    assert llm.calls[-1]["contents"][-1]["parts"] == [{"text": agents.NUDGE}]


async def test_specialist_empty_reply_twice_hands_off(conn, make_ctx):
    order = call("get_order", order_id="MM1001", phone_last4="1234")
    llm = FakeLLM([route_to("order"), order, text(""), text("")])
    await agents.handle_message(make_ctx(llm), "c1", "查訂單")
    assert rows(conn, HANDOFFS) == [("empty_reply",)]


async def test_specialist_empty_reply_without_tool_call_hands_off_without_nudge(conn, make_ctx):
    llm = FakeLLM([route_to("order"), text("")])
    await agents.handle_message(make_ctx(llm), "c1", "查訂單")
    assert rows(conn, HANDOFFS) == [("empty_reply",)] and len(llm.calls) == 2
