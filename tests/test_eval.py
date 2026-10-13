from conftest import FakeLLM, call, route_to, text
from crm_agent import agents
from crm_agent import eval as ev
from crm_agent.gemini import GeminiError


async def run(make_ctx, script, expect, message="你好", **llm_kw):
    ctx = make_ctx(FakeLLM(script, **llm_kw))
    result = await agents.handle_message(ctx, "c1", message)
    return ev.classify(ctx, "c1", result, expect)[0]


async def test_classify_expected_handoff_via_routed_to_human_passes(make_ctx):
    assert await run(make_ctx, [route_to("human")], {"handoff": True}) == "PASS"


async def test_classify_expected_handoff_caused_by_dependency_error_fails(make_ctx):
    assert await run(make_ctx, [GeminiError("down", 500)], {"handoff": True}) == "FAIL"


async def test_classify_rate_limited_before_routing_is_skipped(make_ctx):
    assert await run(make_ctx, [GeminiError("quota", 429)], {"agent": "faq"}) == "SKIP"


async def test_classify_rate_limit_after_wrong_agent_routed_fails(make_ctx):
    assert await run(make_ctx, [route_to("order"), GeminiError("quota", 429)], {"agent": "faq"}) == "FAIL"


async def test_classify_rate_limit_after_correct_agent_routed_is_skipped(make_ctx):
    assert await run(make_ctx, [route_to("order"), GeminiError("quota", 429)], {"agent": "order"}) == "SKIP"


async def test_classify_unexpected_ungrounded_handoff_for_faq_case_fails(make_ctx):
    script = [route_to("faq"), call("search_faq", query="x"), text("沒有來源的答案")]
    assert await run(make_ctx, script, {"agent": "faq", "cites": "R1"}, doc_vec=(1, 0), query_vec=(1, 0)) == "FAIL"


async def test_preload_embedding_failure_does_not_raise(make_ctx):
    ctx = make_ctx(FakeLLM())

    class Down:
        async def embed(self, texts):
            raise GeminiError("down")

    ctx.embedder = Down()
    await ev.preload(ctx)
    assert ctx.faq.vectors is None
