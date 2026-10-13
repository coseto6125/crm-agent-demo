import pytest
from crm_agent.cf_embed import CloudflareEmbedder, EmbedError, vectors_from
from crm_agent.gemini import Gemini, GeminiError, json_object


async def test_embed_without_credentials_raises_embed_error():
    with pytest.raises(EmbedError):
        await CloudflareEmbedder(account_id="", token="").embed(["x"])


@pytest.mark.parametrize("data", [{}, {"result": None}, {"result": {}}, {"result": {"data": "x"}},
                                  {"result": {"data": [[1.0], [2.0]]}}, {"result": {"data": [["a"]]}},
                                  {"result": {"data": [[]]}}, {"result": {"data": [None]}}])
def test_vectors_from_malformed_result_raises_embed_error(data):
    with pytest.raises(EmbedError):
        vectors_from(data, 1)


def test_vectors_from_valid_result_returns_vectors():
    assert vectors_from({"result": {"data": [[0.1, 2]]}}, 1) == [[0.1, 2]]


@pytest.mark.parametrize("raw", ["", "not json", "[1]", "null", '"x"'])
def test_json_object_non_object_body_raises_with_status(raw):
    with pytest.raises(EmbedError) as info:
        json_object(raw, 502, EmbedError)
    assert info.value.status == 502


@pytest.mark.parametrize("data", [{}, {"candidates": None}, {"candidates": []}, {"candidates": [None]},
                                  {"candidates": [{}]}, {"candidates": [{"content": {}}]},
                                  {"candidates": [{"content": {"parts": "x"}}]}])
async def test_generate_malformed_response_raises_gemini_error(data):
    llm = Gemini(api_key="k")

    async def fake_post(endpoint, body):
        return data

    llm._post = fake_post
    with pytest.raises(GeminiError):
        await llm.generate([], "s")
