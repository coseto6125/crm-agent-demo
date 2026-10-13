import os

import aiohttp

from crm_agent.gemini import GeminiError, json_object

EMBED_MODEL = "@cf/google/embeddinggemma-300m"
URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/" + EMBED_MODEL


class EmbedError(GeminiError):
    # a subclass so that embedding failures take the same handoff path as LLM errors
    """The embedding service is unreachable, returned an error or a malformed response."""


def vectors_from(data: dict, expected: int) -> list[list[float]]:
    """Extract result.data and check it holds one list of numbers per input text."""
    try:
        vectors = data["result"]["data"]
    except (KeyError, TypeError) as e:
        raise EmbedError("response has no result.data") from e
    if not (
        isinstance(vectors, list)
        and len(vectors) == expected
        and all(isinstance(v, list) and v and all(isinstance(x, int | float) for x in v) for v in vectors)
    ):
        raise EmbedError(f"result.data is not {expected} lists of numbers")
    return vectors


class CloudflareEmbedder:
    """Embeddings from Workers AI (a Google model hosted by Cloudflare)."""

    def __init__(self, account_id: str | None = None, token: str | None = None, timeout: float = 30.0):
        self.account_id = account_id if account_id is not None else os.environ.get("CF_ACCOUNT_ID", "")
        self.token = token if token is not None else os.environ.get("CF_API_TOKEN", "")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not (self.account_id and self.token):
            raise EmbedError("CF_ACCOUNT_ID or CF_API_TOKEN is not set")
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session, session.post(
                URL.format(account=self.account_id),
                json={"text": texts},
                headers={"Authorization": f"Bearer {self.token}"},
            ) as resp:
                status, raw = resp.status, await resp.text()
        except (aiohttp.ClientError, TimeoutError) as e:
            raise EmbedError(f"request failed: {e!r}") from e
        if status != 200:
            raise EmbedError(f"HTTP {status}: {raw[:200]}", status)
        data = json_object(raw, status, EmbedError)
        if not data.get("success"):
            raise EmbedError(f"HTTP {status}: {str(data.get('errors'))[:200]}", status)
        return vectors_from(data, len(texts))
