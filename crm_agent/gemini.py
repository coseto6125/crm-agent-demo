import asyncio
import json
import os
import re

import aiohttp

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
CHAT_MODEL = "gemini-3.1-flash-lite"
THINKING_LEVEL = "low"
ROUTE_THINKING_LEVEL = "minimal"
RETRY_STATUS = {429, 500, 503}
ATTEMPTS = 4
MAX_RETRY_WAIT = 65
RETRY_DELAY = re.compile(r'"retryDelay":\s*"(\d+)s"')


class GeminiError(Exception):
    """The Gemini API is unreachable, slow, or returned an error or a malformed response."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def json_object(raw: str, status: int, exc: type[GeminiError] = GeminiError) -> dict:
    """Parse a response body that must be a JSON object, or raise exc."""
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise exc(f"HTTP {status}: body is not JSON", status) from e
    if not isinstance(data, dict):
        raise exc(f"HTTP {status}: body is not a JSON object", status)
    return data


class Gemini:
    """Minimal REST client for generateContent with function calling."""

    def __init__(self, api_key: str | None = None, temperature: float = 0.2, timeout: float = 30.0):
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self.temperature = temperature
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def _post(self, endpoint: str, body: dict) -> dict:
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY is not set")
        for attempt in range(ATTEMPTS):
            last = attempt == ATTEMPTS - 1
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session, session.post(
                    f"{BASE_URL}/{endpoint}", json=body, headers={"x-goog-api-key": self.api_key}
                ) as resp:
                    status, raw = resp.status, await resp.text()
            except (aiohttp.ClientError, TimeoutError) as e:
                if last:
                    raise GeminiError(f"request failed: {e!r}") from e
                wait = 2**attempt
            else:
                if status == 200:
                    return json_object(raw, status)
                if status not in RETRY_STATUS or last:
                    raise GeminiError(f"HTTP {status}: {raw[:200]}", status)
                hint = RETRY_DELAY.search(raw)
                wait = int(hint[1]) + 1 if hint else 2**attempt
                if wait > MAX_RETRY_WAIT:
                    raise GeminiError(f"HTTP {status}, retry after {wait}s: {raw[:200]}", status)
            await asyncio.sleep(wait)
        return None

    async def generate(
        self, contents: list[dict], system: str, tools: list[dict] | None = None,
        force: str | None = None, thinking: str = THINKING_LEVEL,
    ) -> list[dict]:
        """
        Return the parts of the first candidate.

        Args:
            force: name of a function the model must call.
            thinking: thinkingLevel (minimal, low, medium or high).
        """
        body = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": self.temperature, "thinkingConfig": {"thinkingLevel": thinking}},
        }
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]
            if force:
                body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [force]}}
        data = await self._post(f"{CHAT_MODEL}:generateContent", body)
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as e:
            raise GeminiError("response has no candidates[0].content.parts") from e
        if not isinstance(parts, list):
            raise GeminiError("response parts is not a list")
        return parts
