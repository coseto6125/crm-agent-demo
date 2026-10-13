import asyncio

import pytest
from crm_agent import db
from crm_agent.app import create_app
from crm_agent.tools import Ctx, FaqIndex
from sanic import Sanic

Sanic.test_mode = True
TOKEN = "secret-token"


class FakeLLM:
    """Scripted stand-in for the Gemini client: generate pops one scripted reply per call."""

    def __init__(self, script: list | None = None, doc_vec=(0.0, 1.0), query_vec=(1.0, 0.0), delay: float = 0.0):
        self.script = list(script or [])
        self.calls: list[dict] = []
        self.embed_calls = 0
        self.delay = delay
        self.doc_vec, self.query_vec = list(doc_vec), list(query_vec)

    async def generate(self, contents, system, tools=None, force=None, thinking=None):
        self.calls.append({"contents": list(contents), "thinking": thinking, "tools": [t["name"] for t in tools or []], "force": force})
        if self.delay:
            await asyncio.sleep(self.delay)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def embed(self, texts):
        self.embed_calls += 1
        vec = self.doc_vec if self.embed_calls == 1 else self.query_vec
        return [vec for _ in texts]


def call(name: str, **args) -> list[dict]:
    return [{"functionCall": {"name": name, "args": args}}]


def text(s: str) -> list[dict]:
    return [{"text": s}]


def route_to(agent: str, intent: str = "x") -> list[dict]:
    return call("route", agent=agent, intent=intent)


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(str(tmp_path / "t.db"))
    yield connection
    connection.close()


@pytest.fixture
def make_ctx(conn):
    return lambda llm: Ctx(conn, llm, llm, FaqIndex.load())


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)

    apps = []

    def build(llm=None):
        llm = llm or FakeLLM()
        apps.append(create_app(str(tmp_path / f"app{len(apps)}.db"), llm, llm, close_on_stop=False))
        return apps[-1]

    yield build
    for app in apps:
        app.ctx.agent.conn.close()


@pytest.fixture
def admin():
    return {"X-Admin-Token": TOKEN}
