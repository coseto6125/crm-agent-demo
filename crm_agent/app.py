import hmac
import json
import os
import re
import uuid
from pathlib import Path

import msgspec
from sanic import Request, Sanic, file
from sanic import json as json_response

from crm_agent import agents, db
from crm_agent.cf_embed import CloudflareEmbedder
from crm_agent.gemini import Gemini, GeminiError
from crm_agent.tools import Ctx, FaqIndex

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONV_ID = re.compile(r"[\w-]{1,64}")
MAX_MESSAGE_CHARS = 1000


class ChatRequest(msgspec.Struct):
    message: str
    conversation_id: str = ""


def create_app(db_path: str = "crm.db", llm=None, embedder=None, close_on_stop: bool = True) -> Sanic:
    app = Sanic("CrmAgent")
    app.config.RESPONSE_TIMEOUT = 150
    app.ctx.agent = Ctx(db.connect(db_path), llm or Gemini(), embedder or CloudflareEmbedder(), FaqIndex.load())
    conn = app.ctx.agent.conn

    @app.before_server_start
    async def embed_faq(app: Sanic):
        try:
            await app.ctx.agent.faq.build(app.ctx.agent.embedder)
        except GeminiError as e:
            print(f"FAQ embedding at startup failed, will retry on first search: {e}")

    if close_on_stop:
        app.after_server_stop(lambda app: app.ctx.agent.conn.close())

    def check_admin(request: Request):
        """Return an error response when the admin token is missing or wrong, else None."""
        if not (token := request.headers.get("X-Admin-Token")):
            return json_response({"error": "missing X-Admin-Token"}, status=401)
        expected = os.environ.get("ADMIN_TOKEN", "")
        if not expected or not hmac.compare_digest(token.encode(), expected.encode()):
            return json_response({"error": "forbidden"}, status=403)
        return None

    @app.get("/")
    async def chat_page(request: Request):
        return await file(STATIC_DIR / "chat.html")

    @app.get("/admin")
    async def admin_page(request: Request):
        return await file(STATIC_DIR / "admin.html")

    @app.post("/api/chat")
    async def chat(request: Request):
        try:
            req = msgspec.json.decode(request.body, type=ChatRequest)
        except msgspec.DecodeError:
            return json_response({"error": "body must be {conversation_id?, message}"}, status=400)
        if not (text := req.message.strip()):
            return json_response({"error": "message is empty"}, status=400)
        if len(text) > MAX_MESSAGE_CHARS:
            return json_response({"error": f"message over {MAX_MESSAGE_CHARS} characters"}, status=400)
        conv_id = req.conversation_id or uuid.uuid4().hex
        if not CONV_ID.fullmatch(conv_id):
            return json_response({"error": "invalid conversation_id"}, status=400)
        return json_response(await agents.handle_message(app.ctx.agent, conv_id, text))

    async def decide(request: Request, ticket_id: int, approved: bool):
        if (denied := check_admin(request)) is not None:
            return denied
        match db.decide_ticket(conn, ticket_id, approved):
            case "ok":
                return json_response({"ticket_id": ticket_id, "status": "approved" if approved else "rejected"})
            case "conflict":
                return json_response({"error": "ticket already decided"}, status=409)
            case _:
                return json_response({"error": "ticket not found"}, status=404)

    @app.post("/admin/tickets/<ticket_id:int>/approve")
    async def approve(request: Request, ticket_id: int):
        return await decide(request, ticket_id, True)

    @app.post("/admin/tickets/<ticket_id:int>/reject")
    async def reject(request: Request, ticket_id: int):
        return await decide(request, ticket_id, False)

    @app.get("/admin/data")
    async def admin_data(request: Request):
        if (denied := check_admin(request)) is not None:
            return denied
        rows = lambda sql: [dict(r) for r in conn.execute(sql)]
        calls: dict[int, list] = {}
        for c in rows("SELECT * FROM tool_calls ORDER BY id"):
            calls.setdefault(c["turn_id"], []).append(
                {"name": c["name"], "args": json.loads(c["args"]), "result": json.loads(c["result"])}
            )
        convs = {r["id"]: {"id": r["id"], "messages": [], "turns": []} for r in rows("SELECT * FROM conversations ORDER BY created_at DESC")}
        for m in rows("SELECT * FROM messages ORDER BY id"):
            convs[m["conversation_id"]]["messages"].append({"id": m["id"], "role": m["role"], "content": m["content"]})
        for t in rows("SELECT * FROM turns ORDER BY id"):
            convs[t["conversation_id"]]["turns"].append({**t, "tool_calls": calls.get(t["id"], [])})
        return json_response({
            "conversations": list(convs.values()),
            "tickets": rows("SELECT * FROM tickets ORDER BY id DESC"),
            "handoffs": rows("SELECT * FROM handoffs ORDER BY id DESC"),
        })

    return app
