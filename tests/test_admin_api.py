import threading

from conftest import FakeLLM, call, route_to, text
from crm_agent import db

ORDER = {"order_id": "MM1005", "phone_last4": "2910"}


async def chat(app, message: str, conv: str = "c1"):
    _, resp = await app.asgi_client.post("/api/chat", json={"conversation_id": conv, "message": message})
    return resp


def make_ticket_app(make_app, action: str = "refund"):
    llm = FakeLLM([
        route_to("complaint"),
        call("propose_ticket", **ORDER, action=action, reason="瑕疵", complaint_type="商品瑕疵", severity="high"),
        text("已送出申請，等待客服人員審核。"),
    ])
    return make_app(llm)


def order_status(app, order_id: str = "MM1005") -> str:
    return app.ctx.agent.conn.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,)).fetchone()[0]


async def test_approve_pending_ticket_updates_order_and_notifies_customer(make_app, admin):
    app = make_ticket_app(make_app)
    await chat(app, "我要退款")
    assert order_status(app) == "已送達"
    _, resp = await app.asgi_client.post("/admin/tickets/1/approve", headers=admin)
    assert resp.status == 200
    assert order_status(app) == "已退款"
    notes = app.ctx.agent.conn.execute("SELECT content FROM messages WHERE role = 'system'").fetchall()
    assert len(notes) == 1 and "已退款" in notes[0][0]


async def test_approve_twice_returns_409_and_does_not_repeat_action(make_app, admin):
    app = make_ticket_app(make_app)
    await chat(app, "我要退款")
    await app.asgi_client.post("/admin/tickets/1/approve", headers=admin)
    app.ctx.agent.conn.execute("UPDATE orders SET status = '已送達' WHERE order_id = 'MM1005'")
    _, resp = await app.asgi_client.post("/admin/tickets/1/approve", headers=admin)
    assert resp.status == 409
    assert order_status(app) == "已送達"
    assert app.ctx.agent.conn.execute("SELECT count(*) FROM messages WHERE role = 'system'").fetchone()[0] == 1


def test_decide_ticket_from_two_connections_succeeds_once_per_ticket(tmp_path):
    path = str(tmp_path / "race.db")
    conns = [db.connect(path), db.connect(path)]
    ids = []
    for n in range(50):
        cur = conns[0].execute(
            "INSERT INTO tickets (conversation_id, order_id, action, reason, created_at) VALUES (?,?,?,?,0)",
            (f"c{n}", f"MM{1001 + n % 20}", "refund", "x"),
        )
        ids.append(cur.lastrowid)
    barrier, results, errors = threading.Barrier(2), {i: [] for i in ids}, []

    def worker(conn):
        try:
            for tid in ids:
                barrier.wait()
                results[tid].append(db.decide_ticket(conn, tid, True))
        except Exception as e:
            errors.append(e)
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(c,)) for c in conns]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert all(sorted(r) == ["conflict", "ok"] for r in results.values())
    notes = conns[0].execute("SELECT conversation_id, count(*) FROM messages WHERE role = 'system' GROUP BY 1").fetchall()
    assert len(notes) == 50 and all(n[1] == 1 for n in notes)
    [c.close() for c in conns]


async def test_reject_pending_ticket_leaves_order_unchanged(make_app, admin):
    app = make_ticket_app(make_app)
    await chat(app, "我要退款")
    _, resp = await app.asgi_client.post("/admin/tickets/1/reject", headers=admin)
    assert resp.status == 200
    assert order_status(app) == "已送達"
    _, again = await app.asgi_client.post("/admin/tickets/1/approve", headers=admin)
    assert again.status == 409


async def test_approve_unknown_ticket_returns_404(make_app, admin):
    _, resp = await make_app().asgi_client.post("/admin/tickets/99/approve", headers=admin)
    assert resp.status == 404


async def test_admin_endpoints_without_or_with_wrong_token_are_rejected(make_app, monkeypatch):
    app = make_ticket_app(make_app)
    await chat(app, "我要退款")
    _, missing = await app.asgi_client.post("/admin/tickets/1/approve")
    _, wrong = await app.asgi_client.post("/admin/tickets/1/approve", headers={"X-Admin-Token": "nope"})
    _, data = await app.asgi_client.get("/admin/data", headers={"X-Admin-Token": "nope"})
    assert (missing.status, wrong.status, data.status) == (401, 403, 403)
    assert order_status(app) == "已送達"
    monkeypatch.delenv("ADMIN_TOKEN")
    _, unset = await app.asgi_client.post("/admin/tickets/1/approve", headers={"X-Admin-Token": ""})
    _, unset2 = await app.asgi_client.post("/admin/tickets/1/approve", headers={"X-Admin-Token": "x"})
    assert (unset.status, unset2.status) == (401, 403)


async def test_admin_data_lists_turns_tool_calls_and_pending_tickets(make_app, admin):
    app = make_ticket_app(make_app)
    await chat(app, "我要退款")
    _, resp = await app.asgi_client.get("/admin/data", headers=admin)
    body = resp.json
    turn = body["conversations"][0]["turns"][0]
    assert turn["agent"] == "complaint"
    assert [c["name"] for c in turn["tool_calls"]] == ["route", "propose_ticket"]
    assert body["tickets"][0]["status"] == "pending"


async def test_chat_empty_or_whitespace_message_returns_400(make_app):
    app = make_app()
    for body in ({"message": ""}, {"message": "  \n "}, {"message": 5}, {}):
        _, resp = await app.asgi_client.post("/api/chat", json=body)
        assert resp.status == 400


async def test_chat_unknown_conversation_id_creates_conversation(make_app):
    app = make_app(FakeLLM([route_to("human")]))
    resp = await chat(app, "你好", conv="brand-new-id")
    assert resp.status == 200 and resp.json["conversation_id"] == "brand-new-id"
    _, bad = await app.asgi_client.post("/api/chat", json={"conversation_id": "a b/c", "message": "hi"})
    assert bad.status == 400


async def test_chat_conversation_id_with_trailing_newline_returns_400(make_app):
    _, resp = await make_app().asgi_client.post("/api/chat", json={"conversation_id": "abc\n", "message": "hi"})
    assert resp.status == 400


async def test_admin_token_with_non_ascii_characters_returns_403(make_app):
    _, resp = await make_app().asgi_client.post("/admin/tickets/1/approve", headers={"X-Admin-Token": "tök".encode()})
    assert resp.status == 403
