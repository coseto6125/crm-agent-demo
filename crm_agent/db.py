import json
import sqlite3
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, customer TEXT NOT NULL, phone TEXT NOT NULL,
    items TEXT NOT NULL, amount INTEGER NOT NULL, status TEXT NOT NULL,
    carrier TEXT, tracking_no TEXT, last_event TEXT);
CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
    role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
    message_id INTEGER NOT NULL, agent TEXT, intent TEXT);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, turn_id INTEGER NOT NULL,
    name TEXT NOT NULL, args TEXT NOT NULL, result TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
    order_id TEXT NOT NULL, action TEXT NOT NULL, reason TEXT NOT NULL,
    complaint_type TEXT, severity TEXT, status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL, decided_at REAL);
CREATE TABLE IF NOT EXISTS handoffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
    turn_id INTEGER, reason TEXT NOT NULL, created_at REAL NOT NULL);
"""

ACTION_STATUS = {"return": "已退貨", "exchange": "換貨處理中", "refund": "已退款"}
ACTION_LABEL = {"return": "退貨", "exchange": "換貨", "refund": "退款"}

PRODUCTS: dict[str, dict] = {
    p["product_id"]: p for p in json.loads((DATA_DIR / "products.json").read_text("utf-8"))
}


def connect(path: str) -> sqlite3.Connection:
    """Open the database in autocommit mode, create tables and seed orders once."""
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    for o in json.loads((DATA_DIR / "orders.json").read_text("utf-8")):
        ship = o["shipment"] or {}
        conn.execute(
            "INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
            (o["order_id"], o["customer"], o["phone"], json.dumps(o["items"]), o["amount"],
             o["status"], ship.get("carrier"), ship.get("tracking_no"), ship.get("last_event")),
        )
    return conn


def ensure_conversation(conn: sqlite3.Connection, conv_id: str) -> None:
    conn.execute("INSERT OR IGNORE INTO conversations VALUES (?,?)", (conv_id, time.time()))


def add_message(conn: sqlite3.Connection, conv_id: str, role: str, content: str) -> int:
    cur = conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?,?,?,?)",
        (conv_id, role, content, time.time()),
    )
    return cur.lastrowid


def log_tool_call(conn: sqlite3.Connection, turn_id: int, name: str, args: dict, result: dict) -> None:
    conn.execute(
        "INSERT INTO tool_calls (turn_id, name, args, result) VALUES (?,?,?,?)",
        (turn_id, name, json.dumps(args, ensure_ascii=False), json.dumps(result, ensure_ascii=False)),
    )


def record_handoff(conn: sqlite3.Connection, conv_id: str, turn_id: int | None, reason: str) -> None:
    conn.execute(
        "INSERT INTO handoffs (conversation_id, turn_id, reason, created_at) VALUES (?,?,?,?)",
        (conv_id, turn_id, reason, time.time()),
    )


def find_order(conn: sqlite3.Connection, order_id: str, phone_last4: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM orders WHERE order_id = ? AND substr(phone, -4) = ?", (order_id, phone_last4)
    ).fetchone()


def decide_ticket(conn: sqlite3.Connection, ticket_id: int, approve: bool) -> str:
    """
    Approve or reject a pending ticket; the only code path that changes an order.

    Returns "ok", "conflict" (already decided) or "notfound". The conditional UPDATE
    is the gate: a second call, or a concurrent one, matches no row.
    """
    new_status = "approved" if approve else "rejected"
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute(
            "UPDATE tickets SET status = ?, decided_at = ? WHERE id = ? AND status = 'pending'",
            (new_status, time.time(), ticket_id),
        )
        if cur.rowcount == 0:
            conn.execute("ROLLBACK")
            found = conn.execute("SELECT 1 FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            return "conflict" if found else "notfound"
        t = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        label = ACTION_LABEL[t["action"]]
        if approve:
            status = ACTION_STATUS[t["action"]]
            conn.execute("UPDATE orders SET status = ? WHERE order_id = ?", (status, t["order_id"]))
            note = f"您的{label}申請（編號 {ticket_id}）已由客服人員核准，訂單 {t['order_id']} 狀態更新為「{status}」。"
        else:
            note = f"您的{label}申請（編號 {ticket_id}）經客服人員審核後未通過，訂單 {t['order_id']} 維持原狀，如有疑問請再與我們聯絡。"
        add_message(conn, t["conversation_id"], "system", note)
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return "ok"
