import heapq
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from operator import itemgetter

from crm_agent import db

FAQ_MIN_SIMILARITY = 0.45
FAQ_TOP_K = 3
SECTION_SPLIT = re.compile(r"(?m)^## ")

ACTIONS = list(db.ACTION_STATUS)
COMPLAINT_TYPES = ["商品瑕疵", "尺寸不合", "配送問題", "服務態度", "其他"]
SEVERITIES = ["low", "medium", "high"]


class Handoff(Exception):
    """Stop the turn and hand the customer to a human agent."""


def _decl(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "OBJECT", "properties": properties, "required": required},
    }


def _str(description: str, enum: list[str] | None = None) -> dict:
    return {"type": "STRING", "description": description} | ({"enum": enum} if enum else {})


ORDER_KEYS = {"order_id": _str("訂單編號，例如 MM1001"), "phone_last4": _str("下單手機號碼末四碼")}

ROUTE_DECL = _decl(
    "route",
    "把顧客訊息交給一個專責代理人處理",
    {
        "agent": _str("要交給的代理人", ["faq", "order", "complaint", "human"]),
        "intent": _str("用一個短詞標示顧客意圖"),
    },
    ["agent", "intent"],
)
SEARCH_FAQ_DECL = _decl("search_faq", "以語意搜尋常見問題文件", {"query": _str("搜尋用的完整問句")}, ["query"])
GET_ORDER_DECL = _decl("get_order", "查詢訂單內容與狀態", ORDER_KEYS, list(ORDER_KEYS))
GET_SHIPMENT_DECL = _decl("get_shipment", "查詢訂單的物流資訊", ORDER_KEYS, list(ORDER_KEYS))
PROPOSE_TICKET_DECL = _decl(
    "propose_ticket",
    "為顧客建立一張待客服人員審核的退換貨或退款申請單，不會直接執行任何處理",
    ORDER_KEYS
    | {
        "action": _str("顧客想要的處理方式", ACTIONS),
        "reason": _str("申請原因摘要"),
        "complaint_type": _str("客訴類型", COMPLAINT_TYPES),
        "severity": _str("嚴重程度", SEVERITIES),
    },
    [*ORDER_KEYS, "action", "reason", "complaint_type", "severity"],
)

FAQ_TOOLS = [SEARCH_FAQ_DECL]
ORDER_TOOLS = [GET_ORDER_DECL, GET_SHIPMENT_DECL]
COMPLAINT_TOOLS = [PROPOSE_TICKET_DECL]


@dataclass
class Chunk:
    id: str
    title: str
    text: str


@dataclass
class FaqIndex:
    """FAQ sections as chunks; embeddings are computed once (at startup or on first search) and kept in memory."""

    chunks: list[Chunk]
    vectors: list[list[float]] | None = field(default=None, repr=False)

    @classmethod
    def load(cls) -> "FaqIndex":
        chunks = []
        for section in SECTION_SPLIT.split((db.DATA_DIR / "faq.md").read_text("utf-8"))[1:]:
            head, _, body = section.partition("\n")
            chunk_id, _, title = head.partition(" ")
            chunks.append(Chunk(chunk_id, title.strip(), body.strip()))
        return cls(chunks)

    async def build(self, embedder) -> None:
        self.vectors = await embedder.embed([f"{c.title}\n{c.text}" for c in self.chunks])

    async def search(self, embedder, query: str) -> list[tuple[float, Chunk]]:
        if self.vectors is None:
            await self.build(embedder)
        [q] = await embedder.embed([query])
        scored = [(cosine(q, v), c) for v, c in zip(self.vectors, self.chunks, strict=True)]
        return heapq.nlargest(FAQ_TOP_K, scored, key=itemgetter(0))


def cosine(a: list[float], b: list[float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    return math.sumprod(a, b) / norm if norm else 0.0


@dataclass
class Ctx:
    conn: sqlite3.Connection
    llm: object
    embedder: object
    faq: FaqIndex


def _text(args: dict, key: str) -> str | None:
    """A non-empty string argument, or None (the model may send a list, a number or nothing)."""
    v = args.get(key)
    return v.strip() if isinstance(v, str) and v.strip() else None


def _bad_args(*keys: str) -> dict:
    return {"error": f"參數 {', '.join(keys)} 必須是非空字串"}


NOT_FOUND = {"found": False, "message": "查無此訂單，或電話末四碼不符"}


async def search_faq(ctx: Ctx, conv_id: str, args: dict) -> dict:
    if (query := _text(args, "query")) is None:
        return _bad_args("query")
    hits = await ctx.faq.search(ctx.embedder, query)
    if (best := hits[0][0]) < FAQ_MIN_SIMILARITY:
        raise Handoff(f"faq_low_similarity best={best:.3f}")
    return {
        "chunks": [
            {"id": c.id, "title": c.title, "text": c.text, "score": round(s, 3)}
            for s, c in hits if s >= FAQ_MIN_SIMILARITY
        ]
    }


def _order_row(ctx: Ctx, args: dict) -> sqlite3.Row | dict | None:
    oid, last4 = _text(args, "order_id"), _text(args, "phone_last4")
    if oid is None or last4 is None:
        return _bad_args("order_id", "phone_last4")
    return db.find_order(ctx.conn, oid, last4)


async def get_order(ctx: Ctx, conv_id: str, args: dict) -> dict:
    row = _order_row(ctx, args)
    if isinstance(row, dict):
        return row
    if row is None:
        return NOT_FOUND
    items = [
        {"name": db.PRODUCTS.get(i["product_id"], {}).get("name", i["product_id"]), "qty": i["qty"]}
        for i in json.loads(row["items"])
    ]
    return {"found": True, "order_id": row["order_id"], "status": row["status"], "amount": row["amount"], "items": items}


async def get_shipment(ctx: Ctx, conv_id: str, args: dict) -> dict:
    row = _order_row(ctx, args)
    if isinstance(row, dict):
        return row
    if row is None:
        return NOT_FOUND
    if row["carrier"] is None:
        return {"found": True, "order_id": row["order_id"], "status": row["status"], "shipped": False}
    return {
        "found": True, "order_id": row["order_id"], "status": row["status"], "shipped": True,
        "carrier": row["carrier"], "tracking_no": row["tracking_no"], "last_event": row["last_event"],
    }


async def propose_ticket(ctx: Ctx, conv_id: str, args: dict) -> dict:
    row = _order_row(ctx, args)
    if isinstance(row, dict):
        return row
    if row is None:
        return NOT_FOUND
    action, reason = _text(args, "action"), _text(args, "reason")
    ctype, severity = _text(args, "complaint_type"), _text(args, "severity")
    if action not in ACTIONS or reason is None:
        return {"error": f"action 必須是 {ACTIONS} 之一，reason 必須是非空字串"}
    ctype = ctype if ctype in COMPLAINT_TYPES else "其他"
    severity = severity if severity in SEVERITIES else "medium"
    if dup := ctx.conn.execute(
        "SELECT id, status FROM tickets WHERE order_id = ? AND action = ? AND status IN ('pending', 'approved')",
        (row["order_id"], action),
    ).fetchone():
        return {"ticket_id": dup["id"], "status": dup["status"], "message": "已有相同的申請單，不重複建立"}
    cur = ctx.conn.execute(
        "INSERT INTO tickets (conversation_id, order_id, action, reason, complaint_type, severity, created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (conv_id, row["order_id"], action, reason, ctype, severity, time.time()),
    )
    return {"ticket_id": cur.lastrowid, "status": "pending", "message": "申請單已建立，需等待客服人員審核，尚未執行任何處理"}


HANDLERS = {
    "search_faq": search_faq,
    "get_order": get_order,
    "get_shipment": get_shipment,
    "propose_ticket": propose_ticket,
}
