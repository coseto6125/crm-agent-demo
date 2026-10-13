# 木棉衣櫥 客服代理人 Demo

## 架構

```
對話頁 -> 編排代理人 -> FAQ / 訂單 / 客訴 代理人 -> 工具層 -> SQLite
```

- 對話頁：`GET /`，呼叫 `POST /api/chat`。
- 編排代理人：每則訊息呼叫一次 `route(agent, intent)`，交給一個專責代理人。
- FAQ 代理人：`search_faq`，embedding 檢索後依段落回答並標註來源。
- 訂單代理人：`get_order`、`get_shipment`。
- 客訴代理人：`propose_ticket`，只建立待審申請單。
- 審核：`POST /admin/tickets/<id>/approve|reject`，需 `X-Admin-Token`；訂單狀態只在核准時改變。
- 後台：`GET /admin`，開啟後在輸入框貼上 `ADMIN_TOKEN`。
- 上限：每則訊息 1 次分流；每個專責代理人每則訊息最多 4 輪、共執行 4 次工具呼叫，超過的呼叫不執行；整則訊息逾 120 秒轉真人。

## 啟動方式

```
uv sync
cp .env.example .env     # 填入 GEMINI_API_KEY、CF_ACCOUNT_ID、CF_API_TOKEN、ADMIN_TOKEN
make dev        # http://127.0.0.1:8000/
make test       # 不連網
```

## 評測

```
make eval       # 20 個案例，逐案輸出 PASS / FAIL / SKIP 與 ACCURACY
```

## 使用的模型

- `gemini-3.1-flash-lite`（Google，對話與路由）
- `@cf/google/embeddinggemma-300m`（Google 模型，Cloudflare Workers AI 代管，embedding 檢索）
