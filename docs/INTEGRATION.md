# Integration with the Grand Mobile backend

The Support Copilot is a standalone HTTP service. It needs no access to the game, its servers or databases: the game backend (Node.js)
sends a ticket, the service returns the triage result. Nothing is ever sent to a player by the service.

```
launcher / in-game report ──► Node.js backend ──POST /triage──► Support Copilot ──► JSON result
                                   │                                (category, priority, draft, needs_human, reasons)
                                   ├─► MariaDB: store the result next to the ticket (admin queue of the game)
                                   └─► ClickHouse: analytics (category, priority, latency, needs_human), no player text needed
```

## 1. Deploy

```bash
docker build -t support-copilot .
docker run -d -p 8000:8000 -v /srv/copilot/data:/app/data --env-file .env support-copilot
```
`/app/data` holds `handout/rules.json`, `handout/taxonomy.json`, the model (`models/complaint_clf.joblib`, built by `make train`) and the SQLite queue.
Health check: `GET /health` (returns the auth mode, the classifier, the model's training date, rule count and configuration warnings).

## 2. Configure (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `API_TOKENS` | empty = auth OFF (explicit demo mode) | `service:<token>,viewer:<token>,admin:<token>`; tokens ≥ 16 characters. **Set it before exposing the service.** A set but invalid value (unknown role, short token) stops the start-up instead of turning authentication off |
| `RATE_LIMIT_PER_MIN` | 0 in demo mode, 1200 with `API_TOKENS` | units per minute per VERIFIED token (otherwise per client IP; an invented header gets no counter) on `/triage` and `/api/*`; `/triage/batch` counts each ticket; 0 = off. In-process: with several replicas, limit at the gateway |
| `HUMAN_CONF_MIN` / `HUMAN_AMBIGUOUS_MIN` | 0.4 / 0.5 | confidence thresholds below which a ticket always goes to a human |
| `USE_MODEL`, `MODEL_PATH` | 1, `data/models/complaint_clf.joblib` | learned complaint classifier (absent = lexicon only) |
| `RULES_PATH`, `TAXONOMY_PATH`, `DB_PATH`, `FAQ_PATH` | under `data/` | data files |
| `ANTHROPIC_API_KEY` + `LLM_*` | empty | optional LLM drafting (see `.env.example`); without it, verified templates |
| `PERSIST_TRIAGE` | 0 | 1 = every `/triage` call is also stored in the admin queue of this service |

## 3. Call it from Node.js

```js
// Node 18+ (global fetch). The service token has the role "service": it may call /triage only.
const COPILOT_URL = process.env.COPILOT_URL;            // e.g. http://support-copilot:8000
const COPILOT_TOKEN = process.env.COPILOT_TOKEN;

async function triage(ticket) {
  const body = {
    kind: ticket.kind,                                   // "complaint" | "report"
    messages: ticket.messages.map(m => ({ role: m.role, text: m.text, attachments: m.imagesCount ?? 0 })),
    // optional context: enables duplicate detection, related complaints and better extraction
    ticket_id: ticket.id, created_at: ticket.createdAt, reporter_id: ticket.reporterHash,
    reported_id: ticket.reportedHash, reported_nick: ticket.reportedNick, rule_refs_raw: ticket.ruleRefsRaw,
    server_lang: ticket.serverLang, region: ticket.region,
  };
  try {
    const res = await fetch(`${COPILOT_URL}/triage`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${COPILOT_TOKEN}`, "x-request-id": String(ticket.id) },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(6000),                 // p95 target is 5 s; the template mode answers in milliseconds
    });
    if (res.status === 429) return fallback(ticket, "rate limited");
    if (!res.ok) return fallback(ticket, `copilot HTTP ${res.status}`);
    return await res.json();                             // store it; show draft_reply.text to the admin, never send it automatically
  } catch (err) {
    return fallback(ticket, err.name);                   // service down or timeout: the ticket simply goes to the normal human queue
  }
}

function fallback(ticket, why) {
  return { needs_human: true, priority: null, category: null, reasons: [`copilot unavailable: ${why}`] };
}
```

Rules for the integration:
* **Treat `draft_reply.text` as a draft.** The service never sends anything to a player; an administrator approves or edits it.
* **`needs_human: true` must be honoured** (payments, ban appeals, account compromise, threats, low confidence, complaints that need a decision).
* **Idempotency:** the result depends only on the ticket (and, for duplicates, on earlier tickets of the same author); repeating a call is safe.
  Duplicate detection needs `reporter_id` and `created_at`.
* **Errors:** `422` = invalid body (fix the caller), `401/403` = token/role, `429` = back off (`Retry-After`), `5xx`/timeout = route to humans.
* **Correlation:** the response carries `X-Request-ID` (yours or generated), `X-Triage-Mode`, `X-Latency-Ms`, `X-Cost-Usd`; the access log is one JSON
  line per request (id, path, status, latency, role) and never contains ticket text.

## 4. Send the administrators' decisions back (the learning loop)

```js
// after an admin closes a ticket in the game's admin tool
await fetch(`${COPILOT_URL}/api/feedback`, {
  method: "POST",
  headers: { "content-type": "application/json", authorization: `Bearer ${COPILOT_TOKEN}` },
  body: JSON.stringify({
    ticket_id: String(ticket.id), kind: ticket.kind, messages: [{ role: "player", text: ticket.text }],
    predicted_category: triage.category, predicted_priority: triage.priority,   // what /triage returned (optional)
    final_category: admin.category, final_priority: admin.priority,             // the admin's decision (codes from taxonomy.json)
    final_reply: admin.replyText, draft_rating: admin.rating,                   // optional; rating 1..5
  }),
});
```
The latest decision per `ticket_id` wins. `GET /api/feedback/summary` (viewer) returns the live category accuracy, priority macro-F1,
the most frequent corrections, accuracy per language and the mean draft rating; `make train` adds these decisions to the training data
(weight 3 against the weak labels). This is how the service gets *measured* and better in production instead of on a hackathon guess.

## 5. Batch

For backfills or offline evaluation: `make predict IN=tickets.jsonl OUT=pred.jsonl` (same pipeline as the API; accepts the dataset's record format,
drops administrator answers before processing; writes `pred.jsonl.stats.json` with p50/p95 latency and cost).

## 6. Monitoring

`GET /metrics` (viewer token) in the Prometheus text format: `copilot_http_requests_total{route,status}`, `copilot_triage_total{dimension,value}`
(category, priority, lang, needs_human, spam_or_duplicate, mode, llm) and the `copilot_triage_latency_ms` histogram. `POST /triage/batch`
takes up to 200 tickets (processed in time order, results in request order, one failing ticket does not fail the batch).

## 7. Admin panel

`/admin` (viewer/admin token): dashboard, queue, ticket card with the reasons, rules, moderation analysis (languages by segment, highlighted
abusive words and who said them), editable draft, local send/edit/escalate, edits exported as training feedback at `/api/edits` (admin).
In production the game's own admin tools would read the stored results instead; this panel shows what such a screen needs.

## 8. What is NOT included
* No user accounts or SSO: static role tokens. Put the service behind the company's gateway/SSO for per-person audit.
* The rate limit and the SQLite queue are per process; several replicas need a shared store (MariaDB) and a gateway limit.
* No automatic sending to players, by design.
