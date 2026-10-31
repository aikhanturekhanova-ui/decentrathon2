# Grand Mobile Support Copilot

Triage service for player complaints and in-game reports (Decentrathon 6.0 × Grand Mobile, case 03).
It detects the language, category and priority, spots spam/duplicates, extracts facts, retrieves the relevant
rules, writes a **draft** reply in the player's language with verified rule citations, and says when (and why)
a human is needed. It is standalone: no access to the game, its servers, APIs or databases is assumed.

> **Status in one line:** the required core (`POST /triage`, `make predict`, RAG over rules, safety checks, report) is implemented and tested, including a third
> independent review (mandatory handoff and priority, place-bound rules, honest «needs review» wording, duplicate history, LLM budget).
> The complaint category is a **hybrid of the lexicon and a learned model** (`make train`): silver agreement on VALID 0.84 → 0.90 (weak labels, not accuracy).
> Accuracy on a hidden test is **unknown**: there are no human labels yet (tools for them are included).
> LLM mode (optional): mock-tested for both providers; ONE real run on artificial tickets exists (the author's, Gemini free tier, before the current
> closed-vocabulary guard, cost unknown); real player tickets are never sent to a model. Real latency, cost and acceptance of the current code are **not measured**.
> A green test suite is not proof of correctness: see [docs/LIMITATIONS.md](docs/LIMITATIONS.md) and [docs/REQUIREMENTS_TRACE.md](docs/REQUIREMENTS_TRACE.md).
> **GitHub Actions shows red for a reason outside the code:** the jobs do not start («recent account payments have failed or your spending limit needs to be increased», an account billing limit). The tests were run locally; see the trace.

## Quick start

```bash
git clone <this repo> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate        # Windows: py -3 -m venv .venv ; .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                       # optional; works without any key
```

### Data (not in this repository, by the hackathon's terms)
Put the organizers' files into `data/handout/` (the folder is git-ignored). `rules.json` is also searched next to the input file of `make predict`, in `./handout/` and in the working directory; without it the service still runs but cites nothing and says so (`reasons`, a warning on stderr):
```
data/handout/rules.json        # required
data/handout/taxonomy.json     # optional (a built-in copy of the 14 category codes is used otherwise)
data/handout/tickets_valid.jsonl, tickets_train.jsonl.gz   # only needed for scripts/evaluation
```
Tests do **not** need them (they use an artificial rulebook in `tests/fixtures/`); tests marked with the real files are skipped when absent.

### Run
```bash
make train                     # learned complaint classifier from TRAIN (~3 min; data/models/, git-ignored). Optional: without it the lexicon decides alone
make test                      # pytest (hermetic: never uses the locally trained model)
make run                       # http://localhost:8000  (docs: /docs, admin panel: /admin)
make predict IN=tickets.jsonl OUT=pred.jsonl   # trains the model once first if it is missing and data/handout/tickets_train.jsonl.gz exists
```
No `make` on Windows? Use the underlying commands:
```powershell
python scripts/train_classifier.py        # with PYTHONPATH=. (set PYTHONPATH=. on Windows)
python -m pytest
python -m uvicorn app.main:app --port 8000
python -m app.cli predict --in tickets.jsonl --out pred.jsonl
```

### Try the contract
```bash
curl -s localhost:8000/triage -H 'content-type: application/json' -d '{
  "kind":"complaint",
  "messages":[{"role":"player","text":"Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП","attachments":2}]}'
```
`make predict` accepts both compact bodies like the one above and **full dataset records** (it drops the administrator's answer and
`verdict/punishment/violation` before anything is processed: `app/adapter.py`). Each output line has the contract fields plus `ticket_id`;
`pred.jsonl.stats.json` gets p50/p95, mode and cost.

Optional knowledge base for recurring game questions: `data/kb/faq.json` (format in `app/faq.py`; every answer needs a `source`, only `reviewed: true` answers remove the human handoff). It is empty by default: no verified material about game mechanics was provided.

### Modes (explicit, never faked)
| Mode | When | What happens |
|---|---|---|
| `template` (default) | no model key, or `LLM_ENABLED=0` | deterministic, citation-safe templates; **no model is called**; `/health` and the UI banner say "DEMO" |
| `llm` | `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` set and `LLM_ENABLED=1` | one model call (Anthropic Messages API or Google Gemini `generateContent`) per **ordinary** ticket writes the draft text from a fact sheet (payments, hacking, lost access, appeals, threats, mass failures, spam and injection attempts never reach the model: verified templates only). The model never writes statements about rules: it places `[[R1]]` markers and the code inserts the verified paraphrase. The text must be in the player's language and pass the output guard and a structural check (no promises, no claimed transfer, no sanction/refund topic without negation, no deadlines, no numbers or contacts that are not in the ticket), otherwise the template is used and the reason is written to `reasons`. The guard is a finite list of patterns, **not** a proof of safety. A time budget (`LLM_TOTAL_BUDGET_S`) and, when prices are set, a per-ticket cost ceiling apply. Needs an **API key** (a Claude.ai Pro subscription does not include API access). Model/prices are env variables; cost is counted per call, unknown prices give an unknown cost (never $0) |

**Honest status table, requirement by requirement, with what was actually run:** [docs/REQUIREMENTS_TRACE.md](docs/REQUIREMENTS_TRACE.md).

## What it does (pipeline)
`adapter → normalize game symbols (ќ→қ …) → language → flags/priority (lexicon) + complaint category (lexicon ⊕ learned model) → dedup/spam/related complaints → RAG (curated rule candidates + BM25, anchor check) → draft (templates or LLM) → guard → response`.
Details and the decisions behind them: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Contradictions in the materials, assumptions and questions for the organizers: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

**Measured with Gemini (`gemini-3.5-flash-lite`, free tier, 18 artificial demo tickets × 2, 2026-10-31):** 17 of 28 model drafts passed every
check (60.7 %; the rest fell back to the verified template), whole-ticket latency p50/p95 1.0/1.4 s, 743/102 tokens in/out per call
(`docs/metrics/llm_smoke.json`). Sensitive tickets never reach the model. **A free LLM tier may use the submitted text for training, so it is
used only with artificial tickets; every evaluation script on real tickets forces `LLM_ENABLED=0`.** Cost per ticket = tokens × the price of your
model (`LLM_PRICE_*`): not assumed here.

## Admin panel
`/admin` opens on the **Дашборд**: workload tiles (queue, needs-human, open p0, spam/duplicates, low confidence, edits), clickable p0 alerts, priority/language/category
distributions of the queue and historical patterns from TRAIN (`make insights` → `docs/metrics/history.json`, aggregates only: verdicts by category, frequent
punishments, repeat behaviour). Then: queue with filters (category, priority, language, needs-human, status, text), ticket card (original text, extracted fields, reasons for handoff,
cited rules with their section, editable draft). Buttons: **«Править и сохранить»** (the edit is stored as feedback in SQLite; export at `/api/edits`), **«Отправить (только локально, игроку не уйдёт)»** (the game is not connected, so nothing is delivered and the UI says so), **«Эскалировать»**. «Шаблоны» shows reply templates proposed from clusters of similar tickets.
`make demo` / the "Демо-данные" button loads 18 artificial tickets. In «Новое обращение» the author id and the accused nick are optional: duplicates are checked only when they are given
(the acceptance time is used as the ticket time); nobody is assigned a shared author, because that would merge strangers' tickets. The duplicate history of the HTTP service survives a restart (SQLite).

Bonus «Шаблоны»: proposals for new reply templates from clusters of similar tickets (admin edits win over auto drafts); review-only.

## Security, integration, moderation
* **Access control:** `API_TOKENS=service:…,viewer:…,admin:…` (roles: the game backend calls `/triage`, viewers read, admins act). Empty = demo mode with
  authentication off, said by `/health` and at startup; a set but INVALID `API_TOKENS` stops the start-up (never «auth off» by mistake). Rate limit per verified token or IP (`RATE_LIMIT_PER_MIN`; off by default in demo mode, a batch counts each ticket), one JSON access-log line per request without ticket text,
  `X-Request-ID` correlation, configurable confidence thresholds (`HUMAN_CONF_MIN`, `HUMAN_AMBIGUOUS_MIN`).
* **Node.js integration, deployment, error handling:** [docs/INTEGRATION.md](docs/INTEGRATION.md). **Pilot plan for Grand Mobile** (shadow → assist →
  partial automation, KPIs, measured numbers): [docs/PILOT_PLAN.md](docs/PILOT_PLAN.md).
* **Learning loop:** the game's admin tool sends final decisions to `POST /api/feedback` (or the admin uses «Решение администратора» in the card):
  live accuracy at `GET /api/feedback/summary` and on the dashboard; `make train` uses them as human labels. `POST /triage/batch` for backlogs,
  `GET /metrics` for Prometheus.
* **Moderation analysis** (`app/toxicity.py`, `app/langseg.py`): languages per segment of a mixed message; abusive language in ru/kk/uz/en/es/pt and
  transliteration, robust to «с.у.к.а», «с у к а», «сууука», «3ае6ал», «cyka», zero-width characters; types (profanity, insult, family, threat, hate, sexual);
  WHO said it (quoted/reported by the author vs the author's own words vs the accused's comment), jokes, game vs real-life threats; evidence spans highlighted
  in the ticket card and in the «Модерация» tool. Mixed alphabets are handled by a phonetic skeleton (қотақ / котак / qotaq / kotak → one stem;
  «фак ю», «blyad», «пiзда»), and each finding says when a word is written in another language's alphabet. A reporting victim is never treated as the offender; profanity never raises the priority by itself.
  Adversarial suite: `python scripts/multilingual_eval.py` (artificial cases, FP/FN listed, known limitations recorded) → `docs/metrics/multilingual.json`.

## Telegram bot (optional)
`TELEGRAM_ENABLED=1`, `TELEGRAM_BOT_TOKEN=...` in `.env`, then `python -m app.telegram_bot`. It is a labelled demo bot that stores messages in the same queue and never claims that staff received them.

## Evaluation and report
```bash
make eval            # data profile + silver agreement (local data needed)
make bench           # HTTP latency p50/p95
make review-set      # CSVs for human annotation (docs/ANNOTATION_GUIDE.md), then: make score-review
make coverage        # behaviour counters on real tickets: substantive answers, escalations, citation validity, fallbacks, kk/uz, sensitive-topic errors
make insights        # aggregated historical patterns for the dashboard (TRAIN only) -> docs/metrics/history.json
make faq-candidates  # SUGGESTED answers for recurring questions from TRAIN admin replies (local, with provenance; never used automatically)
make audit           # all real tickets + fuzz: shape AND meaning checks (about 15 minutes)
make report          # docs/REPORT.md from docs/metrics/*.json
```
Windows without `make`: `set PYTHONPATH=.` and `python scripts\coverage_report.py` (same for the other scripts).
Read [docs/REPORT.md](docs/REPORT.md) for what is measured and what is not.

## Privacy
The repository contains **no** tickets, no derived samples, no logs, no indexes with their content, no secrets (`.gitignore`; check with `git ls-files`).
Only aggregated numbers (`docs/metrics/`) and artificial examples are committed. `rules.json` and `taxonomy.json` are loaded locally: their publication terms are unclear
(question for the organizers in `docs/LIMITATIONS.md`). Rule paraphrases in `app/phrases.py` are our own wording and are only used if the real rule text still contains
the expected anchor words. The data must be deleted within 14 days after the winners are announced.

## Repository map
`app/` service · `tests/` pytest + artificial fixtures · `scripts/` evaluation tools · `docs/` report, architecture, limitations, demo script, presentation · `Makefile`, `Dockerfile`, `.env.example`, `requirements*.txt`.
Models, libraries, sources and licenses: [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).
