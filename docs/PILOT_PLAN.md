# Pilot plan for Grand Mobile

How the Support Copilot can go from the hackathon to production **without risk to players**, and how Grand Mobile can see, with its own
numbers, whether it is worth keeping. Every figure below is measured on the supplied data (sources in `docs/REPORT.md`); what is not
measured yet is said explicitly.

## What it gives the support team

| Benefit | Evidence today |
|---|---|
| **Critical tickets never wait in the queue.** Payments, hacked or lost accounts, threats in real life, mass failures are p0 and always go to a person | 0 errors on every such ticket found in 119,337 real tickets (2,541 in a 20k train sample, 157 in VALID): none left without a human, none below p0, none refused as «invalid» (`make audit`) |
| **Every draft cites only real rules** (an invented rule zeroes a ticket in the case's own evaluation) | 0 invalid citations on 1,000 VALID + 20,000 train tickets; citation = the admin's own rule in 91 % of comparable cases |
| **Less typing.** A ready draft in the player's language (ru, kk, uz, en, es, pt), with the rule, the missing evidence and the next step | 80 % of VALID drafts are grounded in a rule or a defined game term |
| **Consistent categories and priorities** | agreement with the rule codes admins wrote: 0.90 on VALID (weak labels, not accuracy: see below) |
| **Moderation help** for insults written in mixed languages and alphabets («сен котак», «qotaq», «фак ю», «с.у.к.а») with who said what | 54 of 57 artificial adversarial cases; 21 findings on 1,000 real VALID tickets, reviewed by hand |
| **Repeat and related complaints** shown together; duplicates of one author flagged | duplicate + related-complaint logic tested; precision/recall against real duplicates unknown |
| **Practically free and instant** | template mode: $0 per ticket, HTTP p95 ≈ 17 ms on 4 CPUs; the optional LLM mode is budgeted to ≤ $0.002 per ticket |

## What is NOT known yet (and how the pilot measures it)

* **Real accuracy** of category and priority: no human labels exist in the data. In the pilot every admin decision becomes a label
  (`POST /api/feedback` or the «Решение администратора» block), and `GET /api/feedback/summary` / the dashboard show live category accuracy,
  priority macro-F1, the most frequent corrections and accuracy per language.
* **How admins rate the drafts** (the case's primary metric): the same feedback carries a 1–5 rating of the draft.
* **Kazakh and Uzbek wording**: templates are being reviewed by native speakers (`Kazakh_texts_review.xlsx`).

## Rollout in three steps

| Step | Duration | What the copilot does | Players see | Exit criterion |
|---|---|---|---|---|
| **1. Shadow** | 2 weeks | Node.js backend calls `POST /triage` for every new ticket and stores the result; admins work as today and send their final decisions to `POST /api/feedback` | nothing changes | ≥ 1,000 decisions; category accuracy and priority macro-F1 known per category and language; 0 critical tickets below p0 |
| **2. Assist** | 4 weeks | admins see category, priority, reasons, cited rules, moderation analysis and the editable draft in their tool; p0 tickets go to the top of the queue | replies written faster by admins (still by admins) | median handling time per ticket ↓, draft rating ≥ 4 on ≥ 60 % of drafts, no increase in complaints about support |
| **3. Partial automation** (optional) | after review | only for categories where the measured accuracy is high AND the draft needs no decision (off-topic, «please resubmit with evidence», answers taken word for word from a verified rule) — with a sample re-checked by a person every day | an instant clarifying reply | error rate on the daily sample below a threshold agreed with support leads; any doubt → back to step 2 |

Never automated, by design: payments, account access, appeals, complaints about admins, threats, decisions on punishment.

## KPIs on the dashboard

Live from the service: tickets per category/priority/language, share sent to humans, open p0 tickets, duplicates, latency (also in
`GET /metrics` for Prometheus/Grafana), category accuracy and draft rating from admin decisions. Suggested business KPIs for the pilot:
median time to first admin reply, time to first reply for p0, admin minutes per 100 tickets, share of drafts sent with ≤ 1 edit.

## Integration effort for Grand Mobile

* **Backend:** one HTTP call per ticket (`POST /triage`, example in `docs/INTEGRATION.md`), one call per admin decision (`POST /api/feedback`).
  Batch endpoint (`POST /triage/batch`) for backlogs. Timeouts and fallbacks: if the copilot is down the ticket goes to the normal human queue.
* **Ops:** one Docker container, a volume with the rules, the model and SQLite; `/health`, `/metrics`, JSON access logs without ticket
  text, role tokens (`service`, `viewer`, `admin`), rate limit.
* **Data:** nothing leaves the service except the response; the optional LLM sends only ordinary (non-sensitive) tickets, and only if
  enabled. The model is retrained locally with `make train`, now also on the admins' decisions.
* **Rules change?** Replace `rules.json`; citations are re-verified against the new text automatically (a paraphrase is used only if its
  anchor words are still in the rule), so an outdated paraphrase is dropped rather than quoted.

## Risks and how they are handled

| Risk | Mitigation |
|---|---|
| Wrong category or priority | the admin always sees and can correct it; corrections are measured and retrain the model |
| A draft promises something | sensitive topics never get free text; a guard rejects promises, transfers, deadlines, invented numbers; every draft is reviewed by a person |
| Invented rule | citations only from `rules.json`, existence + relevance + applicability checks; 0 invalid citations measured |
| Toxic words in unknown slang | moderation analysis is a helper, not a decision; misses are recorded and the lexicon is extended from admin feedback |
| Service unavailable | the backend falls back to the normal queue (see `docs/INTEGRATION.md`) |
