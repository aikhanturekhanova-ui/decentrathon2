# Architecture

```
tickets.jsonl / POST /triage / admin UI / Telegram
        │
   app/adapter.py      full record -> request. Drops admin messages and verdict/punishment/violation/category/priority
        │
   app/pipeline.py  TriageService.triage()   (the ONLY place where decisions are made; API, CLI, UI, bot all call it)
        ├─ textutil     game-symbol normalization (original kept), language detection (ru kk uz en es pt lat und), names/quotes ignored
        ├─ langseg      language per segment of a mixed message (reasons + moderation panel)
        ├─ toxicity     abusive language with obfuscation handling, speaker/direction, jokes, game vs real-life threats, evidence spans
        ├─ classify     lexicon: flags (appeal, hack, access, payment, threat, mass bug, injection, evidence), priority, extraction;
        │               complaint category = lexicon blended with the learned model (app/ml.py, `make train`) when it is present
        ├─ dedup        spam heuristics; duplicates = same author + same target + same event (see below); related complaints
        │               (other authors, same accused, 24 h); history in SQLite
        ├─ rag + knowledge   RuleBook(rules.json): section-aware ids, BM25, curated candidates, anchor check, place/role context check
        ├─ drafts + phrases  templates in ru/en/kk/uz/es/pt (no promises, no claimed transfer, "according to you", attachments not viewed)
        ├─ llm (optional)    rewrites the draft for ORDINARY tickets only, under a time and cost budget; any failure -> template
        └─ guard        citations, promises, transfer claims, invented facts (numbers, deadlines, contacts, procedures), structural free-text check
        │
   app/main.py (FastAPI) · app/cli.py (batch) · app/store.py (SQLite queue + edits) · app/web/index.html
```

## Decisions and why

| Decision | Reason |
|---|---|
| Hybrid complaint classifier: lexicon + learned model | The data has no category labels, but ~29.5k TRAIN complaints carry a rule code or an appeal verdict in the admin reply (weak «silver» labels). A TF-IDF (char 2–5 + word 1–2) logistic regression trained on them, blended 0.6/0.4 with the lexicon scores, raised silver agreement on VALID from 0.84 to 0.90 and on a reporter-grouped TRAIN hold-out from 0.86 to 0.92 (`docs/metrics/ml_classifier.json`). The model's weight shrinks when most words are unseen (e.g. Kazakh), so a clear lexicon match is not overruled by character fragments. Safety flags, priority, handoff, `complaint.invalid` and report categories stay rule-based: the silver labels say nothing about them. The model file is built locally (`make train`, ~3 min) and never committed (its vocabulary comes from the confidential tickets). |
| Reports stay lexicon-only | Admin replies to reports carry no usable category signal, so there is nothing to learn from without inventing labels. |
| Cite only curated (category/topic → rule) candidates that pass an anchor check **and** a context check | `rules.json` repeats refs across sections (57 refs are used in >1 section); an invented or off-topic point zeroes the ticket. BM25 alone cited an unrelated point (ЖБ 1.5.2), so retrieval is used for ranking, claim verification and the LLM context, **never as the sole reason to cite**. A place-bound rule (Green Zone) needs the place as a fact of the story (see "Applicability" below). |
| Citation format `2.1`, `GZ 1.1`, `ЖБ 1.2.7` | Keeps the contract (`"cited": ["2.1"]`) and resolves the section ambiguity using the codes the dataset itself uses («ОП», «GZ», «ЖБ»). |
| Game terms (DM, DB, SK, TK, RK, MG, PG, NonRP) have no numbers | They are named in the text («Из раздела правил «RP термины»: DM — …») and listed in `draft_reply.terms`, never in `cited`: no number is ever invented for them. |
| Player's rule numbers are claims | They go to `extracted.rule_refs` only if they exist; they are quoted only if they fit the ticket; unknown numbers go to `rule_refs_unverified`; mismatches appear in `reasons`. A typed «GZ 1.2» never proves the place. |
| LLM only writes text for ordinary tickets | Category, priority, handoff and allowed citations are deterministic. Sensitive tickets never reach the model. |
| Template mode is a first-class mode | Zero cost, zero latency risk, always safe; it is also the fallback. UI/health state clearly when no model is used. |
| SQLite, vanilla JS UI, no extra services | Fastest path to a working demo; nothing to deploy. |
| Embeddings not used | No measurement shows a benefit over curated candidates + BM25 on this tiny slang text, and they would add a dependency. Not claimed as a feature. |

## Policies (each has regression tests in `tests/test_review3*.py`)

### Mandatory handoff and sensitive topics
The flags `appeal`, `hack`, `access` (lost access to one's own account), `payment`, `threat_outside_game`, `mass_bug` are computed from the whole text of the
player's messages **before and independently of** the category, the ticket kind, the attachments, the rule answer or the FAQ. A ticket with any of them has
`needs_human = true`, whatever else was found; `payment`, `hack`, `access`, `threat_outside_game`, `mass_bug` also force `priority = p0`.
Pure questions («Как оплатить донат?», «Что такое донат?») are not refund incidents; a refund request hidden behind a question («Верните 500 рублей за покупку.
Можно ли подать жалобу?») is. «Оплатить штрафы/налоги» (in-game payments) is not a payment for real money.
A `report` whose main category would be question/help/offtopic becomes `report.payment` / `report.account` when such a flag is present.
Ordinary safe questions («Можно ли оскорблять в чате?», «Что такое DM?») keep `needs_human = false` (tests prove both directions).

### Priority
Evaluated in a fixed order and only ever **raised**: (1) base priority of the category; (2) `complaint.scam`: p1 for a big amount/valuable property, else p2;
(3) mandatory raises to p0 (flags above). Evidence never changes the priority. **Resolved contradiction:** the taxonomy text says «complaints with proof → p1»,
the PDF example (insult + 2 screenshots) is p2 and 99 % of complaints carry screenshots, so the category decides; this is an interpretation, not a requirement.

### Evidence
Four separate facts: attachments (a count), a claim («скрины есть», a link), an explicit absence («доказательств нет»), a question («какие доказательства нужны?»).
`extracted.evidence_status` ∈ attached | claimed_only | stated_absent | asked_about | none; `has_evidence` is true for attached and claimed_only;
`evidence_verified` is always false (the content of images is never available). If attachments exist but the text says «нет доказательств», the attachments win
and `reasons` reports the conflict. A negation covers its own clause only («видео нет, но скрины есть» → claimed_only). A complaint without any evidence is
`complaint.invalid` (taxonomy), **except** tickets with a sensitive topic: they are never refused with «подайте заново».

### Account takeover wordings
Detected in the whole text of the player (not only the first words) with a takeover verb («угнали/украли/взломали» + account/«учётная запись»/«учётка»/«профиль»), a foreign person next to the player's own account
(«посторонний человек вошёл в мой аккаунт», «к моему профилю привязали чужую почту», «мне сменили почту на аккаунте») and the Kazakh/Uzbek/English/Spanish/Portuguese equivalents. The verb «бұзды/buzdi/buzishdi»
(«broke») means hacking only next to the word «account» (otherwise it is «violated a rule»). A message that also asks «что такое DM?» stays `report.account`, `p0`, human.
The detector is a lexicon: a wording nobody wrote down can be missed (`docs/LIMITATIONS.md`).

### Accused player
`reported_nick` (a field the game fills for complaints) wins. Otherwise the complainant («я, X», «мой ник X», «меня зовут X»), the administrator («админ X») are excluded, a nick after an accusation word
(«жалуюсь на X», «на игрока X») or before a violation verb is preferred, and with several candidates and no marker the field stays empty instead of guessing.

### Appeals that arrive as a report
`appeal` raises the priority to at least `p2` (as `complaint.admin`) and the draft asks for the data an appeal needs (administrator's nickname, time, server, screenshot of the penalty) instead of «no ready answer, what interests you?».
The category of such a report stays what the lexicon found (usually `report.question`): the taxonomy has no report-side appeal category.

### Sensitive-topic drafts
No refusal for missing evidence, no «передано», no outcome. The draft says that a person needs to review it (not that it was passed on: nothing is forwarded,
`PERSIST_TRIAGE=0` stores nothing, nobody is assigned), asks for the facts a person needs, warns not to send passwords/card numbers, and for a threat outside the
game adds a generic safety sentence (local emergency services; no invented number). The guard rejects any sentence that claims a transfer.

### Applicability of rules (existence ≠ relevance ≠ applicability)
1. existence: the point resolves in `rules.json` (section-aware); 2. relevance: its real text contains the anchor words; 3. applicability: a curated trigger fits
the story AND a place-bound point has its three facts (`rag.zone_facts`, `rag.context_ok`): **the place**, **the forbidden action** and **the action not denied**.
* Place: a Green Zone named as such («ЗЗ», «зелёная зона») or an object of the official list that `rules.json` itself contains (section 32, points 2.x; parsed at load time, 40 objects in the
  real file). A city-bound object («Больница г. Арзамас», «Полиция г. Южный») needs its city; «больница» alone or «больница г. Москва» is not established and the draft asks for the city and the object.
  «Банки», «Автосалоны» and the like are listed without a city and count. The moment of the story matters: «вышли из ЗЗ, после чего он меня похитил», «вне ЗЗ», «сначала были в ЗЗ, потом…»
  do not establish the place of the event (the safe direction: nothing is cited, the draft asks). A typed «GZ 1.2» never counts as a place.
* Action: `GZ 1.1` needs killing/damage, `GZ 1.2` needs kidnapping/robbery/ropes. «Связал меня в ЗЗ, убийства не было» cites only `GZ 1.2`; «убил меня в ЗЗ, связывания не было» only `GZ 1.1`.
  A question («что такое ЗЗ?») describes no event, so no facts are required there.
* Unknown or another place («в лесу», «на улице») is never assumed to be a Green Zone. The clarifying question asks only for what is missing: no place at all → where; a listed object without its city → which
  city/object; a stated place that is not in the list → nothing (it is already known not to be a zone).
Special sections for leaders, organisations, families, OPG, wars are not in the curated candidates at all, so they cannot be cited by mistake.
For a `report` about a violation the rules are retrieved by the substance of the violation (`Analysis.topic`), not by the generic category.

### Duplicates
Same author **and** same accused player **and** inside 60 minutes, and the two tickets do not contradict each other in event details (place, server, clock time).
Confident duplicate (`is_spam_or_duplicate = true`): almost the same text (Jaccard ≥ 0.8), or the same category with a shared detail, or a clear paraphrase
(Jaccard ≥ 0.3). Same category without enough shared content = **suspected repeat**: `reasons` only, the flag stays false. Different players/authors/places are never
merged. The history is bounded (40 per author, 50k authors, 2 h of window) and stored in SQLite for the HTTP service, so a restart does not lose it; batch mode
sorts records by time first, so the result does not depend on the file order. Without an author id nothing is compared (and the UI no longer invents a shared author).

### Related complaints (not duplicates)
Complaints from OTHER authors about the same accused player within 24 h are listed in `reasons` («Связанные жалобы: …») and in the debug signals,
so an admin can review them together. They never set `is_spam_or_duplicate` and never change priority. Weak check on the data: no measurable
link to the admin verdict «already punished» (`docs/metrics/dedup.json`), partly because TRAIN contains only 20 % of reporters.

### Spam
Floods, a repeated word, a promotion sent by the player (links, giveaways, promo codes). A complaint about an ad, a quotation of an ad and a question about
advertising are not spam. A bare word such as «казино» is a topic, not a signal.

### LLM mode (optional) and its limits
* Sensitive tickets, spam and injection attempts never reach the model. The model never writes about rules: it places `[[Rn]]` markers and the code inserts the verified paraphrase.
* Three layers check what the model wrote itself: (0) a **whitelist of sentence types** (`guard.check_llm_sentences`): every sentence must be a greeting/thanks, a receipt of the message,
  a request for details or a question about details, «a person needs to review the ticket», an attribution of the player's own words («по вашим словам…») or a NEGATED promise; anything else
  («Рассмотрение займёт пару часов», «Игрок потеряет доступ навсегда», «Решение уже принято в вашу пользу») is rejected without needing to know the exact phrase;
  (1) the pattern guard (promises, completed actions, transfer claims, screenshot claims, URLs/commands/contacts, invented rule numbers);
  (2) a structural check: any sanction/refund/restoration/compensation topic must be negated in its clause (a question mark gives no exemption), no deadlines or time words (also «пару часов»), no numbers that are not in the ticket,
  no procedures («нажмите», «обратитесь в»). A rejection is recorded in `reasons`, `dbg.llm_error` and `guard_notes`, and the verified template is used.
* **This is not a proof of safety.** The whitelist is a finite lexicon in six languages: a sentence of a harmless type can still carry a wrong detail, and a harmless sentence in a phrasing the lexicon does not know is rejected
  (the price of the design: the model adds little beyond the templates). Since the fifth review the check is a CLOSED VOCABULARY per language (template words + neutral support words),
  no secrets at all, strict negated-promise shapes, and no model-written «по вашим словам» (only verbatim from the template, whose dynamic fields are validated). Tested with mocked answers only.
* Budget: one call per ticket; the whole ticket has `LLM_TOTAL_BUDGET_S` (default 4 s; the p95 target is 5 s for the whole ticket). The call runs in a worker thread under a **hard deadline**: the response body is read chunk by chunk and abandoned
  when the budget is gone, and the caller stops waiting at the deadline even if the server keeps dripping bytes (the HTTP timeout alone limits only one wait, not the whole call; tested with a local slow-drip server).
  One retry only for a 400 whose message names `thinking`/`output_config`/`effort`, only if ≥0.8 s remain, never with a higher `max_tokens`; 429/5xx are not retried.
  With prices configured the worst-case cost (estimated input + `max_tokens`) must fit `LLM_BUDGET_PER_TICKET_USD` or `max_tokens` is lowered / the call is skipped.
  Without prices the cost is reported as unknown and the ceiling cannot be enforced (a config warning says so). The MEASURED cost is 0 only when no request left the process; a sent request
  without a usage record (timeout after sending, a rejected first attempt, a malformed answer) makes the cost UNKNOWN; the worst-case estimate is kept in a separate field. At most 16 calls are
  queued or running; a task whose budget expired in the queue sends nothing; a sent request cannot be recalled and may still be billed. One real run exists (author's, Gemini free tier, artificial tickets, before this guard).

### Language
Names, nicknames, words after «игрок/ник/player» and quotations do not decide the language («Меня оскорбил игрок Қайрат» is Russian). Uzbek in Cyrillic (ҳ, ў, typical
words and endings) is recognised; `ў` alone is not turned into the Kazakh `ұ` when the text is Uzbek. The server language is only a tie-breaker for very short texts.
Game symbols: `ќ=қ ѕ=ң ў=ұ ї=ү` agree with the PDF; `ј=ә є=ғ ґ=ө` come from the REAL data (the PDF/README table is wrong for these three; `docs/LIMITATIONS.md` row 10).

## Output contract
Exactly the required fields. Additive, optional: `extracted.attachments`, `extracted.evidence_verified`, `extracted.evidence_status`, `extracted.rule_refs_unverified`, `draft_reply.terms`.
Diagnostics (signals, retrieved rules, latency, cost) are available at `POST /triage/debug` and in response headers `X-Triage-Mode`, `X-Latency-Ms`, `X-Cost-Usd`.
`lang` may be `ru kk uz en es pt lat und` (README superset of the PDF's list; `LANG_STRICT_CONTRACT=1` restricts it to the PDF's five); `lat` = Russian typed in Latin letters, answered in Russian.

## needs_human policy
Always: payments, account hack/access/lost property, appeals and complaints about admins (`taxonomy.needs_human_always`), every complaint about a player
(the punishment decision belongs to an admin), help requests, player reports about violations, unanswerable questions, injection attempts, threats outside the game,
possible mass bugs, low confidence (<0.4, or <0.5 for ambiguous), category inferred only from the player's rule number.
Not required: empty/off-topic reports, formal «please resubmit with evidence / ask in the right channel» replies to non-complaints, questions answered from a verified
rule point or a defined game term.

## Evaluation
Four kinds of evidence, never mixed: (1) synthetic regression checks (authors' own examples; guard against regressions, not accuracy); (2) **silver agreement** with what
administrators did (rule code in the reply, appeal verdict): a weak signal, not annotation; (3) AI-assisted blind review: a bug-finding aid, optimistic; (4) independent
**human** labels (`make review-set` → `make score-review`): the only source of Accuracy/macro-F1, **absent today**, so those numbers are reported as unknown.
**Macro-F1:** averaged over the classes that occur in the gold labels **or** in the predictions (scikit-learn convention, undefined ratios = 0): a class we predict but
annotators never used lowers the score instead of being ignored. The gold-only variant, the classes used by one side only and a per-class table are printed next to it.
The held-out sample for human labelling is drawn from VALID, excludes texts of the development sample and prefers reporters that do not occur in the development sample;
936 of the 1000 valid reporters also occur in train and valid texts were read while the lexicon was built, so it is **not** an untouched set (said in the report).
`scripts/coverage_report.py` adds behaviour counters and keeps three questions apart: (1) **policy compliance for signals the service recognised** (a recognised sensitive signal always reaches a person with the right
priority); (2) **detection completeness and false alarms**, measured on `tests/fixtures/sensitive_expectations.jsonl`, a list whose expectations were written in advance (authors' list, not human labels), because counter (1)
cannot see a topic the detector missed; (3) **whether a cited rule fits the event**, which only a human can judge and which is NOT measured: «technically valid citations» means that the point exists, passes its anchor words and is named in the text.
«Answers with a source attached» means a rule point or a defined term is attached to the draft; it says nothing about usefulness. The share of escalations is reported but is not a target.
