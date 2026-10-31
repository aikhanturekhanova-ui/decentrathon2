# Contradictions, assumptions, limitations, questions

## Contradictions found in the materials (and the behaviour we chose)
| # | Contradiction | Choice |
|---|---|---|
| 1 | The PDF's sample reply says «по скриншотам видно нарушение» and «Игрок … получит мут на 30 минут» for a ticket where only the **count** of screenshots exists, while the constraints forbid promising punishment. | Constraints win. We never claim that attachments were viewed and never state a punishment (guard + tests `test_contract_example_does_not_copy_the_dangerous_sample`). |
| 2 | PDF: validation/`category`,`priority` "exist only in the labelled validation". README + files: `tickets_valid.jsonl` has **no** category/priority. | We have no gold labels. Accuracy/F1 are reported only from human labels we create (none yet). |
| 3 | PDF `lang` values `ru kk uz en und`; README adds `es pt lat`; PDF mentions JSONL, CSV and **ARB** exports but only JSONL/JSON were delivered. | We emit the README superset. |
| 4 | Dataset `lang` labels are heuristic and wrong in places (Cyrillic texts labelled `lat`, Uzbek labelled `pt`, only 31 `kk` in valid). | Own detector; dataset labels used for nothing but a diagnostics table. |
| 5 | Draft taxonomy: `complaint.dm` lists rules `5.3, 5.4, GZ`, but 5.3 is "finish the RP situation" and 5.4 is "killing at a house entrance"; there is **no general DM rule** in the rulebook. | Cite 5.3/5.4/GZ only when the text points to them; a plain "убил без причины" gets no citation and a question about the place (instead of a made-up rule). |
| 6 | Taxonomy priority text: «жалобы с доказательствами» → p1, but the PDF sample (insult + 2 screenshots) is p2, and 99% of complaints carry screenshots. | Priority follows the category; evidence does not upgrade it. |
| 7 | `report.help` is described with rule `3.1`, which actually forbids asking for teleport/currency through reports. | 3.1 is cited only if the player asks for such a thing. |
| 8 | PDF FAQ: "external LLM APIs allowed, **except** case 1 (CPU-only)"; this is case 03. | Allowed; cost is reported. |
| 9 | Bare «помогите» is an example of `report.offtopic`, though it is a request for help. | Follow the taxonomy example. |
| 10 | **The cp1251 look-alike table in the PDF/README is wrong for three letters** (found by a viewer question and verified on all kz/kk texts of train+valid): the data has ј=ә, є=ғ, ґ=ө («Сјлеметсіз», «шаєым», «кґмек», «дјлелдеріѕіз», «сўраєыѕызды»), the table says є=ә, ґ=ғ, ј=ө. ќ=қ, ѕ=ң, ў=ұ, ї=ү agree. | We use the table derived from the data (`app/textutil.py`). Kazakh drafts are written in normal Kazakh by default; `KK_OUTPUT_GAME_SYMBOLS=1` emits the game's look-alikes (for text typed into the game). Please confirm with the organizers. |
| 11 | PDF contract: `lang` ∈ ru kk uz en und; README of the data: also es pt lat. | Default: the README superset. `LANG_STRICT_CONTRACT=1` restricts to the PDF's five (lat→ru, es/pt→und); the draft language stays real. One env variable to flip once the organizers answer. |
| 12 | Taxonomy priority text («complaints with proof → p1») vs the PDF example (insult + 2 screenshots → p2) vs the data (99 % of complaints have screenshots). | One consistent policy, stated as an interpretation: the CATEGORY decides the base priority, evidence never changes it, and mandatory p0 raises (payment, hack, lost access, threat outside the game, mass failure) are applied last and never lowered. |
| 13 | The PDF sample reply says the complaint «передана»/the administrator will act, while `/triage` stores nothing and assigns nobody (`PERSIST_TRIAGE=0`). | Drafts never claim a transfer (the guard rejects such sentences in all six languages): they say a person NEEDS to review the ticket. The local «Отправить» button in the admin panel says explicitly that nothing is delivered. |
| 14 | A text can say «доказательств нет» while attachments > 0, or «скрины есть» with 0 attachments. | Attachments are a count, the words are a claim; `extracted.evidence_status` keeps them apart, the conflict is written to `reasons`, `evidence_verified` stays false. |

## Assumptions (unconfirmed)
* Hidden test records look like the validation file (they may contain admin messages; we ignore them).
* Taxonomy and priority rules are a draft: `help→p2`, `scam→p1 only with a big amount/valuable property else p2`, `cheat/rmt→p1`, `account/payment/threat/mass bug→p0`.
* The confidence value is a hand-made heuristic, **not calibrated**.

## What is NOT verified
* Accuracy/macro-F1 on any human-labelled data (0 labelled rows).
* Kazakh and Uzbek quality: drafts are template-based, written by the authors, **not reviewed by native speakers**; validation has n=31 kk and 119 uz tickets and no human labels for them. Language support is therefore claimed only as "templates exist and pass safety checks".
* Spanish/Portuguese: templates exist, there are practically no real examples (the dataset `pt`/`es` labels are mostly mislabelled Russian/Uzbek).
* LLM mode: mock-tested; one real run on artificial tickets (author's Gemini free-tier run, see «LLM mode» below) predates the current guard; real latency/cost of the current code are not measured.
* Draft quality as seen by administrators (the 30% criterion): unknown. Template drafts are safe but formulaic.

## What the automated checks do NOT prove
* **No guarantee of safety against promises.** The guard is a finite list of patterns in 6 languages plus a structural rule for the model's own text (a sanction/refund topic must be negated, no deadlines, no foreign numbers, no procedures). Three external reviews each found forms the previous version missed; the next phrasing in kk/uz/es/pt may slip through in LLM mode. Mitigations, not guarantees: the model never writes rule statements (markers), sensitive tickets never reach the model, every draft goes to a person.
* "0 violations" in the audit means "none of the checks implemented in the audit fired". The audit now also checks meaning (sensitive topic → human and p0, no refusal for a sensitive topic, no claimed transfer, place-bound rule with a place), because the old shape-only audit missed exactly those defects. It still cannot know about defects nobody thought of.
* `confidence` is heuristic, not calibrated. `cost_usd_*` fields in the batch statistics are `null` unless every ticket's cost is known; the known part is stored separately as `cost_usd_known_sum`.
* Recurring game questions («как устроиться в полицию?») are answered only if the FAQ file (`data/kb/faq.json`) contains a sourced, **reviewed** answer; otherwise they are handed to a human. `make faq-candidates` builds *suggestions* from TRAIN administrator replies (with provenance, cleaned of promises/links/numbers/personal outcomes); nothing is copied automatically and an administrator's reply is not treated as a universal truth.
* The sensitive-topic detector is a lexicon: a refund request or a hacking report in an unseen slang, or in kk/uz/es/pt phrasing we did not think of, can be missed (it then falls to the ordinary path, which still sends complaints and unanswerable questions to a person, but a question-like phrasing could be answered automatically). The test set for it consists of our own phrasings (`tests/test_review3.py`), not of labelled real tickets, because the real tickets are almost free of such cases (valid: 3 payment, 1 access, 0 hack, 0 threat outside the game).
* The place context of a rule is a lexicon too: a place written in an unusual way is not recognised as the Green Zone, the rule is then not cited and the draft asks for the place (the safe direction).
* Duplicate detection compares tickets of ONE author about ONE accused player; event details (place, server, clock time) are a small list of words. Two real events at the same place within an hour look like a duplicate; a paraphrase with no shared word looks like a "suspected repeat" at most. No labelled duplicates exist, so its precision/recall are unknown.
* The priority rule for "complaints with proof" is an interpretation of contradictory materials (see the table above).

## The learned complaint model
* Trained on WEAK labels (rule code or appeal verdict in the admin reply, ~44 % of TRAIN complaints, mostly approved or appealed ones). The labelled subset is not
  representative of all complaints (e.g. insults are over-represented), and admin decisions can be wrong. Agreement with these labels is reported, never as accuracy.
* It has no `complaint.invalid` class and does not touch reports, safety flags, priority raises or the handoff policy (all rule-based).
* Kazakh: too few examples in TRAIN; the model's weight shrinks when most words are unseen, so the lexicon decides there. Uzbek improved on VALID (0.72 → 0.92 agreement, n=25 — small).
* The model file is built locally from confidential data and must be deleted with the data (it is under `data/`).
* `confidence` for complaints is now the blended probability (capped, lowered for near-ties): better grounded than before but still **not calibrated**.

## Moderation analysis (toxicity, language segments)
* A lexicon of stems plus normalization and context rules, written by the authors; no learned toxicity model (no labelled toxicity data exists).
  Euphemisms («слово на букву х») and one-word insults with innocent meanings (Kazakh «ит» = dog) are not recognised (recorded as known limitations).
* Direction (quoted vs own words) relies on quotes, reporting verbs (incl. the Kazakh «деп» and Uzbek «deb») and second-person words; unusual phrasing can
  be misattributed. The result is shown to the admin; it changes the category only by adding evidence for an insult complaint.
* Kazakh and Uzbek in Latin letters share many words; a short message with only shared words is labelled Uzbek.
* The adversarial suite is written by the same authors as the lexicon: it guards against regressions, it does not measure real-world recall.

## LLM mode (Anthropic or Gemini)
* Measured only with Gemini's free tier on 18 artificial tickets: 60.7 % of model drafts accepted, p95 1.4 s; under provider overload the calls
  hit 503 / the time budget and every ticket fell back to the template (p95 3.8 s, still within 5 s). The free tier also answers 429 above its
  per-minute quota, so a 500-ticket batch needs a paid tier or pacing. Anthropic was never called for real (no key).
* The free tier may use submitted text for training: real tickets are never sent to it (`LLM_ENABLED=0` forced in every evaluation script).
* Cost: tokens are measured (≈ 743 in / 102 out per call); the price is whatever the provider charges for the chosen model (`LLM_PRICE_*`).
* The sentence-type whitelist rejects the model's own paraphrases («Согласно вам…», a 9-word thank-you): safe by design, it lowers the acceptance rate.

## Other limits
* The lexicon is built on slang seen in the train/valid files; new slang will be missed (it routes to `other`/`question` with low confidence → human).
* Duplicate history lives in memory (bounded) and, for the HTTP service, in the SQLite file; batch mode does not persist it by design (a re-run must give the same answer).
* No clustering of reply templates beyond the basic lexical one (bonus only partly done).
* Related complaints (other authors, same accused, 24 h) are context for the admin; on the data they showed no link to the «already punished» verdict.
* Rule texts are only in Russian; kk/uz/en/es/pt paraphrases are ours and tied to anchors in the real text.
* **CI: the GitHub Actions runs of this repository fail to START, not to pass.** The run annotations say «The job was not started because recent account payments have failed or your spending limit needs to be increased»: none of the four jobs executed a single line of the project. This is an account/organisation billing limit, not a code or workflow fault, and only the account owner can change it (Settings → Billing and plans). Consequently the Linux/`make`/Python 3.10–3.13 run is **unconfirmed**; what was run locally is listed in `docs/REQUIREMENTS_TRACE.md` (tests, batch command, audit, a clean copy from `git archive`). The workflow itself was made easier to diagnose (fail-fast off, environment printed, health wait loop, server log on failure).
* The model's own sentences must use a CLOSED vocabulary (the words of the verified templates plus a short list of neutral support words, per language), may not mention secrets, and must be of a harmless type; a restatement of the player's claim («по вашим словам…») is accepted only verbatim from the verified template, whose dynamic fields (nick, server, counts) are validated first. This makes the model's contribution small (greeting, which details to ask) and still is a finite lexicon, not a proof. The current guard has only been run against mocked answers: its acceptance rate with a real model is unknown (the author's earlier 60.7 % was measured with the previous, looser guard).
* Accused player: with several nicknames and no accusation marker the field stays empty; a complaint written as «Dias_Volkov и Arman_Petrov» without roles is not resolved. For complaints the game's `reported_nick` field wins when it is present.
* Green Zone list: the objects are matched by word stems and a city; a player who names an object in an unusual way is not recognised (nothing is cited, the draft asks). Listed objects without a city count as zones as listed («Банки», «Автосалоны»): the borders given in the rules («территория включает сам объект и прилегающие участки») are not modelled beyond the words of the story. «Сначала были в ЗЗ, потом…» is treated as «place not established» even when the event really happened inside.
* Appeals sent as a `report` keep the category the lexicon found (the taxonomy has no report-side appeal category); only priority (≥ p2) and the draft are specific.
* The recurring game questions («Как купить машину?») stay unanswered: the rulebook contains conduct rules, not game instructions, and the administrators' replies in TRAIN are almost all acknowledgements (`make faq-candidates` finds 2 usable candidates in 118 337 tickets). Questions about the leader/organisation/family/OPG/war sections of `rules.json` are not answered from them (nothing is cited from sections that were not checked by hand).
* `lang`: the dataset schema lists `ru | kk | uz | en | und` for the player's text; our output also uses `es`, `pt` and `lat` (the client is localised in six languages and the data README names them). `LANG_STRICT_CONTRACT=1` restricts the output to the five; which one the organizers expect is an open question.
* Data contamination: ~100 validation tickets and the most frequent train texts were read while designing the lexicon; the human review sets (`make review-set`) exclude texts shared between dev and final, but valid is not a fully untouched set.
* Windows: `make` is usually absent; use the commands in the README.

## Questions for the organizers
1. May `rules.json`/`taxonomy.json` (or their paraphrases) be published in the repository? (We load them locally and do not commit them.)
2. Is the taxonomy final? Especially priorities of `report.help`, `complaint.scam`, `complaint.admin`, and the treatment of complaints without evidence.
3. Does the hidden test contain administrator messages / `verdict` fields (we ignore them), and `reporter_hash`/`created_at` (needed for dedup)?
4. Should an appeal of one's own punishment be `complaint.admin` (our choice) or something else?
5. Should `lang` follow the dataset heuristic (`lat`, `und`) or be a real language code only?
6. Is there a required repository/submission format beyond "GitHub repo with README, Makefile, report"?
7. Is there a mechanical-metric definition for "relevant citation" (section-aware `ref` or plain `ref`)?

## What needs people (cannot be done by the code or its author)
| Task | Why | Affects |
|---|---|---|
| Label ~300 tickets (`make review-set`, guide `docs/ANNOTATION_GUIDE.md`, then `make score-review`); ideally two annotators | Accuracy/macro-F1 need human gold labels (`docs/WHY_NO_ACCURACY.md`) | classification (25 %) |
| Native Kazakh and Uzbek speakers review the phrases in `app/phrases.py` (sections `kk`, `uz`) and the drafts in `docs/REPORT.md` | the templates were written without a native reviewer | kk/uz (15 %) and drafts (30 %) |
| An LLM key with configured prices (Anthropic or a paid Gemini tier), then `make llm-smoke` on artificial tickets | the current guard was never run against a real model; the only real run (Gemini free tier) predates it and has no cost | cost/latency (10 %), drafts |
| Run `make install test predict` on Linux (or let the CI workflow do it after the first push) | the `Makefile` and Docker image were not executed on the Windows development machine | reproducibility |
| Organizer answers to the questions above | taxonomy, priorities, `lang` values, test conditions | classification |
| Run `make train` before the judged run (or let `make predict` do it) | the model is not in the repository | classification |
| Fill `data/kb/faq.json` with sourced answers (format in `app/faq.py`) | no organizer material about game mechanics exists; recurring questions go to a human until then | question answering |