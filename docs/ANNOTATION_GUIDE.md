# Annotation guide (human labels for the review set)

Run `make review-set`, open `data/local/review_final.csv` (or `review_dev.csv`) in a spreadsheet, fill the columns, save as `review_final_labeled.csv`.
You do not see the model's answer. Fill: `human_category`, `human_priority`, `human_lang` (optional), `annotator` (your name), `notes`.

**Category** (one of the 14 codes of `taxonomy.json`): label what the player's *text* is about, not what the admin did.
* Several topics → the main complaint. Appeals of your own punishment («не согласен», «опру», «дали мут не за что») → `complaint.admin`.
* Complaint with no usable content (only «жб», a link, a question about the game) → `complaint.invalid`. A complaint whose text is clear but which has 0 attachments is also `invalid` (taxonomy: "без доказательств"); in the exports every complaint has ≥1 image.
* Reports: bare «помогите», «+», «ау», «привет» → `report.offtopic` (taxonomy example). Stuck/bug/teleport → `report.help`. Hack/lost access/missing property → `report.account`. Payment/donate → `report.payment`.
**Priority**: p0 hack/lost access, payment, mass bug, threats outside the game · p1 cheats, big scam, RMT/ads · p2 insults, DM, RP violations, help requests · p3 questions, off-topic, invalid.
Two annotators on the same 100 rows allow Cohen's kappa (`--labels2`). Label disagreements should be discussed and written to the draft taxonomy questions in `docs/LIMITATIONS.md`.
Native speakers of Kazakh/Uzbek should additionally rate drafts 1–5 (blind, next to a real answer) in `notes`; those scores are not computed automatically.
