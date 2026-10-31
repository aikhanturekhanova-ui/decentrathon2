# Why Accuracy and macro-F1 are not in the report (and what exists instead)

**Short answer: they cannot be computed honestly from what we have, and I will not invent them.**

1. **The files contain no category or priority labels.** I checked both files: `tickets_valid.jsonl` (1,000 tickets) and
   `tickets_train.jsonl.gz` (118,337) have no `category`/`priority` field. The PDF says labels exist only "in the labelled
   validation" and the hidden test (`tickets_hidden.jsonl`, 500 tickets) is not given to teams until grading. Without gold labels,
   Accuracy (category) and macro-F1 (priority) are undefined.
2. **The administrators' answers are not labels of the category.** `verdict`, `punishment` and `violation` describe what an
   admin did (e.g. «2.1 ОП» → a mute), not what category the player's text belongs to. They exist for only ~15 % of complaints, they
   are missing for all reports, and an appeal that quotes «2.1 ОП» would be mislabelled as an insult. Using them as ground truth would
   be label noise presented as accuracy. They are used only as a *weak agreement check* (≈84–86 %, `docs/metrics/silver.json`), explicitly not accuracy.
3. **Human labels can only come from a human.** The task rules say not to present automatically generated labels as human annotation.
   An AI-assisted reviewer cannot be the annotator for a number that will be read as human-verified, and the lexicon was written by the same author, so those labels
   would be biased towards the system's own mistakes.
4. **The taxonomy is a draft** («Черновик… подтверждает руководитель поддержки»). Even a perfect annotator needs the final priority rules
   (e.g. help requests, scams, complaints with evidence are ambiguous in the materials).

## What exists instead (all labelled for what it is)
| Evidence | What it is | Result | Value |
|---|---|---|---|
| `docs/metrics/ai_review.json` | 110 unambiguous tickets from the **train-derived dev sample**, labelled by an AI assistant *without seeing the service output*; ambiguous rows excluded | category accuracy ≈ 0.85, priority macro-F1 ≈ 0.91 | **Not human, small, biased** (the same author wrote the lexicon, and the classifier was tuned on this sample after the first pass: 0.50 → 0.85). Useful to find bugs, not to claim quality |
| `docs/metrics/silver.json` | agreement with the rule code / appeal verdict in admins' replies | ≈ 0.84 (valid), ≈ 0.86 (20k train sample) | weak, noisy |
| `tests/fixtures/gold_synthetic.jsonl` | 65 artificial examples written by the authors | 1.0 | regression guard only |
| `make audit` | 119,337 real tickets + fuzz | 0 contract violations | correctness of constraints, not classification quality |

## How to get the real number (needs people, ~1–2 hours)
1. `make review-set` creates `data/local/review_final.csv` (300 tickets from the validation file, texts of the dev set removed, one ticket per text,
   max 3 per reporter, rare categories and kk/uz over-represented, **model output hidden**). The AI-labelled sample above was drawn from the *dev* file, so the
   final file is untouched by tuning.
2. Two people fill `human_category`, `human_priority`, `annotator` using `docs/ANNOTATION_GUIDE.md` (the second one on at least 100 rows to measure kappa).
3. `make score-review` prints Accuracy, macro-F1, per-category recall, per-language accuracy and the top confusions into `docs/metrics/human_eval.json`
   (it refuses to run on fewer than 50 valid rows) and `make report` puts them in `docs/REPORT.md`.
The real hidden-test score will still differ: the hidden labels come from the organizers' own conventions (the draft taxonomy), which is why questions 2 and 4 in `docs/LIMITATIONS.md` matter.
