PY ?= python3
VENV ?= .venv
# use the virtualenv when it exists (after `make install`), otherwise the system interpreter
RUN := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(PY))
IN ?= tickets.jsonl
OUT ?= pred.jsonl
MODEL ?= data/models/complaint_clf.joblib
TRAIN_DATA ?= data/handout/tickets_train.jsonl.gz

.PHONY: install test run predict train model insights eval bench report audit coverage faq-candidates demo review-set score-review llm-smoke docker

install:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements-dev.txt

test:
	$(RUN) -m pytest

run: model
	$(RUN) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# learned complaint classifier on TRAIN (silver labels from admin replies) + evaluation -> docs/metrics/ml_classifier.json (~3 min)
train:
	PYTHONPATH=. $(RUN) scripts/train_classifier.py

# build the model once if it is missing and the training data is present (fit only, ~1 min); otherwise the lexicon is used alone
model:
	@if [ ! -f $(MODEL) ] && [ -f $(TRAIN_DATA) ]; then echo "no $(MODEL): training it once (make train for the full evaluation)"; \
	  PYTHONPATH=. $(RUN) scripts/train_classifier.py --fast; \
	elif [ ! -f $(MODEL) ]; then echo "WARNING: no $(MODEL) and no $(TRAIN_DATA): classifying with the lexicon only"; fi

# make predict IN=tickets.jsonl OUT=pred.jsonl   (compact bodies or full dataset records, .gz allowed)
predict: model
	$(RUN) -m app.cli predict --in $(IN) --out $(OUT)

eval:
	PYTHONPATH=. $(RUN) scripts/data_profile.py
	PYTHONPATH=. $(RUN) scripts/silver_eval.py

# invariants of the contract on every real ticket + fuzzing (needs data/handout; ~4 min)
audit:
	PYTHONPATH=. $(RUN) scripts/audit_core.py

bench:
	PYTHONPATH=. $(RUN) scripts/bench_latency.py

report:
	PYTHONPATH=. $(RUN) scripts/make_report.py

# behaviour counters on real tickets -> docs/metrics/coverage.json (needs data/handout)
coverage:
	PYTHONPATH=. $(RUN) scripts/coverage_report.py

# SUGGESTED answers for recurring questions from TRAIN admin replies -> data/local/faq_candidates.json (never used automatically)
faq-candidates:
	PYTHONPATH=. $(RUN) scripts/build_faq_candidates.py

# aggregated historical patterns from TRAIN for the dashboard -> docs/metrics/history.json (~2 min)
insights:
	PYTHONPATH=. $(RUN) scripts/history_insights.py

review-set:
	PYTHONPATH=. $(RUN) scripts/make_review_set.py

score-review:
	PYTHONPATH=. $(RUN) scripts/score_review.py --labels data/local/review_final_labeled.csv

llm-smoke:
	PYTHONPATH=. $(RUN) scripts/llm_smoke.py

demo:
	$(RUN) -m uvicorn app.main:app --port 8000 & sleep 2; curl -s -X POST localhost:8000/api/demo/seed; echo; echo "open http://localhost:8000/admin"

docker:
	docker build -t support-copilot . && docker run --rm -p 8000:8000 -v $$PWD/data:/app/data --env-file .env support-copilot
