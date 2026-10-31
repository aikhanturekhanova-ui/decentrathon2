# Models, libraries, sources, licenses

| Item | Use | License / terms |
|---|---|---|
| Python 3.12+ | runtime | PSF |
| FastAPI 0.162.0, Starlette | HTTP API | MIT / BSD-3 |
| Uvicorn 0.54.0 | server | BSD-3 |
| Pydantic 2.14.2 | validation | MIT |
| httpx 0.28.1 | HTTP client (LLM, Telegram), test transport | BSD-3 |
| scikit-learn 1.9.1 (+ numpy, scipy, joblib, threadpoolctl) | learned complaint classifier (`make train`); optional on Python 3.10 | BSD-3 |
| pytest 9.2.0 (dev) | tests | MIT |
| SQLite (stdlib) | queue and edits | public domain |
| Google Gemini API (`generateContent`), default model `gemini-3.5-flash-lite` (**optional**, only with `GEMINI_API_KEY` and `LLM_ENABLED=1`) | draft rewriting; measured only on artificial demo tickets | Google APIs / Gemini API terms; the free tier may use submitted content to improve Google products, so no real player text is sent to it |
| Anthropic Messages API, default model `claude-haiku-5-5` (**optional**, only when `ANTHROPIC_API_KEY` is set) | draft rewriting | Anthropic commercial terms; prices from <https://platform.claude.com/docs/en/about-claude/pricing> (Haiku 5.5: $0.10 / $0.50 per MTok, checked 2026-10-30). Never used in the default template mode. |
| Telegram Bot API (**optional**) | demo bot | Telegram Bot terms |
| Organizers' `rules.json`, `taxonomy.json`, tickets | RAG corpus / checks / evaluation, loaded locally | provided for Decentrathon 6.0 only; not redistributed; delete within 14 days after winners are announced |

No pretrained model, no external dataset and no embedding model is used. The only learned model is trained locally from the organizers' TRAIN file by `make train` and is not distributed. Lexicons, templates and Kazakh/Uzbek/Spanish/Portuguese phrases were written by the repository authors.
Versions are pinned in `requirements*.txt` (checked to run on Python 3.14; Docker image uses 3.12).
