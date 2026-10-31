FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY scripts ./scripts
# aggregated metrics only (no tickets): the dashboard's history panel and /api/insights read docs/metrics/history.json
COPY docs/metrics ./docs/metrics
# artificial demo tickets used by "Демо-данные" in the admin panel (POST /api/demo/seed)
COPY tests/fixtures/demo_tickets.jsonl ./tests/fixtures/demo_tickets.jsonl
# The organizers' files are NOT in the image: mount them at /app/data/handout (see README "Data").
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
