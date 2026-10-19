FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY kb_bot ./kb_bot
COPY main.py .
COPY data ./data
# Mount your own documents over /app/data; pass OPENAI_API_KEY with -e (omit for offline mode).
ENTRYPOINT ["python", "-m", "kb_bot"]
