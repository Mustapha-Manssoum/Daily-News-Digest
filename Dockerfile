FROM python:3.11-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

# If using HF API local models, add transformers, torch
# RUN pip install transformers torch sentencepiece

WORKDIR /app

COPY daily_news_digest.py .

ENTRYPOINT ["python", "daily_news_digest.py"]