#!/usr/bin/env python3
"""
daily_news_digest.py
Simple daily news collector -> summarizer -> email sender.
"""

from dotenv import load_dotenv
import os
import time
import sqlite3
import smtplib
import textwrap
from email.mime.text import MIMEText
from datetime import datetime
from typing import List, Tuple
import feedparser
from newspaper import Article
import requests, json


# Optional: local transformers
USE_LOCAL_MODEL = False  # set True to use local summarizer (transformers)
# If USE_LOCAL_MODEL True, install: pip install transformers torch sentencepiece

# If using Hugging Face Inference API, set HF_API_TOKEN
USE_HF_API = not USE_LOCAL_MODEL

load_dotenv()  # load .env if present

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"  # or any other HF summarization model

# SMTP / delivery config
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "you@example.com")
SMTP_PASS = os.environ.get("SMTP_PASS", "")  # use app password or env var
EMAIL_FROM = SMTP_USER
EMAIL_TO = os.environ.get("EMAIL_TO", SMTP_USER)

DB_PATH = "news_digest.db"

# Feeds to collect — tweak/add as needed
FEEDS = {
    "IT": [
        "https://techcrunch.com/feed/",
        "https://news.ycombinator.com/rss",
        "https://www.wired.com/feed/rss.xml",
        "https://www.computerweekly.com/rss",
        "https://www.techrepublic.com/rssfeeds/",
        "https://www.gadgets360.com/rss"
    ],
    "Finance": [
        "https://www.nasdaq.com/feed/rssoutbound?category=Top-News",
        "https://www.nasdaq.com/feed/rssoutbound?category=Market-Headlines",
        "https://www.investing.com/rss/stock_stock_picks.rss",
        "https://www.nasdaq.com/feed/rssoutbound?category=Market-News"
    ],
    "Cryptocurrency": [
        "https://www.nasdaq.com/feed/rssoutbound?category=Cryptocurrencies"
    ],
    "Global-Politics": [
        "https://www.crisisgroup.org/rss"
    ],
    "Moroccan-Politics": [
        "https://www.crisisgroup.org/rss/133"
    ],
    "French-Politics": [
        "https://www.crisisgroup.org/rss/169"
    ]
}

# Summarization constraints
MAX_ARTICLES_PER_CATEGORY = 5
SUMMARY_MAX_TOKENS = 120


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sent_articles (
            id INTEGER PRIMARY KEY,
            url TEXT UNIQUE,
            title TEXT,
            date_sent TEXT
        )
    """)
    conn.commit()
    return conn

def mark_sent(conn, url, title):
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO sent_articles (url, title, date_sent) VALUES (?, ?, ?)",
              (url, title, datetime.now().isoformat()))
    conn.commit()

def was_sent(conn, url) -> bool:
    c = conn.cursor()
    c.execute("SELECT 1 FROM sent_articles WHERE url = ? LIMIT 1", (url,))
    return c.fetchone() is not None

# ---- Fetch & extract ----
def fetch_feed_entries(feed_url: str, timeout: int = 15) -> List[dict]:
    try:
        resp = requests.get(feed_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if "xml" not in resp.headers.get("Content-Type",""):
            print(f"Skipping non-RSS feed: {feed_url}")
            return []
        # Feedparser can parse from string
        d = feedparser.parse(resp.content)
        entries = d.entries if hasattr(d, "entries") else []
        return entries
    except requests.exceptions.RequestException as e:
        print(f"[fetch_feed_entries] Request failed: {e}")
        return []

def extract_article_text(url: str) -> Tuple[str,str]:
    # Returns (title, text)
    try:
        art = Article(url)
        art.download()
        art.parse()
        text = art.text
        title = art.title or url
        return title, text
    except Exception as e:
        print(f"[extract] failed for {url}: {e}")
        return "", ""
    
def safe_summarize_hf(text: str) -> str:
    MAX_CHARS = 3000
    if len(text) > MAX_CHARS:
        parts = [text[i:i+MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]
        summaries = []
        for p in parts:
            try:
                summaries.append(summarize_hf_api(p))
            except Exception as e:
                print(f"⚠️  Partial summarization failed: {e}")
                summaries.append("")
        merged = "\n".join([s for s in summaries if s.strip()])
        return summarize_hf_api(merged) if merged else "[Summary unavailable]"
    else:
        try:
            return summarize_hf_api(text)
        except Exception as e:
            print(f"⚠️  Summarization failed: {e}")
            return "[Summary unavailable]"


def summarize_local(text: str) -> str:
    # lazy-load transformers
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    # Use a lightweight summarization model
    model_name = "sshleifer/distilbart-cnn-12-6"
    summarizer = pipeline("summarization", model=model_name, truncation=True)
    # chunk if long
    max_chunk = 1000
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    parts = []
    for ch in chunks:
        out = summarizer(ch, max_length=SUMMARY_MAX_TOKENS, min_length=30, do_sample=False)
        parts.append(out[0]['summary_text'].strip())
    return " ".join(parts)

def summarize_hf_api(text: str, max_retries: int = 3) -> str:
    API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_SUMMARIZER_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
    payload = {"inputs": text[:3000], "parameters": {"max_new_tokens": 120, "min_length": 30}}

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=(5, 30))
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and isinstance(data[0], dict):
                return data[0].get("summary_text") or data[0].get("generated_text", "")
            elif isinstance(data, dict):
                if data.get("error"):
                    print(f"HF error: {data['error']}")
                    return textwrap.shorten(text, width=400)
                return data.get("summary_text", str(data))
            return str(data)
        except requests.exceptions.ReadTimeout:
            print(f"⚠️  HF API timeout (attempt {attempt+1}/{max_retries}), retrying...")
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"⚠️  HF API request failed: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"⚠️  Unexpected summarization error: {e}")
            break

    print("🚫 Summarization failed after retries, skipping.")
    return "[Summary unavailable due to API timeout]"

def summarize(text: str) -> str:
    if not text.strip():
        return ""
    if USE_LOCAL_MODEL:
        return summarize_local(text)
    else:
        return safe_summarize_hf(text)

# ---- Formatter & Delivery ----
def build_mail_body(digest: dict) -> str:
    lines = []
    lines.append(f"Daily News Digest — {datetime.now().date().isoformat()}\n")
    for cat, items in digest.items():
        lines.append(f"--- {cat} ({len(items)} items) ---\n")
        for it in items:
            lines.append(f"{it['title']}\n{it['summary']}\nLink: {it['url']}\n")
        lines.append("\n")
    lines.append("\nEnd of digest.")
    return "\n".join(lines)

def send_email(subject: str, body: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    s = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
    s.ehlo()
    if SMTP_PORT == 587:
        s.starttls()
    if SMTP_USER and SMTP_PASS:
        s.login(SMTP_USER, SMTP_PASS)
    s.send_message(msg)
    s.quit()

# ---- Main flow ----
def collect_and_send():
    conn = init_db()
    digest = {cat: [] for cat in FEEDS.keys()}

    for cat, feed_list in FEEDS.items():
        seen = 0
        for feed in feed_list:
            print(f"Fetching feed {cat} from: {feed}") 
            try:
                entries = fetch_feed_entries(feed)
                print(f" -------------> Found {len(entries)} entries")
            except Exception as e:
                print("feed fetch failed", feed, e)
                continue
            for e in entries:
                if seen >= MAX_ARTICLES_PER_CATEGORY:
                    break
                url = e.get("link") or e.get("id") or e.get("href") or None
                if not url:
                    continue
                if was_sent(conn, url):
                    continue
                title, text = extract_article_text(url)
                if not text:
                    # fallback to summary of description if available
                    text = e.get("summary") or e.get("description") or ""
                summary = summarize(text)
                if not summary:
                    summary = textwrap.shorten(text, width=300)
                digest[cat].append({"url": url, "title": title or e.get("title","(no title)"), "summary": summary})
                mark_sent(conn, url, title or e.get("title",""))
                seen += 1
                time.sleep(1)  # polite
            if seen >= MAX_ARTICLES_PER_CATEGORY:
                break

    body = build_mail_body(digest)
    subject = f"Daily Digest — {datetime.now().date().isoformat()}"
    send_email(subject, body)
    print("Digest sent.")

if __name__ == "__main__":
    collect_and_send()
