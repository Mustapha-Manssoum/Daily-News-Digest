# 📰 Daily AI News Digest

Automatically collects daily news (IT, finance, geopolitics, politics, etc.), summarizes it using a Hugging Face model, and emails you a compact digest — all running automatically through **Docker + GitHub Actions**.

---

## 🚀 Overview

This project:
1. Fetches news from selected RSS feeds  
2. Summarizes each article using a Hugging Face summarization model (you can use a local hosted llm)
3. Sends a daily email digest  
4. Runs automatically once per day via GitHub Actions (no server needed)

Nice for getting an AI-curated summary of the world’s most relevant news every morning.

---

## 🛠️ Setup

### 1️⃣ Fork or clone this repository

```bash
git clone <repo-link>
cd news-digest
```

### 2️⃣ Add your credentials to GitHub Secrets

Go to
Settings → Secrets → Actions → New repository secret and add:
Secret name	    Description
HF_API_TOKEN	Your Hugging Face Inference API token
SMTP_USER	    Your SMTP username or sender email
SMTP_PASS	    Your SMTP app password or token
EMAIL_TO	    Recipient email address

💡 If using Gmail: create an App Password under Google Account → [Security → App passwords.](https://support.google.com/mail/answer/185833?sjid=14668307731930478409-EU)

### 3️⃣ Verify the GitHub Actions workflow

The workflow file is in .github/workflows/daily_digest.yml.

It builds and runs your Docker container once per day:
```bash
on:
  schedule:
    - cron: '0 7 * * *'  # 07:00 UTC daily
  workflow_dispatch:      # allows manual run
```
You can run it manually from the Actions tab → Daily News Digest → Run workflow.

## 🧠 Hugging Face Setup

Go to https://huggingface.co/settings/tokens

Create a new Access Token (read access)

Store it as HF_API_TOKEN in GitHub Secrets

Example summarization API call in your script:
```bash
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
response = requests.post(
    "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn",
    headers=headers,
    json={"inputs": text[:3000]}
)
```
