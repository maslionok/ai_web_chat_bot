import os
import glob
import asyncio
import shelve
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader
import numpy as np
import faiss
import httpx
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
MAGENTO_BASE_URL     = os.getenv("MAGENTO_BASE_URL", "").rstrip('/')
MAGENTO_STORE_CODE   = os.getenv("MAGENTO_STORE_CODE", "eu1_EN")
MAGENTO_BEARER_TOKEN = os.getenv("MAGENTO_BEARER_TOKEN")
DEFAULT_MAX_PAGES    = int(os.getenv("DEFAULT_MAX_PAGES", "200"))
CHATWOOT_API_URL     = os.getenv("CHATWOOT_API_URL")
CHATWOOT_ACCOUNT_ID  = os.getenv("CHATWOOT_ACCOUNT_ID")
CHATWOOT_INBOX_ID    = os.getenv("CHATWOOT_INBOX_ID")
CHATWOOT_API_KEY     = os.getenv("CHATWOOT_API_KEY")

CACHE_DIR = ".cache"
CACHE_DB = "cache"
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_DB)
os.makedirs(CACHE_DIR, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__, template_folder='templates', static_folder='static')
faiss_index = None

# Omitted: utility functions for crawling, embedding, Magento fetch, etc.

# ─── Chatwoot ────────────────────────────────────────────────────────────────
def send_message_to_chatwoot_client(name, email, content):
    headers = {
        "Content-Type": "application/json",
        "api_access_token": CHATWOOT_API_KEY
    }
    contact_payload = {"name": name, "email": email, "inbox_id": CHATWOOT_INBOX_ID}
    contact_url = f"{CHATWOOT_API_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/contacts"
    r = requests.post(contact_url, headers=headers, json=contact_payload)
    if r.status_code == 422:
        search_url = f"{CHATWOOT_API_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/contacts/search"
        r = requests.get(search_url, headers=headers, params={"q": email})
        contact_id = r.json()["payload"][0]["id"]
    else:
        contact_id = r.json().get("payload", {}).get("contact", {}).get("id")
    conv_payload = {"inbox_id": CHATWOOT_INBOX_ID, "contact_id": contact_id}
    conv_url = f"{CHATWOOT_API_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations"
    r = requests.post(conv_url, headers=headers, json=conv_payload)
    conv_id = r.json().get("id")
    msg_payload = {"content": content, "message_type": "incoming"}
    msg_url = f"{CHATWOOT_API_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conv_id}/messages"
    requests.post(msg_url, headers=headers, json=msg_payload)

def escalate_to_human(question, customer_id=None):
    name = "Website User"
    email = f"siteuser_{customer_id or 'unknown'}@example.com"
    send_message_to_chatwoot_client(name, email, f"User requested human help: {question}")
    with shelve.open(CACHE_PATH) as db:
        db[f"chat_escalated||{email}"] = True

# ─── Flask Routes ────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    question = data.get('question', '')
    conversation_id = str(data.get("conversation_id", "default"))
    customer_id = data.get("customer_id")
    email = f"siteuser_{customer_id or 'unknown'}@example.com"

    with shelve.open(CACHE_PATH) as db:
        history_key = f"chat_history||{conversation_id}"
        messages = db.get(history_key, [])
        messages.append({"role": "user", "content": question})
        db[history_key] = messages

        if db.get(f"chat_escalated||{email}"):
            send_message_to_chatwoot_client("Website User", email, question)
            return jsonify({'answer': 'A human agent will respond soon.'})

        if any(phrase in question.lower() for phrase in ["switch to human", "talk to human", "real human"]):
            escalate_to_human(question, customer_id)
            return jsonify({'answer': 'A human agent will contact you shortly.'})

    # Omitted: GPT and FAISS logic
    answer = "Pretend this came from GPT"

    with shelve.open(CACHE_PATH) as db:
        messages = db.get(history_key, [])
        messages.append({"role": "bot", "content": answer})
        db[history_key] = messages

    return jsonify({'answer': answer})

@app.route('/api/chatwoot_webhook', methods=['POST'])
def receive_chatwoot_reply():
    data = request.json
    content = data.get("content")
    sender_type = data.get("sender", {}).get("type")
    raw_conversation_id = data.get("conversation", {}).get("id")

    if not content or not raw_conversation_id:
        return jsonify({"status": "ignored"}), 200

    conversation_id = str(raw_conversation_id)

    if sender_type in ["User", "Agent", "user"]:
        with shelve.open(CACHE_PATH) as db:
            history_key = f"chat_history||{conversation_id}"
            messages = db.get(history_key, [])
            messages.append({"role": "bot", "content": content})
            db[history_key] = messages

    return jsonify({"status": "received"})

@app.route('/api/messages/<conversation_id>', methods=['GET'])
def get_messages(conversation_id):
    with shelve.open(CACHE_PATH) as db:
        history_key = f"chat_history||{conversation_id}"
        return jsonify(db.get(history_key, []))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)