import os
import glob
import asyncio
import shelve
import sys
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader
import numpy as np
import requests
import faiss
import httpx
from openai import OpenAI
from dotenv import load_dotenv

import os
import glob
import asyncio
import shelve
import sys
import time
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader
import numpy as np
import faiss
import httpx
from openai import OpenAI
from dotenv import load_dotenv


# ─── Load configuration ───────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
MAGENTO_BASE_URL     = os.getenv("MAGENTO_BASE_URL", "").rstrip('/')
MAGENTO_STORE_CODE   = os.getenv("MAGENTO_STORE_CODE", "eu1_EN")
MAGENTO_BEARER_TOKEN = os.getenv("MAGENTO_BEARER_TOKEN", "")
DEFAULT_MAX_PAGES    = int(os.getenv("DEFAULT_MAX_PAGES", "200"))

# ─── Cache setup ──────────────────────────────────────────────────────────────
CACHE_DIR  = os.getenv("CACHE_DIR", ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DB   = os.getenv("CACHE_DB", "cache")
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_DB)

# ─── Initialize clients ───────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)
faiss_index = None

# ─── Chatwoot configuration ───────────────────────────────────────────────────
CHATWOOT_BASE_URL   = os.getenv("CHATWOOT_BASE_URL", "https://app.chatwoot.com")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")
CHATWOOT_INBOX_ID   = os.getenv("CHATWOOT_INBOX_ID")
CHATWOOT_API_TOKEN  = os.getenv("CHATWOOT_API_TOKEN")
# Validate Chatwoot credentials
if not all([CHATWOOT_ACCOUNT_ID, CHATWOOT_INBOX_ID, CHATWOOT_API_TOKEN]):
    raise RuntimeError("Missing Chatwoot configuration: ensure ACCOUNT_ID, INBOX_ID, and API_TOKEN are set")

CHATWOOT_HEADERS = {
    "Content-Type": "application/json",
    "api_access_token": CHATWOOT_API_TOKEN
}

# ─── Track escalated conversations ───────────────────────────────────────────
escalated_conversations = {}

# ─── Chatwoot helper functions ────────────────────────────────────────────────
def create_chatwoot_conversation(user_name: str = "Chatbot User"):
    """
    Create a new Chatwoot conversation and return its ID.
    """
    external_id = str(int(time.time() * 1000))
    random_number = np.random.randint(100000, 999999)
    payload = {
        "name": user_name,
        "email": f"testclient{random_number}@example.com",
        "inbox_id": CHATWOOT_INBOX_ID
    }
    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/contacts"
    resp = requests.post(url, headers=CHATWOOT_HEADERS, json=payload)
    contact_data = resp.json()

    contact_id = contact_data["payload"]["contact"]["id"]
    source_id = contact_data["payload"]["contact"]["contact_inboxes"][0]["source_id"]

    print("✅ Contact created")

    # === 2. Create a conversation ===
    conversation_payload = {
        "source_id": source_id,
        "inbox_id": CHATWOOT_INBOX_ID,
        "contact_id": contact_id
    }
    r = requests.post(
        f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations",
        headers=CHATWOOT_HEADERS,
        json=conversation_payload
    )
    conversation_id = r.json()["id"]
    

    
    return conversation_id


def send_to_human_agent(message: str, user_name: str = "Chatbot User"):
    """
    1) Create or reuse a Chatwoot conversation
    2) Send the user's message into it
    """
    if user_name in escalated_conversations:
        conv_id = escalated_conversations[user_name]
    else:
        conv_id = create_chatwoot_conversation(user_name)
        if not conv_id:
            print("❌ Skipping message send due to missing conversation ID.")
            return False
        escalated_conversations[user_name] = conv_id

    # Filter out bot messages containing only "..."
    filtered_message = "\n".join(
        line for line in message.split("\n") if not (line.startswith("bot:") and line.strip() == "bot: ...")
    )

    msg_payload = {"content": filtered_message, "message_type": "incoming"}
    msg_url = (
        f"{CHATWOOT_BASE_URL}/api/v1/accounts/"
        f"{CHATWOOT_ACCOUNT_ID}/conversations/{conv_id}/messages"
    )
    resp = httpx.post(msg_url, headers=CHATWOOT_HEADERS, json=msg_payload)
    if resp.status_code not in (200, 201):
        print(f"❌ Chatwoot message send failed: {resp.status_code} {resp.text}")
        return False
    return True

# ─── Magento headers ───────────────────────────────────────────────────────────
MAGENTO_HEADERS = {
    "Authorization": f"Bearer {MAGENTO_BEARER_TOKEN}",
    "Content-Type": "application/json"
}

# ─── Crawling utilities ──────────────────────────────────────────────────────
async def fetch_url(session, url):
    try:
        r = await session.get(url, timeout=10)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            soup = BeautifulSoup(r.text, 'html.parser')
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            return url, soup.get_text(separator=' ', strip=True), soup
    except Exception:
        pass
    return url, None, None

async def crawl_website_async(start_url, max_pages=DEFAULT_MAX_PAGES):
    visited, to_visit, texts = set(), [start_url], []
    base = "{0.scheme}://{0.netloc}".format(urlparse(start_url))
    async with httpx.AsyncClient(follow_redirects=True) as session:
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            fetched, text, soup = await fetch_url(session, url)
            if text:
                texts.append(text)
                visited.add(fetched)
                for link in soup.find_all("a", href=True):
                    href = urljoin(base, link['href']).split('#')[0]
                    if base in href and href not in visited and href not in to_visit:
                        to_visit.append(href)
    return "\n\n".join(texts)

# ─── Magento API fetcher ─────────────────────────────────────────────────────
async def fetch_magento_products(page_size=100):
    items = []
    async with httpx.AsyncClient() as session:
        page = 1
        while True:
            url = f"{MAGENTO_BASE_URL}/rest/{MAGENTO_STORE_CODE}/V1/products"
            params = {"searchCriteria[currentPage]": page, "searchCriteria[pageSize]": page_size}
            resp = await session.get(url, headers=MAGENTO_HEADERS, params=params, timeout=10)
            resp.raise_for_status()
            batch = resp.json().get("items", [])
            if not batch:
                break
            items.extend(batch)
            page += 1
    return items

# ─── Data flattening ─────────────────────────────────────────────────────────
def product_to_text(product: dict) -> str:
    out = [
        f"SKU: {product.get('sku')}",
        f"Name: {product.get('name')}",
        f"Price: {product.get('price')}",
        f"Status: {'Enabled' if product.get('status') == 1 else 'Disabled'}"
    ]
    for attr in product.get("custom_attributes", []):
        out.append(f"{attr.get('attribute_code')}: {attr.get('value')}")
    return "\n".join(out)

# ─── PDF loader ──────────────────────────────────────────────────────────────
def load_pdfs(folder="extra_info"):
    docs = []
    if not os.path.isdir(folder): return docs
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        try:
            reader = PdfReader(path)
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            if pages:
                docs.append("\n".join(pages))
        except Exception as e:
            print(f"PDF load error {path}: {e}")
    return docs

# ─── Text splitting & embedding ───────────────────────────────────────────────
def split_text(text, max_tokens=300):
    words = text.split()
    return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def embed_chunks(chunks, batch_size=100):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        resp = client.embeddings.create(model="text-embedding-3-small", input=chunks[i:i+batch_size])
        embeddings.extend([d.embedding for d in resp.data])
    return np.array(embeddings).astype("float32")

# ─── Language detection & ChatGPT call ────────────────────────────────────────
def detect_language(question):
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"Detect language and reply with only the language name."},
                  {"role":"user","content":question}]
    )
    return resp.choices[0].message.content.strip()

def ask_chatgpt(question, chunks, model="gpt-4.1-mini"):
    context = "\n\n".join(chunks)
    lang = detect_language(question)
    sys_msg = (f"You are a precise assistant. Answer ONLY based on provided context. Respond in {lang}. "
               "Short, step-by-step if needed, only rely on context.")
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys_msg},
                  {"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip()

# ─── Flask app setup ─────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data      = request.json or {}
    question  = data.get('question', '').strip()
    model     = data.get('model', 'gpt-4.1-mini')
    crawl_url = data.get('url')
    max_pages = int(data.get('max_pages', DEFAULT_MAX_PAGES))
    user_name = data.get('user_name', 'Chatbot User')
    history   = data.get('history', [])  # Retrieve conversation history

    # Check if conversation is already escalated
    if user_name in escalated_conversations:
        full_conversation = "\n".join(
            [f"{msg['who']}: {msg['text']}" for msg in history if not (msg['who'] == 'bot' and msg['text'] == "...")]
        )
        send_to_human_agent(full_conversation, user_name)
        return jsonify({
            "answer": "Your message has been forwarded to a human agent. They will assist you shortly."
        }), 200

    # Human-Agent handoff detection
    triggers = ["human", "agent", "support", "operator", "representative"]
    if any(t in question.lower() for t in triggers):
        full_conversation = "\n".join(
            [f"{msg['who']}: {msg['text']}" for msg in history if not (msg['who'] == 'bot' and msg['text'] == "...")]
        )
        send_to_human_agent(full_conversation, user_name)
        return jsonify({
            "answer": "Your message has been forwarded to a human agent. They will assist you shortly."
        }), 200

    with shelve.open(CACHE_PATH) as db:
        # Magento
        mag_key = f"magento||{MAGENTO_STORE_CODE}"
        if mag_key in db:
            mag_chunks, mag_embs = db[mag_key]
        else:
            prods = asyncio.run(fetch_magento_products(page_size=200))
            texts = [product_to_text(p) for p in prods] + load_pdfs()
            mag_chunks = []
            for t in texts:
                mag_chunks.extend(split_text(t))
            mag_embs = embed_chunks(mag_chunks)
            db[mag_key] = (mag_chunks, mag_embs)

        combined_chunks = list(mag_chunks)
        combined_embs = mag_embs

        # Crawl
        if crawl_url:
            crawl_key = f"crawl||{crawl_url}||{max_pages}"
            if crawl_key in db:
                crawl_chunks, crawl_embs = db[crawl_key]
            else:
                crawled = asyncio.run(crawl_website_async(crawl_url, max_pages))
                crawl_chunks = split_text(crawled)
                crawl_embs = embed_chunks(crawl_chunks)
                db[crawl_key] = (crawl_chunks, crawl_embs)
            combined_chunks.extend(crawl_chunks)
            combined_embs = np.vstack([combined_embs, crawl_embs])

    # Build FAISS index
    global faiss_index
    if faiss_index is None or faiss_index.ntotal != combined_embs.shape[0]:
        idx = faiss.IndexFlatL2(combined_embs.shape[1])
        idx.add(combined_embs)
        faiss_index = idx

    # Query
    resp_emb = client.embeddings.create(model="text-embedding-3-small", input=[question])
    q_emb   = np.array([resp_emb.data[0].embedding]).astype("float32")
    dist, inds = faiss_index.search(q_emb, 10)
    top = [combined_chunks[i] for i in inds[0]]

    answer = ask_chatgpt(question, top, model)
    return jsonify({"answer": answer})


@app.route('/chatwoot/webhook', methods=['POST'])
def chatwoot_webhook():
    payload = request.get_json(force=True)
    print("[DEBUG] Webhook payload received:", payload)  # Debugging print

    event = payload.get("event") or payload.get("webhook")
   

    if event == "message.created":
        data = payload["data"]
       

        conv_id = data["conversation"]["id"]
        message = data["message"]["content"]
        sender = data["sender"].get("identifier")
       

        # if user said “talk to a human”
        if "talk to human" in message.lower():
           

            endpoint = f"{CHATWOOT_BASE_URL.rstrip}/api/v1/conversations/{conv_id}/messages"
            body = {
              "content": f"User requested human intervention: {message}",
              "message_type": 1,
              "private": False
            }
            headers = {
              "api_access_token": CHATWOOT_API_TOKEN,
              "Content-Type": "application/json"
            }

            try:
                r = httpx.post(endpoint, json=body, headers=headers, timeout=10)
                r.raise_for_status()
                
            except Exception as e:
                print("[ERROR] Failed to escalate:", e)

    return "", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
