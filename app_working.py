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
import faiss
import httpx
from openai import OpenAI
from dotenv import load_dotenv

# ─── Load configuration ───────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
MAGENTO_BASE_URL     = os.getenv("MAGENTO_BASE_URL", "").rstrip('/')
MAGENTO_STORE_CODE   = os.getenv("MAGENTO_STORE_CODE", "eu1_EN")
MAGENTO_BEARER_TOKEN = os.getenv("MAGENTO_BEARER_TOKEN")
DEFAULT_MAX_PAGES    = int(os.getenv("DEFAULT_MAX_PAGES", "200"))

# ─── Setup a .cache directory ─────────────────────────────────────────────────
CACHE_DIR  = os.getenv("CACHE_DIR", ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
# we'll use a single shelve database in that folder
CACHE_DB       = os.getenv("CACHE_DB", "cache")  # no extension!
CACHE_PATH     = os.path.join(CACHE_DIR, CACHE_DB)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Flask app setup
tmpl_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app = Flask(__name__, template_folder=tmpl_dir, static_folder=static_dir)

# In-memory FAISS index (re-built if data size changes)
faiss_index = None

# Headers for Magento API calls
MAGENTO_HEADERS = {
    "Authorization": f"Bearer {MAGENTO_BEARER_TOKEN}",
    "Content-Type":  "application/json",
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

async def crawl_website_async(start_url, max_pages=200):
    visited = set()
    to_visit = [start_url]
    texts   = []
    base    = "{0.scheme}://{0.netloc}".format(urlparse(start_url))

    async with httpx.AsyncClient(follow_redirects=True) as session:
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            fetched, text, soup = await fetch_url(session, url)
            if fetched and text:
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
            params = {
                "searchCriteria[currentPage]": page,
                "searchCriteria[pageSize]":    page_size,
            }
            resp = await session.get(url, headers=MAGENTO_HEADERS, params=params, timeout=10)
            resp.raise_for_status()
            data  = resp.json()
            batch = data.get("items", [])
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
        f"Status: {'Enabled' if product.get('status') == 1 else 'Disabled'}",
    ]
    for attr in product.get('custom_attributes', []):
        out.append(f"{attr.get('attribute_code')}: {attr.get('value')}")
    return "\n".join(out)

# ─── PDF loader ──────────────────────────────────────────────────────────────
def load_pdfs(folder="extra_info"):
    docs = []
    if not os.path.isdir(folder):
        return docs
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        try:
            reader = PdfReader(path)
            pages  = [p.extract_text() for p in reader.pages if p.extract_text()]
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
        batch = chunks[i:i+batch_size]
        resp  = client.embeddings.create(model="text-embedding-3-small", input=batch)
        embeddings.extend([d.embedding for d in resp.data])
    return np.array(embeddings).astype('float32')

# ─── Language detection & ChatGPT call ────────────────────────────────────────
def detect_language(question):
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",  "content": "Detect language and reply with only the language name."},
            {"role": "user",    "content": question}
        ]
    )
    return resp.choices[0].message.content.strip()

def ask_chatgpt(question, chunks, model="gpt-4.1-mini"):
    context = "\n\n".join(chunks)
    lang    = detect_language(question)
    sys_msg = (
        f"You are a precise assistant. Answer ONLY based on provided context. Respond in {lang}. "
        "Short, step-by-step if needed, only rely on context."
    )
    prompt  = f"Context:\n{context}\n\nQuestion: {question}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# ─── Flask routes ─────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data      = request.json or {}
    question  = data.get('question', '')
    model     = data.get('model', 'gpt-4.1-mini')
    crawl_url = data.get('url', None)
    max_pages = int(data.get('max_pages', DEFAULT_MAX_PAGES))

    # open our single on-disk cache
    with shelve.open(CACHE_PATH) as db:

        # 1) Magento cache
        mag_key = f"magento||{MAGENTO_STORE_CODE}"
        if mag_key in db:
            mag_chunks, mag_embs = db[mag_key]
        else:
            prods      = asyncio.run(fetch_magento_products(page_size=200))
            texts      = [product_to_text(p) for p in prods] + load_pdfs()
            mag_chunks = []
            for t in texts:
                mag_chunks.extend(split_text(t))
            mag_embs   = embed_chunks(mag_chunks)
            db[mag_key] = (mag_chunks, mag_embs)

        combined_chunks = list(mag_chunks)
        combined_embs   = mag_embs

        # 2) Crawl cache (optional)
        if crawl_url:
            crawl_key = f"crawl||{crawl_url}||{max_pages}"
            if crawl_key in db:
                crawl_chunks, crawl_embs = db[crawl_key]
            else:
                crawled = asyncio.run(crawl_website_async(crawl_url, max_pages))
                crawl_chunks = split_text(crawled)
                crawl_embs   = embed_chunks(crawl_chunks)
                db[crawl_key] = (crawl_chunks, crawl_embs)

            combined_chunks.extend(crawl_chunks)
            combined_embs = np.vstack([combined_embs, crawl_embs])

    # 3) (Re)build FAISS index if needed
    global faiss_index
    if faiss_index is None or faiss_index.ntotal != combined_embs.shape[0]:
        idx = faiss.IndexFlatL2(combined_embs.shape[1])
        idx.add(combined_embs)
        faiss_index = idx

    # 4) Query embedding + search
    q   = client.embeddings.create(model="text-embedding-3-small", input=[question])
    q_emb = np.array([q.data[0].embedding]).astype('float32')
    dist, inds = faiss_index.search(q_emb, 10)
    top = [combined_chunks[i] for i in inds[0]]

    # 5) Ask GPT and return
    answer = ask_chatgpt(question, top, model)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
