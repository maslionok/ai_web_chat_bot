import os
import glob
import asyncio
import sys
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
import numpy as np
import faiss
import httpx
from openai import OpenAI
from dotenv import load_dotenv

# ─── Load config ──────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
MAGENTO_BASE_URL    = os.getenv("MAGENTO_BASE_URL", "").rstrip('/')
MAGENTO_STORE_CODE  = os.getenv("MAGENTO_STORE_CODE", "eu1_EN")
MAGENTO_BEARER_TOKEN= os.getenv("MAGENTO_BEARER_TOKEN")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Flask setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
)

# in-memory cache for embeddings & data
cache = {}

# common headers for Magento calls
MAGENTO_HEADERS = {
    "Authorization": f"Bearer {MAGENTO_BEARER_TOKEN}",
    "Content-Type": "application/json",
}

# ─── Magento product fetcher ─────────────────────────────────────────────────

async def fetch_magento_products(page_size: int = 100):
    """Pull every product via Magento REST, paginating until empty."""
    items = []
    async with httpx.AsyncClient() as session:
        page = 1
        while True:
            url = f"{MAGENTO_BASE_URL}/rest/{MAGENTO_STORE_CODE}/V1/products"
            params = {
                "searchCriteria[currentPage]": page,
                "searchCriteria[pageSize]":  page_size,
            }
            resp = await session.get(url, headers=MAGENTO_HEADERS, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("items", [])
            if not batch:
                break
            items.extend(batch)
            page += 1
    return items

def product_to_text(p: dict) -> str:
    """
    Flatten key fields of one product into a tiny human-readable snippet.
    """
    parts = [
        f"SKU: {p.get('sku')}",
        f"Name: {p.get('name')}",
        f"Price: {p.get('price')}",
        f"Status: {'Enabled' if p.get('status') == 1 else 'Disabled'}",
    ]
    for attr in p.get("custom_attributes", []):
        code = attr.get("attribute_code")
        val  = attr.get("value")
        parts.append(f"{code}: {val}")
    return "\n".join(parts)

# ─── PDF loader (unchanged) ───────────────────────────────────────────────────

def load_pdfs(folder="extra_info"):
    texts = []
    if not os.path.isdir(folder):
        return texts
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        try:
            reader = PdfReader(path)
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            if pages:
                texts.append("\n".join(pages))
        except Exception as e:
            print(f"PDF load error {path}: {e}")
    return texts

# ─── Text splitting & embedding (unchanged) ──────────────────────────────────

def split_text(text, max_tokens=300):
    words = text.split()
    return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def embed_chunks(chunks, batch_size=100):
    embs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        embs.extend([d.embedding for d in resp.data])
    return np.array(embs).astype('float32')

def get_top_chunks(query, chunks, embeddings, k=10, candidate_mul=3):
    q = client.embeddings.create(model="text-embedding-3-small", input=query)
    q_emb = np.array([q.data[0].embedding]).astype('float32')
    # build FAISS index once
    if 'faiss' not in cache:
        idx = faiss.IndexFlatL2(q_emb.shape[1])
        idx.add(embeddings)
        cache['faiss'] = idx
    else:
        idx = cache['faiss']
    dist, inds = idx.search(q_emb, k * candidate_mul)
    cands = [{'dist':d, 'txt': chunks[i]} for d,i in zip(dist[0], inds[0])]
    cands.sort(key=lambda x: x['dist'])
    return [c['txt'] for c in cands[:k]]

# ─── Language detection & Chat completion (unchanged) ─────────────────────────

def detect_language(question):
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"Detect language and reply with only the language name."},
            {"role":"user","content":question}
        ]
    )
    return resp.choices[0].message.content.strip()

def ask_chatgpt(question, chunks, model="gpt-4.1-mini"):
    context = "\n\n".join(chunks)
    lang = detect_language(question)
    sys_msg = (
        f"You are a precise assistant. Answer ONLY based on provided context. Respond in {lang}. "
        "Short, step-by-step if needed, only rely on context."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":sys_msg},
            {"role":"user","content":prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# ─── Flask routes ────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data     = request.json
    question = data.get('question', '')
    model    = data.get('model', 'gpt-4.1-mini')

    # caching key
    key = f"magento||{MAGENTO_STORE_CODE}"

    if key in cache:
        chunks, embs = cache[key]
    else:
        # 1) fetch all products, turn to text
        products = asyncio.run(fetch_magento_products(page_size=200))
        texts    = [product_to_text(p) for p in products]

        # 2) optionally load local PDFs
        texts += load_pdfs()

        # 3) split & embed
        chunks = []
        for t in texts:
            chunks.extend(split_text(t))
        embs = embed_chunks(chunks)

        cache[key] = (chunks, embs)

    # 4) retrieve top chunks + ask GPT
    top   = get_top_chunks(question, cache[key][0], cache[key][1])
    ans   = ask_chatgpt(question, top, model)

    return jsonify({'answer': ans})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
