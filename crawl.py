#!/usr/bin/env python3
"""
crawl.py

Crawl a site breadth-first (up to max_pages),
extract all visible text, and write to a Unicode PDF.
Automatically downloads & extracts DejaVuSans.ttf if missing.
"""

import argparse
import os
import io
import zipfile
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from collections import deque

# SourceForge zip of DejaVu TTFs
FONT_ZIP_URL = (
    "https://downloads.sourceforge.net/project/dejavu/dejavu/2.37/"
    "dejavu-fonts-ttf-2.37.zip"
)

def ensure_font(font_path: str):
    if os.path.isfile(font_path):
        return
    print(f"[ ] '{font_path}' not found; downloading & extracting TTF…")
    resp = requests.get(FONT_ZIP_URL, timeout=30)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    # look for the DejaVuSans.ttf entry in any subfolder
    target = next((n for n in z.namelist() if n.endswith("/ttf/DejaVuSans.ttf")), None)
    if target is None:
        raise RuntimeError("DejaVuSans.ttf not found inside the ZIP")
    with open(font_path, "wb") as f:
        f.write(z.read(target))
    print(f"[+] Extracted '{font_path}'")

def crawl_site(start_url: str, max_pages: int) -> str:
    visited = set()
    queue   = deque([start_url])
    base    = "{0.scheme}://{0.netloc}".format(urlparse(start_url))
    texts   = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue

        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"[!] Failed to fetch {url}: {e}")
            continue

        if "text/html" not in r.headers.get("Content-Type", ""):
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript"]):
            tag.decompose()

        txt = soup.get_text(separator="\n", strip=True)
        if txt:
            print(f"[+] Crawled ({len(visited)+1}): {url} ({len(txt)} chars)")
            texts.append(f"--- {url} ---\n{txt}\n")
        visited.add(url)

        for a in soup.find_all("a", href=True):
            href = urljoin(base, a["href"]).split("#")[0]
            if href.startswith(base) and href not in visited and href not in queue:
                queue.append(href)

    return "\n\n".join(texts)

def text_to_pdf(text: str, output_path: str, font_path: str):
    # register the Unicode TrueType font
    pdfmetrics.registerFont(TTFont("DejaVu", font_path))

    c = canvas.Canvas(output_path, pagesize=A4)
    w, h = A4
    margin = 40
    text_obj = c.beginText(margin, h - margin)
    text_obj.setFont("DejaVu", 12)
    max_w = w - 2*margin

    for line in text.splitlines():
        if c.stringWidth(line, "DejaVu", 12) <= max_w:
            text_obj.textLine(line)
        else:
            # simple word-wrap
            buf = ""
            for word in line.split():
                test = f"{buf} {word}" if buf else word
                if c.stringWidth(test, "DejaVu", 12) <= max_w:
                    buf = test
                else:
                    text_obj.textLine(buf)
                    buf = word
            if buf:
                text_obj.textLine(buf)

        if text_obj.getY() < margin:
            c.drawText(text_obj)
            c.showPage()
            text_obj = c.beginText(margin, h - margin)
            text_obj.setFont("DejaVu", 12)

    c.drawText(text_obj)
    c.save()
    print(f"[+] PDF saved to {output_path}")

def main():
    p = argparse.ArgumentParser(
        description="Crawl a site and save all text to Unicode PDF"
    )
    p.add_argument("--url",       "-u", required=True, help="Start URL")
    p.add_argument("--max-pages", "-m", type=int, default=200, help="Max pages to crawl")
    p.add_argument("--output",    "-o", default="site_text.pdf", help="Output PDF file")
    p.add_argument("--font",      "-f", default="DejaVuSans.ttf", help="Path to TTF font")
    args = p.parse_args()

    ensure_font(args.font)
    print(f"[*] Crawling {args.url} (up to {args.max_pages} pages)…")
    full_text = crawl_site(args.url, args.max_pages)
    if not full_text.strip():
        print("[!] No text extracted; PDF not created.")
    else:
        text_to_pdf(full_text, args.output, args.font)

if __name__ == "__main__":
    main()
