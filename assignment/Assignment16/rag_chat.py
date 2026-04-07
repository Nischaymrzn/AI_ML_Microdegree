import re
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configuration

# Knowledge base: at least 2 distinct, publicly accessible URLs
URLS = [
    "https://www.geeksforgeeks.org/machine-learning/getting-started-with-transformers/",
    "https://www.datacamp.com/tutorial/how-transformers-work",
    "https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/",
    "https://www.ibm.com/think/topics/transformer-model",
    "https://medium.com/data-science/transformers-89034557de14",
    "https://serokell.io/blog/transformers-in-ml",
]

# Embedding model — converts text → 384-dim numerical vectors
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 200     # words per chunk
CHUNK_OVERLAP = 40   # overlap between consecutive chunks

# STEP 1: Fetch & clean web pages

def fetch_page_text(url):
    """
    Download a web page and extract only the article text.

    Web pages contain HTML, scripts, menus, ads, etc.
    We strip all of that and keep just the readable content.

    Uses a full browser-like session with realistic headers
    so sites like DataCamp and Medium don't block us.
    """
    # Create a session with realistic browser headers
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;"
                  "q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    })

    response = session.get(url, timeout=20, allow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content tags
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                              "aside", "noscript", "iframe", "svg", "form",
                              "button", "input", "select", "textarea"]):
        tag.decompose()

    # Try to find the main article content first (more precise)
    article = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", {"class": re.compile(r"article|content|post|entry", re.I)})
        or soup.find("div", {"id": re.compile(r"article|content|post|entry", re.I)})
        or soup.body
        or soup
    )

    text = article.get_text(separator=" ", strip=True)

    # Clean up whitespace and web artifacts
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\s*edit\s*\]", "", text)
    text = re.sub(r"\[\d+\]", "", text)

    return text.strip()

# STEP 2: Chunk text (sliding window with overlap)

def chunk_text(text, source_url):
    """
    Split text into overlapping chunks that end at sentence boundaries.

    WHY CHUNK?
    - Full articles are too long to search efficiently
    - Smaller pieces = more precise retrieval
    - Overlap ensures info at boundaries isn't lost

    WHY SENTENCE BOUNDARIES?
    - Cutting at exactly 200 words often splits mid-sentence
    - "...introduced in 2017 in the Attention" ← bad!
    - We extend or trim each chunk to the nearest period
    """
    # First split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    chunks = []
    current_words = []
    current_sentences = []

    for sent in sentences:
        words = sent.split()
        current_words.extend(words)
        current_sentences.append(sent)

        # When we reach chunk size, save this chunk
        if len(current_words) >= CHUNK_SIZE:
            chunk_text_str = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text_str,
                "source": source_url,
            })

            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                w = len(s.split())
                if overlap_count + w > CHUNK_OVERLAP:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += w

            current_sentences = overlap_sentences
            current_words = []
            for s in current_sentences:
                current_words.extend(s.split())

    # Don't forget the last chunk
    if current_sentences and len(current_words) >= 20:
        chunks.append({
            "text": " ".join(current_sentences),
            "source": source_url,
        })

    return chunks

# STEP 3: Embed chunks & build FAISS vector index

def build_index(chunks, embedder):
    """
    Convert text chunks → vectors, then build a search index.

    HOW EMBEDDINGS WORK:
      - Each text chunk becomes a vector (list of 384 numbers)
      - Similar meaning = similar vectors (close together)
      - "self-attention" and "attention mechanism" → nearby vectors

    HOW FAISS WORKS:
      - FAISS = Facebook AI Similarity Search
      - Quickly finds which stored vectors are closest to a query
      - We use cosine similarity (via normalized inner product)
    """
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize so inner product = cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]  # 384
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


# STEP 4: Retrieve top-k relevant chunks

def retrieve(query, embedder, index, chunks, k=3):
    """
    Find the k most relevant chunks for a user question.

    Process:
      1. Embed the question with the same model
      2. FAISS finds the k nearest chunk vectors
      3. Return those chunks as evidence
    """
    q_vec = embedder.encode([query])
    q_vec = np.array(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "score": float(score),
        })
    return results

# STEP 5: Generate a natural answer from retrieved evidence

# Words that indicate a sentence is navigation/junk, not content
JUNK_PATTERNS = [
    "twitter", "facebook", "linkedin", "whatsapp", "telegram",
    "copy link", "share this", "read more", "sign up", "log in",
    "subscribe", "newsletter", "cookie", "privacy policy",
    "min read", "last updated", "table of contents",
    "click here", "advertisement", "sponsored",
    "courses", "tutorials", "interview prep",
    "in this article", "in this tutorial", "in this post",
    "article by", "written by", "published by",
    "get it in your inbox", "featured in",
]


def _is_junk(sentence):
    """Check if a sentence is navigation/boilerplate junk."""
    s = sentence.lower()
    # Too short to be useful
    if len(sentence) < 50:
        return True
    # Contains junk patterns
    for pattern in JUNK_PATTERNS:
        if pattern in s:
            return True
    # Looks like a page title or heading dump (no period at end, lots of caps)
    if not sentence.rstrip().endswith(('.', '!', '?', ')')):
        words = sentence.split()
        if len(words) > 5:
            caps = sum(1 for w in words if w[0:1].isupper())
            if caps / len(words) > 0.6:
                return True
    return False


def _split_sentences(text):
    """Split text into clean content sentences, filtering out junk."""
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    clean = []
    for s in raw:
        s = s.strip()
        if len(s) > 50 and not _is_junk(s):
            clean.append(s)
    return clean


def _score_sentence(sentence, question):
    """Score relevance of a sentence to the question using keyword overlap."""
    stop = {"the","a","an","is","are","was","were","in","on","at","to","for",
            "of","and","or","it","this","that","with","by","from","as","be",
            "has","have","had","do","does","did","what","how","why","when",
            "which","who","where","can","will","would","could","should","i",
            "you","they","its","their","these","those","such","been","not",
            "also","more","very","just","about","like","over","into","than"}
    q_words = set(question.lower().split()) - stop
    s_words = set(sentence.lower().split()) - stop
    if not q_words:
        return 0
    return len(q_words & s_words) / len(q_words)


def generate_answer(question, evidence):
    """
    Build a natural, readable answer from ALL retrieved evidence.

    HOW IT WORKS (Extractive Summarization):
      1. Collect clean sentences from ALL evidence chunks
      2. Filter out junk (nav text, social links, menus)
      3. Score each sentence by relevance to the question
      4. Pick the best unique sentences
      5. Format as natural prose with source citations

    This is the "G" (Generation) in RAG — extractive style.
    In production, you'd use an LLM (GPT/LLaMA) for fluent answers.
    """
    if not evidence:
        return "Sorry, I couldn't find relevant information in the knowledge base."

    from urllib.parse import urlparse

    # --- Collect and score clean sentences from all evidence ---
    scored = []
    for e in evidence:
        sentences = _split_sentences(e["text"])
        domain = urlparse(e["source"]).netloc.replace("www.", "")
        for s in sentences:
            score = _score_sentence(s, question)
            scored.append({
                "text": s, "score": score,
                "domain": domain, "url": e["source"],
            })

    # Sort by relevance, take best unique ones
    scored.sort(key=lambda x: x["score"], reverse=True)

    seen = set()
    top = []
    for item in scored:
        key = item["text"][:60].lower()
        if key in seen:
            continue
        seen.add(key)
        top.append(item)
        if len(top) >= 8:
            break

    if not top:
        return "I found some related content but couldn't extract a clear answer."

    # --- Build the answer ---
    # Smart opening based on question type
    q_lower = question.lower().strip().rstrip("?").strip()

    if q_lower.startswith("what is") or q_lower.startswith("what are"):
        topic = q_lower.replace("what is", "").replace("what are", "").strip()
        opening = f"### {topic.title()}\n\n"
    elif q_lower.startswith("how do") or q_lower.startswith("how does") or q_lower.startswith("how is"):
        opening = f"### {question.strip().rstrip('?')}?\n\n"
    elif q_lower.startswith("why"):
        opening = f"### {question.strip().rstrip('?')}?\n\n"
    else:
        opening = f"### {question.strip()}\n\n"

    # First sentence as the intro paragraph
    intro = top[0]["text"]

    # Remaining sentences as supporting details
    details = []
    for item in top[1:]:
        details.append(item["text"])

    # Combine: intro paragraph + detail paragraph(s)
    body = intro + "\n\n"
    if details:
        body += " ".join(details)

    # Source list
    seen_urls = set()
    sources = []
    for e in evidence:
        url = e["source"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        domain = urlparse(url).netloc.replace("www.", "")
        sources.append(f"- {domain}")

    return f"{opening}{body}\n\n---\n*Sources: {', '.join(sources)}*"

# Pipeline loader (used by both CLI and Streamlit)

def load_pipeline(urls=None):
    """
    Full setup: fetch pages → chunk → embed → index.
    Returns dict with chunks, embedder, index.
    """
    if urls is None:
        urls = URLS

    all_chunks = []
    source_stats = {}  # track how many chunks per source
    for url in urls:
        try:
            print(f"Fetching: {url}")
            text = fetch_page_text(url)
            c = chunk_text(text, source_url=url)
            word_count = len(text.split())
            print(f"  → {word_count} words → {len(c)} chunks")
            source_stats[url] = len(c)
            all_chunks.extend(c)
        except Exception as e:
            print(f"  ⚠ Skipping {url} — {e}")
            source_stats[url] = 0
            continue

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Chunks per source:")
    for url, count in source_stats.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {count:>3} chunks — {url}")

    if not all_chunks:
        raise RuntimeError("No text could be fetched from any URL. Check your internet connection.")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Building FAISS index...")
    index = build_index(all_chunks, embedder)

    print("Pipeline ready!")
    return {"chunks": all_chunks, "embedder": embedder, "index": index}


# CLI entry point (test without Streamlit)

if __name__ == "__main__":
    pipe = load_pipeline()

    while True:
        q = input("\nAsk (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit"):
            break

        hits = retrieve(q, pipe["embedder"], pipe["index"], pipe["chunks"])
        print("\n--- Evidence ---")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] score={h['score']:.3f} | {h['source']}")
            print(f"    {h['text'][:120]}...\n")

        print("--- Answer ---")
        print(generate_answer(q, hits))