from __future__ import annotations

import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =========================
# CONFIG
# =========================

URLS = [
    "https://www.geeksforgeeks.org/machine-learning/getting-started-with-transformers/",
    "https://www.datacamp.com/tutorial/how-transformers-work",
    "https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/",
    "https://www.ibm.com/think/topics/transformer-model",
    "https://medium.com/data-science/transformers-89034557de14",
    "https://serokell.io/blog/transformers-in-ml"
]

EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "distilgpt2"

CHUNK_SIZE = 300
OVERLAP = 50


# =========================
# FETCH
# =========================

def fetch_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(" ")
        text = " ".join(text.split())
        return text
    except:
        return ""


def load_data():
    chunks, sources = [], []

    for url in URLS:
        text = fetch_text(url)
        words = text.split()

        for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
            chunk = " ".join(words[i:i+CHUNK_SIZE])
            if len(chunk) > 50:
                chunks.append(chunk)
                sources.append(url)

    return chunks, sources


# =========================
# EMBEDDING + INDEX
# =========================

def build_index(chunks):
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return embedder, index


# =========================
# RETRIEVE
# =========================

def retrieve(query, embedder, index, chunks, sources, k=3):
    q = embedder.encode([query])
    dist, idx = index.search(np.array(q), k)

    results = []
    for i in idx[0]:
        results.append((chunks[i], sources[i]))

    return results


# =========================
# GENERATE
# =========================

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)

    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def answer(query, hits, generator):
    context = "\n\n".join([h[0] for h in hits])

    prompt = f"""
Answer based on context:

{context}

Question: {query}
Answer:
"""

    out = generator(prompt, max_length=300)[0]["generated_text"]
    return out