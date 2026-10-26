import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_chunks(docs_dir=DOCS_DIR):
    """One chunk per '## Section'. Chunk id = <doc>#<section-slug> (used as the citation)."""
    chunks = []
    for p in sorted(Path(docs_dir).glob("*.md")):
        for block in re.split(r"^## ", p.read_text(encoding="utf-8"), flags=re.M)[1:]:
            head, _, body = block.partition("\n")
            chunks.append({"id": f"{p.stem}#{slug(head)}", "doc": p.stem,
                           "section": head.strip(), "text": body.strip()})
    return chunks


def _chunk_text(c):
    return f"{c['section']}. {c['text']}"


# ---------- Embedders (anything with encode(list[str]) -> L2-normalised float32 matrix) ----------
class SentenceTransformerEmbedder:
    """Real semantic embeddings. Downloads the model from Hugging Face on first use."""

    def __init__(self, name=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name or os.getenv("POLICYLENS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    def encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype="float32")


class HashingEmbedder:
    """Deterministic, download-free lexical stand-in. Used in tests; NOT semantic."""

    def __init__(self, dim=1024):
        self.v = HashingVectorizer(n_features=dim, stop_words="english", ngram_range=(1, 2),
                                   norm="l2", alternate_sign=False)

    def encode(self, texts):
        return self.v.transform(texts).toarray().astype("float32")


# ---------- Retrievers (shared contract: search(query, k) -> [{id, doc, section, text, score}]) ----------
class VectorRetriever:
    """Dense retrieval: embeddings + FAISS inner-product index (= cosine on normalised vectors)."""

    def __init__(self, chunks=None, embedder=None, min_score=0.35):
        import faiss
        self.chunks = chunks or load_chunks()
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.min_score = min_score
        vecs = self.embedder.encode([_chunk_text(c) for c in self.chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, k: int = 3):
        scores, ids = self.index.search(self.embedder.encode([query]), min(k, len(self.chunks)))
        return [{**self.chunks[i], "score": round(float(s), 3)}
                for s, i in zip(scores[0], ids[0]) if i >= 0 and s >= self.min_score]


class TfidfRetriever:
    """Lexical fallback; needs no model download."""

    def __init__(self, chunks=None, min_score=0.05):
        self.chunks = chunks or load_chunks()
        self.min_score = min_score
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([_chunk_text(c) for c in self.chunks])

    def search(self, query: str, k: int = 3):
        sims = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 3)} for i in order if sims[i] >= self.min_score]


Retriever = TfidfRetriever  # backwards-compatible alias


def get_retriever():
    """POLICYLENS_RETRIEVER = auto (default: vector, fall back to TF-IDF) | vector | tfidf"""
    mode = os.getenv("POLICYLENS_RETRIEVER", "auto")
    if mode in ("auto", "vector"):
        try:
            return VectorRetriever()
        except Exception as e:
            if mode == "vector":
                raise
            print(f"[policylens] vector retriever unavailable ({e}); falling back to TF-IDF", file=sys.stderr)
    return TfidfRetriever()
