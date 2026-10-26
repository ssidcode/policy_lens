"""Add/remove documents in the knowledge base (data/docs). The agent starts a fresh MCP server for every
question, so new documents are searchable immediately."""
import io
import re
from pathlib import Path

from .retriever import DOCS_DIR

MAX_BYTES = 5_000_000
TARGET_CHARS = 900  # approx. size of an auto-generated section
SUPPORTED = (".md", ".txt", ".pdf")


def _extract_text(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext in (".md", ".txt"):
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader
        text = "\n\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        if not text.strip():
            raise ValueError("No extractable text (scanned PDF?). OCR it first.")
        return text
    raise ValueError(f"Unsupported file type '{ext}'. Use: {', '.join(SUPPORTED)}")


def _auto_chunk(text: str):
    """Group paragraphs (or sentences, for very long paragraphs) into ~TARGET_CHARS sections."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) > TARGET_CHARS:
            units += [s for s in re.split(r"(?<=[.!?])\s+", para) if s]
        elif para:
            units.append(para)
    groups, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) > TARGET_CHARS:
            groups.append(cur)
            cur = u
        else:
            cur = f"{cur} {u}".strip()
    if cur:
        groups.append(cur)
    return groups


def save_document(name: str, data: bytes, docs_dir=DOCS_DIR):
    """Returns (doc_id, n_sections). Markdown with '## ' headings is kept as-is; anything else is auto-sectioned."""
    if len(data) > MAX_BYTES:
        raise ValueError("File too large (max 5 MB).")
    text = _extract_text(name, data)
    doc_id = re.sub(r"[^a-z0-9_]+", "_", Path(name).stem.lower()).strip("_") or "document"
    if re.search(r"^## ", text, flags=re.M):
        md = text
    else:
        groups = _auto_chunk(text)
        if not groups:
            raise ValueError("The file contains no text.")
        parts = []
        for i, g in enumerate(groups, 1):
            title = " ".join(re.sub(r"[^\w\s]", "", g).split()[:6])
            parts.append(f"## Part {i} {title}\n{g}")
        md = f"# {doc_id}\n\n" + "\n\n".join(parts)
    Path(docs_dir).mkdir(parents=True, exist_ok=True)
    (Path(docs_dir) / f"{doc_id}.md").write_text(md, encoding="utf-8")
    return doc_id, len(re.findall(r"^## ", md, flags=re.M))


def list_documents(docs_dir=DOCS_DIR):
    return [(p.stem, len(re.findall(r"^## ", p.read_text(encoding="utf-8"), flags=re.M)))
            for p in sorted(Path(docs_dir).glob("*.md"))]


def delete_document(doc_id: str, docs_dir=DOCS_DIR) -> bool:
    p = (Path(docs_dir) / f"{doc_id}.md").resolve()
    if p.parent != Path(docs_dir).resolve() or not p.exists():
        return False
    p.unlink()
    return True
