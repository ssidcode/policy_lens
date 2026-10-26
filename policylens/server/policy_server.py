"""MCP server 1/3: policy knowledge. Tool: search_policy. Resources: policy://index, policy://doc/{doc_id}.
Prompt: policy_assistant (the agent's system prompt is served over MCP)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid macOS duplicate-OpenMP abort (faiss + torch)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
from mcp.server.fastmcp import FastMCP
from ..prompts import SYSTEM
from ..retriever import DOCS_DIR, get_retriever, load_chunks

mcp = FastMCP("policylens-policy")
_retriever = get_retriever()


@mcp.tool()
def search_policy(query: str, k: int = 3) -> dict:
    """Semantic search over policy documents. Returns top-k sections with ids to cite."""
    return {"results": _retriever.search(query, min(max(k, 1), 5))}


@mcp.resource("policy://index")
def index() -> str:
    """List of all documents and their section ids."""
    docs = {}
    for c in load_chunks():
        docs.setdefault(c["doc"], []).append(c["id"])
    return json.dumps(docs, indent=1)


@mcp.resource("policy://doc/{doc_id}")
def doc(doc_id: str) -> str:
    """Full text of one policy document."""
    p = (DOCS_DIR / f"{doc_id}.md").resolve()
    if p.parent != DOCS_DIR.resolve() or not p.exists():
        raise ValueError("unknown document")
    return p.read_text(encoding="utf-8")


@mcp.prompt()
def policy_assistant() -> str:
    """System prompt for the policy Q&A agent."""
    return SYSTEM


if __name__ == "__main__":
    mcp.run()
