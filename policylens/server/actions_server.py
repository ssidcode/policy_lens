"""MCP server 3/3: actions with a human-approval gate. The model can only DRAFT (draft_request). Approving or
rejecting (decide_draft) and listing (list_drafts) are marked host-only in mcp_servers.json, so the model
can never call them; the user does, via buttons in the UI. The only writable database lives here."""
import sqlite3
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP

DB = Path(__file__).resolve().parent.parent.parent / "data" / "actions.db"
mcp = FastMCP("policylens-actions")
PENDING = "pending approval"


def _db():
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS drafts(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, summary TEXT,
                   status TEXT, request_id TEXT, created REAL)""")
    return con


@mcp.tool()
def draft_request(kind: str, summary: str) -> dict:
    """Create a DRAFT request (e.g. kind='re-evaluation') for the user. Nothing is submitted until the user approves."""
    kind, summary = kind.strip().lower()[:40], summary.strip()[:500]
    if not summary:
        return {"error": "summary is required"}
    with _db() as con:
        cur = con.execute("INSERT INTO drafts(kind, summary, status, created) VALUES (?,?,?,?)",
                          (kind or "other", summary, PENDING, time.time()))
        return {"draft_id": cur.lastrowid, "status": PENDING,
                "note": "Waiting for the user's approval in the app; it has NOT been submitted."}


@mcp.tool()
def list_drafts(status: str = PENDING) -> dict:
    """HOST-ONLY. List drafts with the given status."""
    with _db() as con:
        rows = con.execute("SELECT id, kind, summary, status, request_id FROM drafts WHERE status = ? ORDER BY id",
                           (status,)).fetchall()
    return {"drafts": [dict(r) for r in rows]}


@mcp.tool()
def decide_draft(draft_id: int, approve: bool) -> dict:
    """HOST-ONLY. Approve (submit) or reject a pending draft."""
    with _db() as con:
        row = con.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        if not row or row["status"] != PENDING:
            return {"ok": False, "error": "draft not found or not pending"}
        status, req = ("submitted", f"REQ-{2000 + draft_id}") if approve else ("rejected", None)
        con.execute("UPDATE drafts SET status = ?, request_id = ? WHERE id = ?", (status, req, draft_id))
    return {"ok": True, "status": status, "request_id": req}


if __name__ == "__main__":
    mcp.run()
