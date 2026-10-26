"""MCP server 2/3: live records (READ-ONLY). Tools: check_deadline, get_request_status."""
import re
import sqlite3
from pathlib import Path
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent.parent
DB = ROOT / "data" / "policy.db"
ACTIONS_DB = ROOT / "data" / "actions.db"
mcp = FastMCP("policylens-records")


def _db():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)  # read-only: writes raise an error
    con.row_factory = sqlite3.Row
    return con


def _tok(s):
    return set(re.findall(r"[a-z]+", s.lower()))


@mcp.tool()
def check_deadline(item: str) -> dict:
    """Look up a live deadline (e.g. 'course withdrawal') from the deadlines table."""
    with _db() as con:
        rows = con.execute("SELECT item, due_date, policy_ref FROM deadlines").fetchall()
    q = _tok(item)
    best = max(rows, key=lambda r: len(q & _tok(r["item"])), default=None)
    if best is None or not q & _tok(best["item"]):
        return {"found": False}
    return {"found": True, **dict(best)}


@mcp.tool()
def get_request_status(request_id: str) -> dict:
    """Look up the status of a request by id, format REQ-<digits>."""
    if not re.fullmatch(r"REQ-\d+", request_id):
        return {"found": False, "error": "invalid request id format"}
    with _db() as con:
        row = con.execute("SELECT * FROM requests WHERE request_id = ?", (request_id,)).fetchone()
    if row:
        return {"found": True, **dict(row)}
    if ACTIONS_DB.exists():  # requests the user approved through PolicyLens itself
        with sqlite3.connect(f"file:{ACTIONS_DB}?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            r2 = con.execute("SELECT request_id, kind, status FROM drafts WHERE request_id = ?", (request_id,)).fetchone()
        if r2:
            return {"found": True, **dict(r2), "source": "approved in PolicyLens"}
    return {"found": False}


if __name__ == "__main__":
    mcp.run()
