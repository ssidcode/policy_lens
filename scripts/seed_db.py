"""Create the sample SQLite DB the MCP tools read from."""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "policy.db"
DB.unlink(missing_ok=True)
con = sqlite3.connect(DB)
con.executescript("""
CREATE TABLE deadlines(item TEXT, due_date TEXT, policy_ref TEXT);
CREATE TABLE requests(request_id TEXT PRIMARY KEY, kind TEXT, status TEXT, updated TEXT);
""")
con.executemany("INSERT INTO deadlines VALUES (?,?,?)", [
    ("course withdrawal", "2026-09-15", "academic_regulations#course-withdrawal"),
    ("re-evaluation request", "2026-12-20", "academic_regulations#re-evaluation"),
    ("grade appeal filing", "2026-12-17", "academic_regulations#grade-appeals"),
    ("semester fee payment", "2026-08-30", None),
])
con.executemany("INSERT INTO requests VALUES (?,?,?,?)", [
    ("REQ-1001", "re-evaluation", "under review", "2026-10-20"),
    ("REQ-1002", "grade appeal", "escalated to Appeals Committee", "2026-10-22"),
    ("REQ-1003", "data access request", "completed", "2026-10-18"),
])
con.commit()
print("Seeded", DB)
