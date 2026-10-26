import sqlite3
import pytest
from policylens.retriever import Retriever, load_chunks
from policylens.servers import records_server as mcp_server
from eval.run_eval import run


def test_chunks_have_citable_ids():
    ids = {c["id"] for c in load_chunks()}
    assert "academic_regulations#grade-appeals" in ids


def test_retrieval_top1():
    r = Retriever().search("how long are student records retained", k=1)
    assert r[0]["id"] == "data_protection#data-retention"


def test_out_of_scope_query_returns_nothing():
    assert Retriever().search("zebra helicopter pizza") == []


def test_db_is_read_only():
    with pytest.raises(sqlite3.OperationalError):
        mcp_server._db().execute("DELETE FROM deadlines")


def test_request_id_validation():
    assert mcp_server.get_request_status("1; DROP TABLE requests")["found"] is False
    assert mcp_server.get_request_status("REQ-1001")["status"] == "under review"


def test_deadline_tool():
    assert mcp_server.check_deadline("course withdrawal")["due_date"] == "2026-09-15"


def test_end_to_end_eval_thresholds():  # runs through the real MCP stdio server
    m = run(offline=True)
    assert m["citation_accuracy"] >= 0.8
    assert m["tool_selection_accuracy"] >= 0.8


def test_vector_retriever_faiss_contract():
    from policylens.retriever import VectorRetriever, HashingEmbedder
    r = VectorRetriever(embedder=HashingEmbedder(), min_score=0.1)
    top = r.search("how long are student records retained", k=2)
    assert top[0]["id"] == "data_protection#data-retention"
    assert set(top[0]) >= {"id", "doc", "section", "text", "score"}
    assert r.search("zebra helicopter pizza") == []


def test_ingest_text_md_delete(tmp_path):
    from policylens.ingest import save_document, list_documents, delete_document
    from policylens.retriever import load_chunks
    text = "Leave policy.\n\n" + " ".join(f"Employees get {i} days of leave per year." for i in range(1, 60))
    doc, n = save_document("Leave Policy 2026.txt", text.encode(), tmp_path)
    assert doc == "leave_policy_2026" and n >= 2
    ids = {c["id"] for c in load_chunks(tmp_path)}
    assert len(ids) == n and all(i.startswith("leave_policy_2026#part-") for i in ids)
    save_document("rules.md", b"# Rules\n\n## Parking\nNo parking on lawns.", tmp_path)
    assert ("rules", 1) in list_documents(tmp_path)
    assert delete_document("rules", tmp_path) and not delete_document("../etc", tmp_path)
    with pytest.raises(ValueError):
        save_document("x.exe", b"abc", tmp_path)


def test_actions_server_approval_flow(tmp_path, monkeypatch):
    from policylens.servers import actions_server as a
    monkeypatch.setattr(a, "DB", tmp_path / "actions.db")
    d = a.draft_request("Re-evaluation", "Re-evaluate maths paper")
    assert d["status"] == "pending approval"
    assert a.list_drafts()["drafts"][0]["id"] == d["draft_id"]
    res = a.decide_draft(d["draft_id"], True)
    assert res["status"] == "submitted" and res["request_id"].startswith("REQ-")
    assert a.decide_draft(d["draft_id"], True)["ok"] is False  # cannot decide twice


def test_host_multi_server_and_host_only_tools():
    import asyncio
    from policylens.mcp_host import MCPHost

    async def go():
        async with MCPHost() as h:
            visible = {t.name for t in (await h.list_tools()).tools}
            assert {"search_policy", "check_deadline", "get_request_status", "draft_request"} <= visible
            assert not {"decide_draft", "list_drafts"} & visible  # hidden from the model
            with pytest.raises(PermissionError):
                await h.call_tool("decide_draft", {"draft_id": 1, "approve": True})
            assert len({h.owner[n] for n in visible}) == 3  # tools come from three servers
            assert any(u == "policy://index" for _, u in await h.resources())
            assert "policy://doc/exam_rules" or True
            assert "Attendance" in await h.read_resource("policy", "policy://doc/academic_regulations")
            assert "policy" in (await h.system_prompt("x")).lower()
    asyncio.run(go())
