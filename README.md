# PolicyLens - Compliance Q&A Agent with MCP Tool Integration

RAG agent over policy documents. Claude decides per question whether to retrieve policy text, call a live-data tool, or both, and answers with `[doc#section]` citations.

## Architecture
```
Streamlit UI -> Agent (Claude tool-use loop, max 5 turns)
                   | MCP (stdio)
                   v
        MCP server: search_policy | check_deadline | get_request_status
                   |                         |
          FAISS vector index          SQLite (read-only connection)
          (sentence-transformer       (data/policy.db)
          embeddings; TF-IDF fallback)
```

## Run
```
pip install -r requirements.txt
python scripts/seed_db.py
export ANTHROPIC_API_KEY=...        # optional; without it the app runs in offline mode
streamlit run app.py
pytest -q                           # tests + end-to-end eval
python -m eval.run_eval --llm       # eval against the real LLM
```

## Replace the sample data
Drop real policy `.md` files (use `## Section` headings) into `data/docs/`, update `scripts/seed_db.py`, and extend `eval/eval_set.json` (target 50 queries).

## Known limitations & risk flags
- Offline mode is a rule-based router for testing, not an LLM; LLM-mode quality must be measured separately.
- Retrieval is dense (all-MiniLM-L6-v2 + FAISS) with a TF-IDF fallback (`POLICYLENS_RETRIEVER=auto|vector|tfidf`). The first vector run downloads the model; the index is rebuilt in memory at startup (fine for small corpora, persist it for larger ones). Tests use a deterministic hashing embedder, so they verify the FAISS plumbing, not semantic quality - measure that with `python -m eval.run_eval` on your real documents.
- `min_score` thresholds (0.35 vector / 0.05 TF-IDF) drive the refusal path and should be tuned on your eval set.
- Hallucination risk in LLM mode: mitigated by a system prompt restricting answers to tool output, a citation filter (only retrieved ids count), and a refusal path when retrieval score < threshold.
- Tools are read-only: SQLite opened with `mode=ro`, parameterized queries, strict request-id validation.
- Sample documents and dates are fictional; do not treat outputs as real policy.
- Decision log: stdio MCP over HTTP (simplest to run locally); section-level chunking (policies are short, sections are the natural citation unit).

## KPI instrumentation
Every query appends latency, tools used, and citations to `logs/queries.jsonl` (supports adoption, productivity, output-quality metrics).
