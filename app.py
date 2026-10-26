import os, streamlit as st
from policylens.agent import ask, provider
from policylens.ingest import save_document, list_documents, delete_document

st.set_page_config(page_title="PolicyLens", page_icon="📜")
st.title("📜 PolicyLens")
st.caption("Policy Q&A with source citations and live MCP tool lookups")
offline = st.sidebar.toggle("Offline mode (no LLM)", value=provider() is None)
st.sidebar.caption(f"LLM provider: {provider() or 'none (offline only)'}")
with st.sidebar.expander("📁 Knowledge base", expanded=False):
    files = st.file_uploader("Add documents (.md, .txt, .pdf)", type=["md", "txt", "pdf"], accept_multiple_files=True)
    if files and st.button("Ingest"):
        for f in files:
            try:
                doc, n = save_document(f.name, f.getvalue())
                st.success(f"{doc}: {n} sections")
            except Exception as e:
                st.error(f"{f.name}: {e}")
    for doc, n in list_documents():
        c1, c2 = st.columns([4, 1])
        c1.caption(f"{doc} ({n})")
        if c2.button("🗑", key=f"del_{doc}", help=f"Delete {doc}"):
            delete_document(doc)
            st.rerun()
q = st.text_input("Ask a policy question", placeholder="When is the course withdrawal deadline?")
if q:
    with st.spinner("Thinking..."):
        r = ask(q, offline=offline)
    if r.get("error"):
        st.error(r["error"])
        if provider() is None:
            st.warning("No LLM key found. Set GROQ_API_KEY or GEMINI_API_KEY in the terminal that launched Streamlit.")
        st.stop()
    st.write(r["answer"])
    st.markdown("**Sources:** " + (", ".join(f"`{c}`" for c in r["citations"]) or "none"))
    with st.expander("Tool trace"):
        st.json({"tools": r["tools"], "latency_s": r["latency_s"], "retrieved": r["retrieved"]})
