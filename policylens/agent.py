"""Agent: talks to ALL MCP servers through MCPHost; the LLM decides which tools to call.
LLM mode needs a provider key (see README); offline=True uses a deterministic rule-based router (used by tests)."""
import asyncio, json, os, re, time
from pathlib import Path
from .mcp_host import MCPHost
from .prompts import SYSTEM

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.getenv("POLICYLENS_MODEL", "claude-sonnet-4-6")
LOG = ROOT / "logs" / "queries.jsonl"
ID_RE = r"\b[a-z0-9_]+#[a-z0-9-]+"  # citation id format: <doc>#<section-slug>


async def _call(s, name, args, trace, retrieved):
    try:
        r = await s.call_tool(name, args)  # host enforces the model-visible allowlist
        out = "".join(c.text for c in r.content if c.type == "text")
    except Exception as e:
        out = json.dumps({"error": f"{type(e).__name__}: {e}"})
    trace.append({"tool": name, "server": s.owner.get(name), "args": args, "ids": re.findall(ID_RE, out)})
    if name == "search_policy":
        try:
            retrieved.update(x["id"] for x in json.loads(out)["results"])
        except Exception:
            pass
    return out


async def _offline(s, q, trace, retrieved):
    if re.match(r"\s*draft\b", q, re.I):
        out = await _call(s, "draft_request", {"kind": "other", "summary": q}, trace, retrieved)
        return f"I created a draft. It is waiting for your approval below. {out}"
    hits = json.loads(await _call(s, "search_policy", {"query": q}, trace, retrieved))["results"]
    live = []
    if re.search(r"deadline|due|last date|when", q, re.I):
        live.append(await _call(s, "check_deadline", {"item": q}, trace, retrieved))
    if m := re.search(r"REQ-\d+", q):
        live.append(await _call(s, "get_request_status", {"request_id": m.group()}, trace, retrieved))
    if not hits:
        return "I cannot find this in the policy documents."
    return f"{hits[0]['text']} [{hits[0]['id']}]" + "".join(f"\nLive data: {x}" for x in live)


async def _llm_anthropic(s, q, trace, retrieved):
    import anthropic
    client = anthropic.AsyncAnthropic()
    tools = [{"name": t.name, "description": t.description, "input_schema": t.inputSchema}
             for t in (await s.list_tools()).tools]
    msgs = [{"role": "user", "content": q}]
    kw = dict(model=MODEL, max_tokens=1000, system=s.system, tools=tools)
    resp = await client.messages.create(messages=msgs, **kw)
    for _ in range(5):  # hard cap on tool-use turns
        if resp.stop_reason != "tool_use":
            break
        msgs.append({"role": "assistant", "content": resp.content})
        results = [{"type": "tool_result", "tool_use_id": b.id,
                    "content": await _call(s, b.name, b.input, trace, retrieved)}
                   for b in resp.content if b.type == "tool_use"]
        msgs.append({"role": "user", "content": results})
        resp = await client.messages.create(messages=msgs, **kw)
    return "".join(b.text for b in resp.content if b.type == "text")


PROVIDERS = {  # name: (base_url, api-key env var, default model) - all OpenAI-compatible; override model with POLICYLENS_MODEL
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-flash-latest"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "ollama": ("http://localhost:11434/v1", None, "llama3.1"),  # local, no key
}


def provider():
    """POLICYLENS_PROVIDER = groq|gemini|openrouter|ollama|anthropic, else auto-detect from API keys."""
    name = os.getenv("POLICYLENS_PROVIDER")
    if name:
        return name
    for n, key in [("groq", "GROQ_API_KEY"), ("gemini", "GEMINI_API_KEY"),
                   ("openrouter", "OPENROUTER_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")]:
        if os.getenv(key):
            return n
    return None


EXCLUDE = ("whisper", "tts", "guard", "embed", "orpheus", "compound", "playai", "vision", "safeguard", "moderation")
PREFER = ("gpt-oss-120b", "qwen3.6", "qwen3.8", "llama-3.3-70b", "gpt-oss-20b", "llama-3.1-8b")
_MODEL_CACHE = {}


async def _pick_model(client):
    """Ask the provider which models exist and choose the best tool-capable chat model."""
    ids = [m.id for m in (await client.models.list()).data if not any(x in m.id.lower() for x in EXCLUDE)]
    for pref in PREFER:
        for i in ids:
            if pref in i.lower():
                return i
    if ids:
        return ids[0]
    raise RuntimeError("The provider returned no usable chat models; set POLICYLENS_MODEL manually.")


async def _llm_openai(s, q, trace, retrieved, name):
    from openai import AsyncOpenAI, NotFoundError
    base_url, key_env, default_model = PROVIDERS[name]
    client = AsyncOpenAI(base_url=base_url, api_key=(os.getenv(key_env) if key_env else "ollama"))
    model = _MODEL_CACHE.get(name) or os.getenv("POLICYLENS_MODEL", default_model)
    tools = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}}
             for t in (await s.list_tools()).tools]
    msgs = [{"role": "system", "content": s.system}, {"role": "user", "content": q}]
    for _ in range(6):  # hard cap on tool-use turns
        try:
            resp = await client.chat.completions.create(model=model, messages=msgs, tools=tools)
        except NotFoundError:  # model retired or not on this account: discover a working one
            if os.getenv("POLICYLENS_MODEL"):
                raise
            model = _MODEL_CACHE[name] = await _pick_model(client)
            resp = await client.chat.completions.create(model=model, messages=msgs, tools=tools)
        m = resp.choices[0].message
        if not m.tool_calls:
            return m.content or ""
        msgs.append({"role": "assistant", "content": m.content, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in m.tool_calls]})
        for tc in m.tool_calls:
            out = await _call(s, tc.function.name, json.loads(tc.function.arguments or "{}"), trace, retrieved)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    # Tool budget exhausted: force a final answer from what was retrieved instead of failing silently.
    msgs.append({"role": "user", "content": "Tool budget used up. Answer now using only the tool results above, "
                 "citing ids in [square brackets]. If they do not contain the answer, say you cannot find it "
                 "in the policy documents."})
    final = await client.chat.completions.create(model=model, messages=msgs)
    return final.choices[0].message.content or "I cannot find this in the policy documents."


async def _llm(s, q, trace, retrieved):
    name = provider()
    if name is None:
        raise RuntimeError("No LLM configured. Set GROQ_API_KEY or GEMINI_API_KEY (free), or run Ollama locally "
                           "with POLICYLENS_PROVIDER=ollama. See README.")
    if name == "anthropic":
        return await _llm_anthropic(s, q, trace, retrieved)
    if name not in PROVIDERS:
        raise RuntimeError(f"Unknown POLICYLENS_PROVIDER '{name}'. Options: {', '.join(PROVIDERS)}, anthropic")
    return await _llm_openai(s, q, trace, retrieved, name)


async def answer_many(questions, offline=False, log=True):
    out = []
    async with MCPHost() as s:
        s.system = await s.system_prompt(SYSTEM)  # system prompt is served by the policy MCP server
        for q in questions:
            trace, retrieved, t0 = [], set(), time.time()
            try:
                text, err = await (_offline if offline else _llm)(s, q, trace, retrieved), None
            except Exception as e:  # surface the real error instead of an opaque ExceptionGroup
                text, err = "", f"{type(e).__name__}: {e}"
            allowed = retrieved | {i for t in trace for i in t["ids"]}  # ids that actually appeared in tool output
            cited = list(dict.fromkeys(c for c in re.findall(ID_RE, text) if c in allowed))
            rec = {"question": q, "answer": text, "citations": cited, "retrieved": sorted(retrieved),
                   "tools": [t["tool"] for t in trace], "servers": sorted({t["server"] for t in trace if t["server"]}),
                   "latency_s": round(time.time() - t0, 2), "error": err}
            out.append(rec)
            if log:  # KPI instrumentation: latency, tool usage, citation presence per query
                LOG.parent.mkdir(exist_ok=True)
                with LOG.open("a") as f:
                    f.write(json.dumps({**rec, "ts": time.time(), "mode": "offline" if offline else "llm"}) + "\n")
    return out


def _run(coro):
    try:
        return asyncio.run(coro)
    except BaseExceptionGroup as eg:  # unwrap to the real cause (e.g. an MCP server failed to start)
        leaf = eg
        while isinstance(leaf, BaseExceptionGroup):
            leaf = leaf.exceptions[0]
        raise RuntimeError(f"{type(leaf).__name__}: {leaf}") from None


def ask(question, offline=False):
    return _run(answer_many([question], offline))[0]


def host_call(server, name, args):
    """Called by the APP (not the model) for host-only tools, e.g. approving a draft. Starts only that server."""
    async def go():
        async with MCPHost(only=[server]) as h:
            r = await h.call_tool(name, args, internal=True)
            return json.loads("".join(c.text for c in r.content if c.type == "text"))
    return _run(go())


def mcp_info():
    """Inspect all configured MCP servers: tools (and whether model-visible) plus resources."""
    async def go():
        async with MCPHost() as h:
            servers = {}
            for n, t in h._tools.items():
                servers.setdefault(h.owner[n], []).append((n, n in h.hidden))
            return {"servers": servers, "resources": await h.resources()}
    return _run(go())
