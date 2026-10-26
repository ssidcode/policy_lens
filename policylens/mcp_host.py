"""MCP host: starts every server listed in mcp_servers.json, merges their tools, and enforces which tools the
model may call. Duck-types the bits of ClientSession the agent uses (list_tools / call_tool)."""
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "mcp_servers.json"
ENV = {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}


class MCPHost:
    def __init__(self, only=None, config_path=CONFIG):
        cfg = json.loads(Path(config_path).read_text())["servers"]
        self.cfg = [c for c in cfg if not only or c["name"] in only]
        self.sessions, self.owner, self._tools, self.hidden = {}, {}, {}, set()
        self.system = None
        self._stack = AsyncExitStack()

    async def __aenter__(self):
        await self._stack.__aenter__()
        try:
            for srv in self.cfg:  # a server is either a python module or any external command (third-party MCP servers)
                params = StdioServerParameters(
                    command=srv.get("command") or sys.executable,
                    args=srv.get("args") or ["-m", srv["module"]], cwd=str(ROOT),
                    env={**os.environ, **ENV, **srv.get("env", {})})
                r, w = await self._stack.enter_async_context(stdio_client(params))
                s = await self._stack.enter_async_context(ClientSession(r, w))
                await s.initialize()
                self.sessions[srv["name"]] = s
                for t in (await s.list_tools()).tools:
                    if t.name in self.owner:
                        raise RuntimeError(f"Tool name collision: '{t.name}' in {self.owner[t.name]} and {srv['name']}")
                    self.owner[t.name], self._tools[t.name] = srv["name"], t
                self.hidden |= set(srv.get("hidden_tools", []))
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc):
        return await self._stack.__aexit__(*exc)

    async def list_tools(self):
        """Only the tools the MODEL may see (host-only tools are filtered out)."""
        return SimpleNamespace(tools=[t for n, t in self._tools.items() if n not in self.hidden])

    async def call_tool(self, name, args, internal=False):
        if name not in self.owner:
            raise KeyError(f"Unknown tool '{name}'")
        if name in self.hidden and not internal:
            raise PermissionError(f"Tool '{name}' is host-only and cannot be called by the model")
        return await self.sessions[self.owner[name]].call_tool(name, args)

    async def resources(self):
        out = []
        for name, s in self.sessions.items():
            try:
                out += [(name, str(r.uri)) for r in (await s.list_resources()).resources]
                out += [(name, str(t.uriTemplate)) for t in (await s.list_resource_templates()).resourceTemplates]
            except Exception:
                pass
        return out

    async def read_resource(self, server, uri):
        r = await self.sessions[server].read_resource(AnyUrl(uri))
        return "".join(getattr(c, "text", "") for c in r.contents)

    async def system_prompt(self, fallback):
        for s in self.sessions.values():
            try:
                return (await s.get_prompt("policy_assistant")).messages[0].content.text
            except Exception:
                continue
        return fallback
