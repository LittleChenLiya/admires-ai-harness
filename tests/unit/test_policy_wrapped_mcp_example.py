from __future__ import annotations

import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest

from admires_harness.adapters.mcp import MCPToolExecutor
from admires_harness.contracts import ToolCall

EXAMPLE_PATH = Path(__file__).parents[2] / "examples" / "policy_wrapped_mcp.py"
SPEC = importlib.util.spec_from_file_location("policy_wrapped_mcp", EXAMPLE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
secure_readonly_tool = MODULE.secure_readonly_tool


class OfflineClient:
    async def list_tools(self):
        return {
            "tools": [
                {"name": "inspect", "inputSchema": {"type": "object"}},
                {"name": "mutate", "inputSchema": {"type": "object"}},
            ]
        }

    async def call_tool(self, name, arguments=None):
        return {"content": [{"text": f"inspected:{name}"}], "isError": False}


@pytest.mark.asyncio
async def test_example_namespaces_allowlists_and_applies_policy() -> None:
    executor = await MCPToolExecutor.connect(
        OfflineClient(),
        server_name="local_demo",
        allow_tools=frozenset({"inspect"}),
    )
    secured = secure_readonly_tool(executor, "local_demo__inspect")

    assert [schema["function"]["name"] for schema in secured.schemas()] == ["local_demo__inspect"]
    allowed = await secured.execute(
        ToolCall(name="local_demo__inspect"),
        run_id=uuid4(),
        idempotency_key="allowed",
    )
    denied = await secured.execute(
        ToolCall(name="local_demo__mutate"),
        run_id=uuid4(),
        idempotency_key="denied",
    )

    assert allowed.ok and allowed.content == "inspected:inspect"
    assert denied.error_code == "policy_denied"
