# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process SDK MCP tools: one span per call, nested under the current turn.

The SDK invokes MCP handlers from the read loop, a task created at connect time,
so the tools are called from a task that exists before any span does -- the
shape every test below reproduces.
"""

from __future__ import annotations

from typing import Any

import anyio
import claude_agent_sdk
from claude_agent_sdk.types import ClaudeAgentOptions

from ._replay import ReplayTransport, single_turn


def _make_tool(name: str):
    @claude_agent_sdk.tool(name, f"{name} tool", {"a": int})
    async def handler(args):
        return {"content": [{"type": "text", "text": "ok"}]}

    return handler


def _mcp_stream(tool_name: str = "add") -> list[dict[str, Any]]:
    """A turn whose tool call is an in-process MCP tool (mcp__<server>__<tool>)."""
    return single_turn(
        tool_name=f"mcp__demo__{tool_name}", tool_use_id="toolu_mcp"
    )


def _tool_spans(span_exporter) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.name.startswith("execute_tool")
    ]


def _agent_spans(span_exporter) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.name.startswith("invoke_agent")
    ]


async def _call_after(delay: float, tool) -> None:
    await anyio.sleep(delay)
    await tool.handler({"a": 1})


def test_in_process_tool_emits_exactly_one_span(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    tool = _make_tool("add")
    server = claude_agent_sdk.create_sdk_mcp_server(name="demo", tools=[tool])
    options = ClaudeAgentOptions(mcp_servers={"demo": server})

    async def _run() -> None:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_call_after, 0.06, tool)
            async for _ in claude_agent_sdk.query(
                prompt="x",
                options=options,
                transport=ReplayTransport(_mcp_stream()),
            ):
                pass

    anyio.run(_run)

    # The stream reports mcp__demo__add; the handler span is the only one kept.
    tools = _tool_spans(span_exporter)
    assert [span.name for span in tools] == ["execute_tool add"]


def test_in_process_tool_nests_under_current_turn(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    tool = _make_tool("add")
    server = claude_agent_sdk.create_sdk_mcp_server(name="demo", tools=[tool])
    options = ClaudeAgentOptions(mcp_servers={"demo": server})

    async def _run() -> None:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_call_after, 0.06, tool)
            async for _ in claude_agent_sdk.query(
                prompt="x",
                options=options,
                transport=ReplayTransport(_mcp_stream(), delay=0.03),
            ):
                pass

    anyio.run(_run)

    agent = _agent_spans(span_exporter)[0]
    (span,) = _tool_spans(span_exporter)
    assert span.parent is not None
    assert span.parent.span_id == agent.context.span_id


def test_two_clients_keep_their_own_turn(
    span_exporter, instrument_claude_agent_sdk
):
    async def _client(tag: str, delay: float) -> None:
        tool = _make_tool(tag)
        server = claude_agent_sdk.create_sdk_mcp_server(name=tag, tools=[tool])
        options = ClaudeAgentOptions(
            mcp_servers={tag: server}, model=f"{tag}-model"
        )

        async with anyio.create_task_group() as tg:
            tg.start_soon(_call_after, delay, tool)
            async for _ in claude_agent_sdk.query(
                prompt="x",
                options=options,
                transport=ReplayTransport(_mcp_stream(tag), delay=0.03),
            ):
                pass

    async def _run() -> None:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_client, "alpha", 0.06)
            tg.start_soon(_client, "beta", 0.10)

    anyio.run(_run)

    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    by_id = {
        span.context.span_id: span
        for span in span_exporter.get_finished_spans()
    }
    alpha_parent = by_id[spans["execute_tool alpha"].parent.span_id]
    beta_parent = by_id[spans["execute_tool beta"].parent.span_id]
    assert alpha_parent.attributes["gen_ai.request.model"] == "alpha-model"
    assert beta_parent.attributes["gen_ai.request.model"] == "beta-model"


def test_second_turn_tool_nests_under_second_turn(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    tool = _make_tool("add")
    server = claude_agent_sdk.create_sdk_mcp_server(name="demo", tools=[tool])
    options = ClaudeAgentOptions(mcp_servers={"demo": server})

    async def _run() -> None:
        for _ in range(2):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_call_after, 0.06, tool)
                async for _ in claude_agent_sdk.query(
                    prompt="x",
                    options=options,
                    transport=ReplayTransport(_mcp_stream(), delay=0.03),
                ):
                    pass

    anyio.run(_run)

    agents = sorted(
        _agent_spans(span_exporter), key=lambda span: span.start_time
    )
    tools = sorted(
        _tool_spans(span_exporter), key=lambda span: span.start_time
    )
    by_id = {
        span.context.span_id: span
        for span in span_exporter.get_finished_spans()
    }
    assert len(agents) == 2
    assert len(tools) == 2
    assert by_id[tools[0].parent.span_id] is agents[0]
    assert by_id[tools[1].parent.span_id] is agents[1]
