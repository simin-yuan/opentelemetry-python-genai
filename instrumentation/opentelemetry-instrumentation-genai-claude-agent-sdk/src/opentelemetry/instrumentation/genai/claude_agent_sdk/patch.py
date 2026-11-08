# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
"""Prototype instrumentation for the Claude Agent SDK.

Only public entry points are wrapped: ``claude_agent_sdk.query`` and the
``ClaudeSDKClient`` message-receiving methods. The SDK's ``InternalClient`` is
deliberately left alone.

Token accounting: the ``ResultMessage`` carries the run totals and is treated
as authoritative. Usage reported on individual ``AssistantMessage`` objects is
summed only as a fallback for runs that end without a ``ResultMessage``
(aborted or closed early), because counting both would double count.
"""

from __future__ import annotations

import contextvars
import importlib
from collections.abc import AsyncIterator
from typing import Any, cast

from wrapt import FunctionWrapper

from opentelemetry import trace
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.stream import AsyncStreamWrapper
from opentelemetry.util.genai.types import OutputMessage, TextPart

PROVIDER = "anthropic"

# ``receive_response`` iterates ``receive_messages`` internally, so wrapping both
# would report two spans for one turn. The flag makes the inner call a plain
# pass-through.
_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "claude_agent_sdk_instrumented", default=False
)

# In-process SDK MCP tools: tool name -> server instance, plus the invocation of
# the turn currently in flight for each server instance. The SDK calls MCP
# handlers from its read loop, a task created at connect time, so the OTel
# context there never holds the agent span; the span is looked up per server and
# passed explicitly as the parent instead.
_MCP_TOOL_SERVERS: dict[str, Any] = {}
_SERVER_TURNS: dict[int, Any] = {}


def _server_for_tool(name: str) -> Any:
    """Server owning a tool. Accepts both the declared and the mcp__x__y form."""
    if name in _MCP_TOOL_SERVERS:
        return _MCP_TOOL_SERVERS[name]
    if name.startswith("mcp__"):
        return _MCP_TOOL_SERVERS.get(name.split("__")[-1])
    return None


def _is_in_process_tool(name: str) -> bool:
    """Only the stream form needs suppressing; the handler is already traced."""
    return name.startswith("mcp__") and _server_for_tool(name) is not None


def _server_instances(options: Any) -> list[Any]:
    servers = getattr(options, "mcp_servers", None) or {}
    instances = []
    for config in servers.values():
        instance = config.get("instance") if isinstance(config, dict) else config
        if instance is not None:
            instances.append(instance)
    return instances


def _register_turn(options: Any, invocation: Any) -> None:
    for instance in _server_instances(options):
        _SERVER_TURNS[id(instance)] = invocation


def _clear_turn(options: Any) -> None:
    for instance in _server_instances(options):
        _SERVER_TURNS.pop(id(instance), None)


def _tool_parent_context(tool_name: str) -> Any:
    server = _server_for_tool(tool_name)
    if server is None:
        return None
    span = getattr(_SERVER_TURNS.get(id(server)), "span", None)
    if span is None:
        return None
    return trace.set_span_in_context(span)


def _usage_of(message: Any) -> dict[str, Any]:
    raw: Any = getattr(message, "usage", None)
    if not isinstance(raw, dict):
        return {}
    return dict(cast("dict[str, Any]", raw))


def _blocks(message: Any) -> list[Any]:
    raw: Any = getattr(message, "content", None)
    if not isinstance(raw, list):
        return []
    return list(cast("list[Any]", raw))


def _message_kind(chunk: Any) -> str:
    return type(chunk).__name__


class ClaudeAgentStreamWrapper(AsyncStreamWrapper[Any]):
    """Turns a Claude Agent SDK message stream into one invoke_agent span."""

    def __init__(
        self,
        stream: Any,
        invocation: Any,
        handler: TelemetryHandler,
        turn_options: Any = None,
    ) -> None:
        super().__init__(stream, invocation)
        self._self_turn_options = turn_options
        # Output accumulation lives here rather than in a shared helper: the base
        # wrapper only scopes the invocation, it does not know about agent output.
        self._self_out_parts: list[str] = []
        invocation.suspend()
        self._self_agent: Any = invocation
        self._self_handler = handler
        self._self_result_usage: dict[str, Any] | None = None
        self._self_assistant_input = 0
        self._self_assistant_output = 0
        self._self_completed = False
        # tool_use id -> ToolInvocation, for tools executed outside this process
        self._self_pending_tools: dict[str, Any] = {}
        # Set when a ResultMessage closes the turn; a multi-turn stream keeps
        # going after that, so the wrapper has to stop at this point.
        self._self_turn_done = False

    def _process_chunk(self, chunk: Any) -> None:
        # Newer util releases scope the invocation around this hook; older ones
        # do not, and there the tool spans would be created without a parent.
        # Activating here is harmless when the base already did it.
        activate = getattr(self._self_agent, "activate", None)
        if activate is None:
            self._handle_chunk(chunk)
            return
        with activate():
            self._handle_chunk(chunk)

    def _handle_chunk(self, chunk: Any) -> None:
        kind = _message_kind(chunk)

        if kind == "SystemMessage":
            return

        if kind == "AssistantMessage":
            usage = _usage_of(chunk)
            self._self_assistant_input += usage.get("input_tokens") or 0
            self._self_assistant_output += usage.get("output_tokens") or 0
            for block in _blocks(chunk):
                text = getattr(block, "text", None)
                if text:
                    self._add_text(text)
                if type(block).__name__ == "ToolUseBlock":
                    self._start_tool(block)
            return

        if kind == "UserMessage":
            for block in _blocks(chunk):
                if type(block).__name__ == "ToolResultBlock":
                    self._end_tool(block)
            return

        if kind == "ResultMessage":
            self._self_completed = True
            self._self_turn_done = True
            usage = _usage_of(chunk)
            if usage:
                self._self_result_usage = usage
            stop_reason = getattr(chunk, "stop_reason", None)
            if stop_reason:
                self._self_agent.response_finish_reasons = [stop_reason]

    def _start_tool(self, block: Any) -> None:
        tool_id = getattr(block, "id", None)
        name = getattr(block, "name", None) or "unknown"
        if not tool_id:
            return
        if _is_in_process_tool(name):
            # Instrumented at the handler; recording the observed interval too
            # would produce two execute_tool spans for one call.
            return
        # Observed interval only: the tool runs inside the CLI, so the span
        # covers tool_use -> tool_result, not the tool's own execution time.
        invocation = self._self_handler.tool(
            name=name, tool_call_id=tool_id, tool_type="function"
        )
        self._self_pending_tools[tool_id] = invocation

    def _end_tool(self, block: Any) -> None:
        tool_id = getattr(block, "tool_use_id", None)
        if not isinstance(tool_id, str):
            return
        invocation = self._self_pending_tools.pop(tool_id, None)
        if invocation is None:
            return
        if getattr(block, "is_error", False):
            invocation.fail(RuntimeError("tool reported an error"))
        else:
            invocation.stop()

    def _finalize_pending_tools(
        self, error: BaseException | None = None
    ) -> None:
        while self._self_pending_tools:
            _, invocation = self._self_pending_tools.popitem()
            try:
                if error is None:
                    invocation.stop()
                else:
                    invocation.fail(error)
            except Exception:
                pass

    def _apply_usage(self) -> None:
        invocation = self._self_agent
        if self._self_result_usage is not None:
            usage = self._self_result_usage
            invocation.input_tokens = usage.get("input_tokens")
            invocation.output_tokens = usage.get("output_tokens")
            cache_read = usage.get("cache_read_input_tokens")
            if cache_read is not None:
                invocation.cache_read_input_tokens = cache_read
            cache_write = usage.get("cache_creation_input_tokens")
            if cache_write is not None:
                invocation.cache_write_input_tokens = cache_write
            return
        if self._self_assistant_input or self._self_assistant_output:
            invocation.input_tokens = self._self_assistant_input
            invocation.output_tokens = self._self_assistant_output

    def _add_text(self, text: str) -> None:
        if self._self_agent.should_capture_content:
            self._self_out_parts.append(text)

    def _apply_output(self) -> None:
        if not self._self_out_parts:
            return
        if not self._self_agent.should_capture_content:
            return
        text = "".join(self._self_out_parts)
        self._self_out_parts = []
        self._self_agent.output_messages = [
            OutputMessage(role="assistant", parts=[TextPart(content=text)])
        ]

    def end_turn(self) -> None:
        """Finalize this turn's span; the underlying stream stays open."""
        self._apply_usage()
        self._finalize_pending_tools()
        self._apply_output()
        _clear_turn(self._self_turn_options)
        self._self_agent.stop()

    def _on_stream_end(self) -> None:
        self.end_turn()

    def _on_stream_error(self, error: BaseException) -> None:
        self._apply_usage()
        self._finalize_pending_tools(error)
        self._apply_output()
        _clear_turn(self._self_turn_options)
        self._self_agent.fail(error)


def _wrap_query(handler: TelemetryHandler) -> Any:
    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        stream = wrapped(*args, **kwargs)
        options = kwargs.get("options")
        invocation = handler.invoke_local_agent(
            request_model=getattr(options, "model", None),
            agent_name=getattr(options, "agent_name", None)
            or "claude-agent-sdk",
        )
        _register_turn(options, invocation)
        return ClaudeAgentStreamWrapper(
            stream, invocation, handler, turn_options=options
        )

    return wrapper


_PATCHED: list[tuple[Any, str, Any]] = []


def _patch(module_name: str, attr: str, handler: TelemetryHandler) -> None:
    module = importlib.import_module(module_name)
    original = getattr(module, attr)
    if isinstance(original, FunctionWrapper):
        return
    setattr(module, attr, FunctionWrapper(original, _wrap_query(handler)))
    _PATCHED.append((module, attr, original))


async def _multi_turn_messages(
    stream: Any, handler: TelemetryHandler, options: Any, token: Any
) -> AsyncIterator[Any]:
    """One span per turn: ``receive_messages`` covers a whole session.

    The underlying iterator is pulled directly so that no invocation is
    created for a turn that never produces a message.
    """
    aiter = stream.__aiter__()
    wrapper: ClaudeAgentStreamWrapper | None = None
    try:
        while True:
            try:
                chunk = await aiter.__anext__()
            except StopAsyncIteration:
                if wrapper is not None:
                    wrapper.end_turn()
                return
            if wrapper is None:
                invocation = handler.invoke_local_agent(
                    request_model=getattr(options, "model", None),
                    agent_name="claude-agent-sdk",
                )
                _register_turn(options, invocation)
                wrapper = ClaudeAgentStreamWrapper(
                    aiter, invocation, handler, turn_options=options
                )
            wrapper._process_chunk(chunk)
            yield chunk
            if wrapper._self_turn_done:
                wrapper.end_turn()
                wrapper = None
    except BaseException as error:
        if wrapper is not None:
            try:
                wrapper._on_stream_error(error)
            except Exception:
                pass
        raise
    finally:
        if wrapper is not None:
            try:
                wrapper.end_turn()
            except Exception:
                pass
        _ACTIVE.reset(token)


def _wrap_client_iterator(handler: TelemetryHandler) -> Any:
    """Wrap a ``ClaudeSDKClient`` message iterator once, without nesting."""

    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        stream = wrapped(*args, **kwargs)
        if _ACTIVE.get():
            return stream
        token = _ACTIVE.set(True)
        return _multi_turn_messages(
            stream, handler, getattr(instance, "options", None), token
        )

    return wrapper


def _wrap_tool_decorator(handler: TelemetryHandler) -> Any:
    """SDK MCP tools run in this process, so their real duration is observable."""

    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        decorator = wrapped(*args, **kwargs)
        tool_name = args[0] if args else kwargs.get("name", "unknown")

        def decorator_wrapper(func: Any) -> Any:
            async def traced(*call_args: Any, **call_kwargs: Any) -> Any:
                # The SDK runs handlers from its read loop, which holds no agent
                # span, so the parent is looked up per server and passed in.
                invocation = handler.tool(
                    name=tool_name,
                    tool_type="function",
                    context=_tool_parent_context(tool_name),
                )
                try:
                    result = await func(*call_args, **call_kwargs)
                except BaseException as error:
                    invocation.fail(error)
                    raise
                invocation.stop()
                return result

            return decorator(traced)

        return decorator_wrapper

    return wrapper


def _wrap_mcp_server() -> Any:
    """Remember which server each in-process tool belongs to."""

    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        config = wrapped(*args, **kwargs)
        server = config.get("instance") if isinstance(config, dict) else None
        if server is None:
            return config
        passed = list(kwargs.get("tools") or [])
        for value in args:
            if isinstance(value, (list, tuple)):
                passed.extend(value)
        for sdk_tool in passed:
            name = getattr(sdk_tool, "name", None)
            if name:
                _MCP_TOOL_SERVERS[name] = server
        return config

    return wrapper


def instrument(handler: TelemetryHandler) -> None:
    for module_name in ("claude_agent_sdk.query", "claude_agent_sdk"):
        _patch(module_name, "query", handler)
    client = importlib.import_module("claude_agent_sdk")
    original_tool = getattr(client, "tool", None)
    if original_tool is not None and not isinstance(
        original_tool, FunctionWrapper
    ):
        setattr(
            client,
            "tool",
            FunctionWrapper(original_tool, _wrap_tool_decorator(handler)),
        )
        _PATCHED.append((client, "tool", original_tool))
    original_server = getattr(client, "create_sdk_mcp_server", None)
    if original_server is not None and not isinstance(
        original_server, FunctionWrapper
    ):
        setattr(
            client,
            "create_sdk_mcp_server",
            FunctionWrapper(original_server, _wrap_mcp_server()),
        )
        _PATCHED.append((client, "create_sdk_mcp_server", original_server))
    for attr in ("receive_response", "receive_messages"):
        original = getattr(client.ClaudeSDKClient, attr)
        if isinstance(original, FunctionWrapper):
            continue
        setattr(
            client.ClaudeSDKClient,
            attr,
            FunctionWrapper(original, _wrap_client_iterator(handler)),
        )
        _PATCHED.append((client.ClaudeSDKClient, attr, original))


def uninstrument() -> None:
    _MCP_TOOL_SERVERS.clear()
    _SERVER_TURNS.clear()
    while _PATCHED:
        module, attr, original = _PATCHED.pop()
        setattr(module, attr, original)
