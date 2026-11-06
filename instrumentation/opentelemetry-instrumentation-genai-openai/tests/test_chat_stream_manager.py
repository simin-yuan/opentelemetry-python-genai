# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""The span must end when ``chat.completions.stream()`` leaves its ``with`` block.

The SDK's ``.stream()`` helper returns a manager whose ``close()`` closes the
underlying HTTP response instead of the stream we wrap. Closing that response
therefore has to finalize the invocation; otherwise a caller that breaks out of
the loop early leaves the span to the wrapper's ``__del__``, which records it
as an abandoned stream (error status) once the object is garbage collected.

These tests use a mock transport so they run without a network or cassette.
"""

from __future__ import annotations

import gc

import openai
import pytest

try:
    # openai v3 environments install httpx2 and drop httpx.
    import httpx as _http_lib
except ImportError:  # pragma: no cover - exercised on the latest openai env
    import httpx2 as _http_lib

from opentelemetry.trace import StatusCode

_CHUNK = (
    '{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,'
    '"model":"gpt-4o-mini","choices":[{"index":0,'
    '"delta":{"content":"Hi"},"finish_reason":null}]}'
)
_BODY = f"data: {_CHUNK}\n\ndata: {_CHUNK}\n\ndata: [DONE]\n\n".encode()
_HEADERS = {"content-type": "text/event-stream"}
_MESSAGES = [{"role": "user", "content": "hi"}]


def _stream_helper_available() -> bool:
    client = openai.OpenAI(api_key="test", base_url="http://test/v1")
    return hasattr(client.chat.completions, "stream")


pytestmark = pytest.mark.skipif(
    not _stream_helper_available(),
    reason="openai versions before the chat.completions.stream() helper",
)


def _response(request: _http_lib.Request) -> _http_lib.Response:
    return _http_lib.Response(200, content=_BODY, headers=_HEADERS)


def _sync_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=_http_lib.Client(
            transport=_http_lib.MockTransport(_response)
        ),
    )


def _async_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=_http_lib.AsyncClient(
            transport=_http_lib.MockTransport(_response)
        ),
    )


def _assert_single_chat_span(span_exporter) -> None:
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.request.stream"] is True
    assert span.status.status_code is StatusCode.UNSET
    # Finalization is once-only: collecting the wrapper must not add a span.
    gc.collect()
    assert len(span_exporter.get_finished_spans()) == 1


def test_stream_helper_finalizes_when_the_context_manager_exits(
    instrument_with_content, span_exporter
) -> None:
    client = _sync_client()
    with client.chat.completions.stream(
        model="gpt-4o-mini", messages=_MESSAGES
    ) as stream:
        for _ in stream:
            break

    _assert_single_chat_span(span_exporter)


def test_stream_helper_finalizes_after_a_full_drain(
    instrument_with_content, span_exporter
) -> None:
    client = _sync_client()
    with client.chat.completions.stream(
        model="gpt-4o-mini", messages=_MESSAGES
    ) as stream:
        assert list(stream)

    _assert_single_chat_span(span_exporter)


@pytest.mark.asyncio
async def test_async_stream_helper_finalizes_when_the_context_manager_exits(
    instrument_with_content, span_exporter
) -> None:
    client = _async_client()
    async with client.chat.completions.stream(
        model="gpt-4o-mini", messages=_MESSAGES
    ) as stream:
        async for _ in stream:
            break

    _assert_single_chat_span(span_exporter)


@pytest.mark.asyncio
async def test_async_stream_helper_finalizes_after_a_full_drain(
    instrument_with_content, span_exporter
) -> None:
    client = _async_client()
    async with client.chat.completions.stream(
        model="gpt-4o-mini", messages=_MESSAGES
    ) as stream:
        assert [event async for event in stream]

    _assert_single_chat_span(span_exporter)


def test_stream_helper_response_proxy_keeps_sdk_response_behaviour(
    instrument_with_content, span_exporter
) -> None:
    """Closing the response ends the span without changing what callers see.

    ``ChatCompletionStream`` reads ``raw_stream.response`` and closes it, so the
    instrumented stream exposes a proxy there. Everything a caller does with an
    httpx response has to keep working through it.
    """
    client = _sync_client()
    with client.chat.completions.stream(
        model="gpt-4o-mini", messages=_MESSAGES
    ) as stream:
        response = stream._raw_stream.response
        # Same object on every access, like the raw response.
        assert response is stream._raw_stream.response
        assert isinstance(response, _http_lib.Response)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        assert response.url.path == "/v1/chat/completions"
        assert response.request is not None
        assert response.read()
        assert hasattr(response, "iter_lines")
        for _ in stream:
            break

    assert span_exporter.get_finished_spans()[0].status.status_code is (
        StatusCode.UNSET
    )


def test_stream_helper_finalizes_once_when_closed_repeatedly(
    instrument_with_content, span_exporter
) -> None:
    client = _sync_client()
    with client.chat.completions.stream(
        model="gpt-4o-mini", messages=_MESSAGES
    ) as stream:
        list(stream)
    # The manager already closed on exit; closing again must not add a span.
    stream.close()
    stream.close()
    gc.collect()

    _assert_single_chat_span(span_exporter)


def test_stream_helper_ends_the_span_when_the_block_raises(
    instrument_with_content, span_exporter
) -> None:
    """An exception inside the block still ends the span exactly once.

    The SDK's manager closes the response from ``__exit__`` without passing the
    exception along, so the instrumentation sees a close and not a failure; the
    span ends with the default status. Ending it here is what keeps it from
    being exported later as an abandoned stream.
    """
    client = _sync_client()
    with pytest.raises(RuntimeError):
        with client.chat.completions.stream(
            model="gpt-4o-mini", messages=_MESSAGES
        ) as stream:
            for _ in stream:
                break
            raise RuntimeError("caller failed")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    gc.collect()
    assert len(span_exporter.get_finished_spans()) == 1
