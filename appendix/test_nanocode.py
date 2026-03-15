"""Tests for Nanocode Streaming (Appendix A)."""

import json
from nanocode import Thought, ToolCall


# --- Test Helpers ---

def parse_sse_events(raw_lines):
    """Parse raw SSE lines into a list of event dicts."""
    events = []
    for line in raw_lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        try:
            events.append(json.loads(line[6:]))
        except (json.JSONDecodeError, ValueError):
            continue
    return events


def build_thought_from_events(events, print_fn=None, thinking_print_fn=None):
    """Build a Thought from parsed SSE events. Returns (Thought, input_tokens)."""
    text_parts = []
    thinking_parts = []
    raw_content = []
    tool_calls = []
    current_tool = None
    input_tokens = 0

    for data in events:
        event_type = data.get("type")

        if event_type == "message_start":
            input_tokens = data.get("message", {}).get("usage", {}).get("input_tokens", 0)

        elif event_type == "error":
            error = data.get("error", {})
            raise Exception(f"Stream error: {error.get('message', data)}")

        elif event_type == "content_block_start":
            block = data.get("content_block", {})
            if block.get("type") == "tool_use":
                current_tool = {"id": block["id"], "name": block["name"], "input": ""}

        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "thinking_delta":
                text = delta["thinking"]
                if thinking_print_fn:
                    thinking_print_fn(text)
                thinking_parts.append(text)
            elif delta.get("type") == "text_delta":
                text = delta["text"]
                if print_fn:
                    print_fn(text)
                text_parts.append(text)
            elif delta.get("type") == "input_json_delta":
                if current_tool:
                    current_tool["input"] += delta.get("partial_json", "")

        elif event_type == "content_block_stop":
            if current_tool:
                tool_input = json.loads(current_tool["input"]) if current_tool["input"] else {}
                tool_calls.append(ToolCall(
                    id=current_tool["id"],
                    name=current_tool["name"],
                    args=tool_input
                ))
                current_tool = None

    full_text = "".join(text_parts)
    full_thinking = "".join(thinking_parts)
    if full_thinking:
        raw_content.append({"type": "thinking", "thinking": full_thinking})
    if full_text:
        raw_content.append({"type": "text", "text": full_text})
    for tc in tool_calls:
        raw_content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})

    return Thought(
        text=full_text or None,
        tool_calls=tool_calls,
        raw_content=raw_content,
        thinking=full_thinking or None
    ), input_tokens


# --- SSE Parsing Tests ---

def test_parse_sse_events_extracts_data_lines():
    """Verify parse_sse_events extracts JSON from data: lines."""
    raw_lines = [
        b'event: message_start',
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":42}}}',
        b'',
        b'event: content_block_delta',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
    ]
    events = parse_sse_events(raw_lines)
    assert len(events) == 2
    assert events[0]["type"] == "message_start"
    assert events[1]["type"] == "content_block_delta"


def test_parse_sse_events_skips_non_data_lines():
    """Verify non-data lines are ignored."""
    raw_lines = [
        b'event: ping',
        b': comment',
        b'',
        b'data: {"type":"message_stop"}',
    ]
    events = parse_sse_events(raw_lines)
    assert len(events) == 1


def test_parse_sse_events_handles_string_input():
    """Verify parse_sse_events works with string lines (not just bytes)."""
    raw_lines = [
        'data: {"type":"message_stop"}',
    ]
    events = parse_sse_events(raw_lines)
    assert len(events) == 1


def test_parse_sse_events_skips_malformed_json():
    """Verify malformed JSON data lines are skipped."""
    raw_lines = [
        b'data: not json',
        b'data: {"type":"message_stop"}',
    ]
    events = parse_sse_events(raw_lines)
    assert len(events) == 1
    assert events[0]["type"] == "message_stop"


# --- Thought Building Tests ---

def test_build_thought_text_only():
    """Verify text-only response is assembled correctly."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 100}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    thought, tokens = build_thought_from_events(events)
    assert thought.text == "Hello world"
    assert tokens == 100
    assert not thought.tool_calls
    assert len(thought.raw_content) == 1
    assert thought.raw_content[0]["type"] == "text"


def test_build_thought_with_tool_call():
    """Verify tool call response is assembled correctly."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 200}}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "tool_123", "name": "read_file"
        }},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "input_json_delta", "partial_json": '{"path":'
        }},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "input_json_delta", "partial_json": ' "test.py"}'
        }},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    thought, tokens = build_thought_from_events(events)
    assert thought.text is None
    assert len(thought.tool_calls) == 1
    assert thought.tool_calls[0].name == "read_file"
    assert thought.tool_calls[0].args == {"path": "test.py"}
    assert tokens == 200


def test_build_thought_mixed_text_and_tool():
    """Verify response with both text and tool calls."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 50}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me read that."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {
            "type": "tool_use", "id": "tool_456", "name": "read_file"
        }},
        {"type": "content_block_delta", "index": 1, "delta": {
            "type": "input_json_delta", "partial_json": '{"path": "main.py"}'
        }},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]
    thought, tokens = build_thought_from_events(events)
    assert thought.text == "Let me read that."
    assert len(thought.tool_calls) == 1
    assert thought.tool_calls[0].name == "read_file"
    assert len(thought.raw_content) == 2


def test_build_thought_tracks_streaming_output():
    """Verify print_fn is called for each text delta."""
    chunks = []
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "one"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " two"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " three"}},
        {"type": "message_stop"},
    ]
    thought, _ = build_thought_from_events(events, print_fn=chunks.append)
    assert chunks == ["one", " two", " three"]
    assert thought.text == "one two three"


def test_build_thought_empty_response():
    """Verify empty event stream produces empty Thought."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {"type": "message_stop"},
    ]
    thought, tokens = build_thought_from_events(events)
    assert thought.text is None
    assert thought.tool_calls == []
    assert tokens == 5


def test_build_thought_tool_with_empty_input():
    """Verify tool call with no input args."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 30}}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "tool_789", "name": "list_files"
        }},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    thought, _ = build_thought_from_events(events)
    assert len(thought.tool_calls) == 1
    assert thought.tool_calls[0].name == "list_files"
    assert thought.tool_calls[0].args == {}


def test_build_thought_with_thinking():
    """Verify thinking deltas are accumulated and stored."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 50}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": " reason"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Here's the answer."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]
    thought, tokens = build_thought_from_events(events)
    assert thought.thinking == "Let me reason"
    assert thought.text == "Here's the answer."
    assert tokens == 50
    assert len(thought.raw_content) == 2  # thinking + text
    assert thought.raw_content[0]["type"] == "thinking"


def test_build_thought_thinking_print_fn():
    """Verify thinking_print_fn is called for each thinking delta."""
    chunks = []
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "step one"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": " step two"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    thought, _ = build_thought_from_events(events, thinking_print_fn=chunks.append)
    assert chunks == ["step one", " step two"]
    assert thought.thinking == "step one step two"


def test_build_thought_error_event_raises():
    """Verify SSE error events raise an exception."""
    import pytest
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    ]
    with pytest.raises(Exception, match="Stream error: Overloaded"):
        build_thought_from_events(events)
