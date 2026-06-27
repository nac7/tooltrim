import json

from tooltrim import ToolCompressor
from tooltrim.proxy import (
    CompressionStats,
    compress_messages,
    transform_request_body,
)


def _big_tool_output():
    return json.dumps([{"id": i, "note": f"row {i}"} for i in range(400)]
                      + [{"id": 999, "note": "the refund was issued to customer 4417"}])


def test_compress_messages_compresses_tool_role():
    tc = ToolCompressor(max_tokens=80, add_footer=False)
    messages = [
        {"role": "user", "content": "did customer 4417 get a refund?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": _big_tool_output()},
    ]
    out, stats = compress_messages(messages, tc)
    assert isinstance(stats, CompressionStats)
    assert stats.messages_compressed == 1
    assert stats.saved_tokens > 0
    # the tool message shrank; others untouched
    assert len(out[-1]["content"]) < len(messages[-1]["content"])
    assert out[0] == messages[0]


def test_compress_messages_keeps_relevant_fact():
    tc = ToolCompressor(max_tokens=80, add_footer=False)
    messages = [
        {"role": "user", "content": "refund customer 4417"},
        {"role": "tool", "tool_call_id": "t1", "content": _big_tool_output()},
    ]
    out, _ = compress_messages(messages, tc)
    assert "4417" in out[-1]["content"]


def test_small_tool_output_passes_through():
    tc = ToolCompressor(max_tokens=512, add_footer=False)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "t1", "content": "short result"},
    ]
    out, stats = compress_messages(messages, tc)
    assert stats.messages_compressed == 0
    assert out[-1]["content"] == "short result"


def test_user_and_assistant_never_compressed():
    tc = ToolCompressor(max_tokens=10, add_footer=False)
    big = "word " * 5000
    messages = [
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
    ]
    out, stats = compress_messages(messages, tc)
    assert stats.messages_compressed == 0
    assert out[0]["content"] == big and out[1]["content"] == big


def test_transform_request_body_roundtrip():
    tc = ToolCompressor(max_tokens=80, add_footer=False)
    body = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "refund customer 4417"},
            {"role": "tool", "tool_call_id": "t1", "content": _big_tool_output()},
        ],
    }).encode()
    new_body, stats = transform_request_body(body, tc)
    payload = json.loads(new_body)
    assert stats.messages_compressed == 1
    assert payload["model"] == "gpt-4o-mini"
    assert "4417" in payload["messages"][-1]["content"]


def test_transform_non_json_passes_through():
    tc = ToolCompressor()
    raw = b"not json at all"
    out, stats = transform_request_body(raw, tc)
    assert out == raw
    assert stats.messages_compressed == 0
