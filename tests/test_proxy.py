import json

from tooltrim import ToolCompressor
from tooltrim.proxy import (
    CompressionStats,
    compress_anthropic_messages,
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


# --- Anthropic Messages format -------------------------------------------------

def test_anthropic_compresses_string_tool_result():
    tc = ToolCompressor(max_tokens=80, add_footer=False)
    messages = [
        {"role": "user", "content": "did customer 4417 get a refund?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu1", "name": "lookup", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1",
             "content": _big_tool_output()}]},
    ]
    out, stats = compress_anthropic_messages(messages, tc)
    assert stats.messages_compressed == 1
    block = out[-1]["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "tu1"
    assert "4417" in block["content"]
    assert len(block["content"]) < len(_big_tool_output())
    # the real user query message is untouched
    assert out[0] == messages[0]


def test_anthropic_compresses_block_list_tool_result():
    tc = ToolCompressor(max_tokens=80, add_footer=False)
    messages = [
        {"role": "user", "content": "refund customer 4417"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": [
                {"type": "text", "text": _big_tool_output()}]}]},
    ]
    out, stats = compress_anthropic_messages(messages, tc)
    assert stats.messages_compressed == 1
    inner = out[-1]["content"][0]["content"]
    assert inner[0]["type"] == "text"
    assert "4417" in inner[0]["text"]


def test_anthropic_small_tool_result_passes_through():
    tc = ToolCompressor(max_tokens=512, add_footer=False)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "short"}]},
    ]
    out, stats = compress_anthropic_messages(messages, tc)
    assert stats.messages_compressed == 0
    assert out == messages


def test_anthropic_non_tool_result_blocks_untouched():
    tc = ToolCompressor(max_tokens=10, add_footer=False)
    big = "word " * 5000
    messages = [
        {"role": "user", "content": [{"type": "text", "text": big}]},
    ]
    out, stats = compress_anthropic_messages(messages, tc)
    assert stats.messages_compressed == 0
    assert out[0]["content"][0]["text"] == big


def test_transform_request_body_anthropic():
    tc = ToolCompressor(max_tokens=80, add_footer=False)
    body = json.dumps({
        "model": "claude-haiku-4-5",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "refund customer 4417"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1",
                 "content": _big_tool_output()}]},
        ],
    }).encode()
    new_body, stats = transform_request_body(body, tc, api="anthropic")
    payload = json.loads(new_body)
    assert stats.messages_compressed == 1
    assert payload["model"] == "claude-haiku-4-5"
    assert "4417" in payload["messages"][-1]["content"][0]["content"]
