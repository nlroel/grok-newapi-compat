#!/usr/bin/env python3
"""Strict-schema self test for the Grok Responses compatibility bridge.

This deliberately checks the actual fields Grok's strict client has rejected in
the field (name, arguments, annotations, end_index) as well as the required
fields in the official xAI ModelResponse/output schemas.
"""

import copy
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = Path(os.getenv("XAI_OPENAPI", "/tmp/xai-openapi.json"))

spec = importlib.util.spec_from_file_location("grok_compat", HERE / "grok_compat.py")
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)

schema_doc = json.loads(SCHEMA_PATH.read_text()) if SCHEMA_PATH.exists() else None
schemas = schema_doc.get("components", {}).get("schemas", {}) if schema_doc else {}


def required(name):
    return list(schemas.get(name, {}).get("required", []))


def assert_required(testcase, obj, fields, where):
    if not isinstance(obj, dict):
        testcase.fail(f"{where}: expected object, got {type(obj).__name__}")
    missing = [key for key in fields if key not in obj]
    testcase.assertFalse(missing, f"{where}: missing fields {missing}")


def assert_output_item(testcase, item, where):
    mapping = {
        "message": "OutputMessage",
        "reasoning": "Reasoning",
        "function_call": "FunctionToolCall",
        "web_search_call": "WebSearchCall",
        "shell_call": "ShellCall",
        "custom_tool_call": "CustomToolCall",
        "mcp_call": "McpCall",
        "file_search_call": "FileSearchCall",
        "code_interpreter_call": "CodeInterpreterCall",
    }
    item_type = item.get("type")
    testcase.assertIn(item_type, mapping, f"{where}: unsupported output item type {item_type}")
    assert_required(testcase, item, required(mapping[item_type]), f"{where}.{item_type}")

    if item_type == "message":
        for index, part in enumerate(item.get("content") or []):
            if part.get("type") == "output_text":
                testcase.assertIn("annotations", part, f"{where}.message.content[{index}]")
                for annotation in part["annotations"]:
                    assert_required(
                        testcase,
                        annotation,
                        ["type", "url", "start_index", "end_index"],
                        f"{where}.message.content[{index}].annotation",
                    )
    if item_type == "function_call":
        testcase.assertIsInstance(item.get("arguments"), str, f"{where}.function_call.arguments")
    if item_type in ("web_search_call",):
        testcase.assertIsInstance(item.get("action"), dict, f"{where}.web_search_call.action")


def assert_response(testcase, response, where):
    assert_required(testcase, response, required("ModelResponse"), where)
    for index, item in enumerate(response.get("output") or []):
        assert_output_item(testcase, item, f"{where}.output[{index}]")


def parse_sse(path):
    p = Path(path)
    if not p.exists():
        return []
    events = []
    for line in p.read_text(errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw and raw != "[DONE]":
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    return events


class SerializationTests(unittest.TestCase):
    def test_canonical_model_response_has_all_required_fields(self):
        req = {
            "model": "glm-5.3-flash",
            "tools": [{"type": "function", "name": "test", "parameters": {"type": "object"}}],
        }
        response = gc._response_object(
            req,
            response_id="resp_test",
            status="completed",
            output=[],
            model="glm-5.3-flash",
            created_at=1,
            usage={"input_tokens": 1, "output_tokens": 1},
            completed_at=2,
        )
        assert_response(self, response, "canonical response")

    def test_native_response_repairs_items_and_annotations(self):
        req = {"model": "glm-5.3-flash", "tools": [{"type": "web_search"}]}
        response = {
            "id": "resp_test",
            "created_at": 1,
            "status": "completed",
            "model": "glm-5.3-flash",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "web_search_call", "action": {"type": "search", "query": "q"}},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "hello",
                        "annotations": [{"type": "url_citation", "url": "https://example.com"}],
                    }],
                },
                {"type": "function_call", "call_id": "call_1", "name": "test"},
            ],
        }
        normalized = gc._normalize_native_response(req, response)
        assert_response(self, normalized, "normalized native response")
        annotation = normalized["output"][2]["content"][0]["annotations"][0]
        self.assertEqual(annotation["start_index"], 0)
        self.assertEqual(annotation["end_index"], 0)
        self.assertEqual(normalized["output"][3]["arguments"], "")

    def test_native_sse_repairs_every_known_strict_shape(self):
        cases = [
            {"type": "response.output_item.added", "item": {
                "type": "function_call", "call_id": "call_1", "name": "test", "status": "in_progress"
            }},
            {"type": "response.output_item.added", "item": {
                "type": "web_search_call", "id": "ws_1", "status": "in_progress"
            }},
            {"type": "response.output_item.added", "item": {
                "type": "message", "role": "assistant", "content": [{
                    "type": "output_text", "text": ""
                }], "status": "in_progress"
            }},
            {"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}},
            {"type": "response.output_text.annotation.added", "annotation": {
                "type": "url_citation", "url": "https://example.com"
            }},
            {"type": "response.output_item.added", "item": {"type": "file_search_call", "status": "in_progress"}},
            {"type": "response.output_item.added", "item": {"type": "code_interpreter_call", "status": "in_progress"}},
        ]
        for index, event in enumerate(cases):
            normalized = gc._normalize_native_sse_payload(copy.deepcopy(event))
            if normalized.get("type") == "response.output_item.added":
                assert_output_item(self, normalized["item"], f"event[{index}]")
            elif normalized.get("type") == "response.content_part.added":
                self.assertIn("annotations", normalized["part"])
            elif normalized.get("type") == "response.output_text.annotation.added":
                self.assertIn("start_index", normalized["annotation"])
                self.assertIn("end_index", normalized["annotation"])

    def test_native_request_repairs_bad_history(self):
        req = {
            "model": "glm-5.3-flash",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "function_call", "call_id": "old", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "old", "output": "ok"},
            ],
            "tools": [{"type": "web_search", "name": None}, {"type": "x_search", "name": None}],
        }
        normalized = gc._normalize_native_request(req)
        self.assertEqual(normalized["input"][1]["name"], "unknown_tool")
        self.assertNotIn("name", normalized["tools"][0])
        self.assertNotIn("name", normalized["tools"][1])

    def test_chat_stream_bridge_output_indices_and_required_items(self):
        req = {
            "model": "glm-5.3-flash",
            "tools": [
                {"type": "function", "name": "shell", "parameters": {"type": "object"}},
                {"type": "shell", "environment": {"type": "local"}},
            ],
        }
        events = []
        bridge = gc.StreamBridge(req, [], lambda raw: events.append(json.loads(raw)))
        bridge.start()
        bridge.process_chunk({"model": "m", "choices": [{"delta": {"reasoning_content": "think"}}]})
        bridge.process_chunk({"model": "m", "choices": [{"delta": {"content": "hi"}}]})
        bridge.process_chunk({"model": "m", "choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "shell", "arguments": "{\"commands\":[\"pwd\"]}"}
        }]}}]})
        bridge.process_chunk({"model": "m", "choices": [{"finish_reason": "stop"}], "usage": {
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
        }})
        bridge.finish()

        self.assertEqual([x.get("sequence_number") for x in events], list(range(1, len(events) + 1)))
        # The reasoning item and the message item must retain their own output_index.
        added = [x for x in events if x.get("type") == "response.output_item.added"]
        reasoning = next(x for x in added if x["item"]["type"] == "reasoning")
        message = next(x for x in added if x["item"]["type"] == "message")
        reasoning_done = next(x for x in events if x.get("type") == "response.output_item.done" and x["item"]["type"] == "reasoning")
        message_done = next(x for x in events if x.get("type") == "response.output_item.done" and x["item"]["type"] == "message")
        self.assertEqual(reasoning["output_index"], reasoning_done["output_index"])
        self.assertEqual(message["output_index"], message_done["output_index"])
        completed = next(x for x in events if x.get("type") == "response.completed")
        assert_response(self, completed["response"], "chat bridge completed response")

    def test_chat_sse_parser_handles_newline_less_stream(self):
        obj1 = {"choices": [{"delta": {"content": "计"}, "index": 0}]}
        obj2 = {"choices": [{"delta": {"content": "算"}, "index": 0}]}
        # No newlines between data frames, and UTF-8 is split at a chunk edge.
        raw = (b'data:' + json.dumps(obj1, ensure_ascii=False).encode('utf-8')
               + b'data:' + json.dumps(obj2, ensure_ascii=False).encode('utf-8')
               + b'data: [DONE]')
        chunks = list(gc._iter_chat_sse_chunks([raw[:38], raw[38:80], raw[80:]]))
        self.assertEqual(chunks, [obj1, obj2])

    def test_structured_output_repairs_legacy_and_empty_fields(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "evidence", "next_step", "blocker_key"],
            "properties": {
                "decision": {"type": "string", "enum": ["continue", "candidate_complete", "blocked"]},
                "evidence": {"type": "string", "minLength": 1},
                "next_step": {"type": "string", "minLength": 1},
                "blocker_key": {"type": "string"},
            },
        }
        req = {"text": {"format": {"type": "json_schema", "json_schema": {
            "name": "structured_output", "strict": True, "schema": schema
        }}}}
        legacy = '```json\n{"continue":false,"candidate_complete":true,"blocked":false,"blocker_key":"","evidence":"done"}\n```'
        repaired = json.loads(gc._normalize_structured_output_text(req, legacy))
        self.assertEqual(repaired["decision"], "candidate_complete")
        self.assertTrue(repaired["next_step"])
        self.assertNotIn("continue", repaired)

        empty_next = json.dumps({"decision": "candidate_complete", "evidence": "done", "next_step": "", "blocker_key": ""})
        repaired = json.loads(gc._normalize_structured_output_text(req, empty_next))
        self.assertTrue(repaired["next_step"])

    def test_structured_output_repairs_blocker_key_semantics(self):
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["decision", "evidence", "next_step", "blocker_key"],
            "properties": {
                "decision": {"type": "string", "enum": ["continue", "candidate_complete", "blocked"]},
                "evidence": {"type": "string", "minLength": 1},
                "next_step": {"type": "string", "minLength": 1},
                "blocker_key": {"type": "string"},
            },
        }
        req = {"text": {"format": {"type": "json_schema", "json_schema": {
            "name": "structured_output", "strict": True, "schema": schema
        }}}}

        blocked = gc._normalize_structured_output_text(req, json.dumps({
            "decision": "blocked", "evidence": "  missing access  ",
            "next_step": "", "blocker_key": "Missing GitHub Access!",
        }))
        repaired = json.loads(blocked)
        self.assertEqual(repaired["decision"], "blocked")
        self.assertEqual(repaired["blocker_key"], "missing_github_access")
        self.assertTrue(repaired["next_step"])

        complete = gc._normalize_structured_output_text(req, json.dumps({
            "decision": "candidate_complete", "evidence": "tests passed",
            "next_step": "no more work", "blocker_key": "should_be_empty",
        }))
        repaired = json.loads(complete)
        self.assertEqual(repaired["blocker_key"], "")

    def test_stream_usage_is_always_present(self):
        events = []
        bridge = gc.StreamBridge({"model": "m"}, [], lambda raw: events.append(json.loads(raw)))
        bridge.start()
        bridge.process_chunk({"choices": [{"delta": {"content": "hi"}}]})
        bridge.finish()
        completed = next(x for x in events if x.get("type") == "response.completed")
        self.assertIsInstance(completed["response"]["usage"], dict)
        self.assertIn("total_tokens", completed["response"]["usage"])

    def test_structured_output_stream_suppresses_raw_deltas(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "evidence", "next_step", "blocker_key"],
            "properties": {
                "decision": {"type": "string", "enum": ["continue", "candidate_complete", "blocked"]},
                "evidence": {"type": "string", "minLength": 1},
                "next_step": {"type": "string", "minLength": 1},
                "blocker_key": {"type": "string"},
            },
        }
        req = {"model": "m", "text": {"format": {"type": "json_schema", "json_schema": {
            "name": "structured_output", "strict": True, "schema": schema
        }}}}
        events = []
        bridge = gc.StreamBridge(req, [], lambda raw: events.append(json.loads(raw)))
        bridge.start()
        bridge.process_chunk({"choices": [{"delta": {"content": '{"decision":"candidate_complete","evidence":"done"'}}]})
        bridge.process_chunk({"choices": [{"delta": {"content": ',"next_step":"","blocker_key":""}'}}]})
        bridge.finish()
        self.assertNotIn("response.output_text.delta", [x.get("type") for x in events])
        completed = next(x for x in events if x.get("type") == "response.completed")
        text = completed["response"]["output"][0]["content"][0]["text"]
        self.assertTrue(json.loads(text)["next_step"])

    def test_chat_input_folds_reasoning_into_next_assistant(self):
        req = {"model": "m", "input": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "first"}]},
            {"type": "message", "role": "assistant", "content": "answer"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "trailing"}]},
            {"type": "message", "role": "user", "content": "next"},
        ]}
        chat = gc.responses_to_chat_request(req)
        self.assertEqual(chat["messages"][0]["role"], "assistant")
        self.assertEqual(chat["messages"][0]["content"], "answer")
        self.assertEqual(chat["messages"][0]["reasoning_content"], "first")
        self.assertEqual(chat["messages"][1]["role"], "user")
        self.assertEqual(len(chat["messages"]), 2)

    def test_output_history_keeps_reasoning_and_combines_tool_calls(self):
        response = {"output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
            {"type": "function_call", "call_id": "c1", "name": "one", "arguments": "{}"},
            {"type": "shell_call", "call_id": "c2", "action": {"type": "exec", "commands": ["pwd"]}},
        ]}
        messages = gc.chat_output_messages_from_response(response)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["reasoning_content"], "think")
        self.assertEqual([c["id"] for c in messages[0]["tool_calls"]], ["c1", "c2"])

    def test_chat_stream_upstream_error_emits_failed(self):
        events = []
        bridge = gc.StreamBridge({"model": "m"}, [], lambda raw: events.append(json.loads(raw)))
        bridge.start()
        bridge.process_chunk({"choices": [{"delta": {"content": "partial"}}]})
        bridge.process_chunk({"error": {"code": 42, "message": "upstream exploded"}})
        bridge.finish()
        failed = [x for x in events if x.get("type") == "response.failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["response"]["status"], "failed")
        self.assertEqual(failed[0]["response"]["error"]["code"], "42")
        self.assertEqual(failed[0]["response"]["error"]["message"], "upstream exploded")
        self.assertIsInstance(failed[0]["response"]["usage"], dict)
        self.assertNotIn("response.completed", [x.get("type") for x in events])

    def test_structured_output_recursive_schema_repair(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["outer", "count"],
            "properties": {
                "count": {"type": "integer"},
                "outer": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "enabled"],
                    "properties": {
                        "text": {"type": "string", "minLength": 1},
                        "enabled": {"type": "boolean"},
                    },
                },
            },
        }
        req = {"text": {"format": {"type": "json_schema", "json_schema": {
            "name": "generic", "strict": True, "schema": schema
        }}}}
        repaired = json.loads(gc._normalize_structured_output_text(req, json.dumps({
            "count": "3", "outer": {"enabled": "yes"}, "ignored": True,
        })))
        self.assertEqual(repaired["count"], 3)
        self.assertEqual(repaired["outer"]["text"], "N/A")
        self.assertTrue(repaired["outer"]["enabled"])
        self.assertNotIn("ignored", repaired)

    def test_real_native_capture_after_normalization(self):
        source = os.getenv("GROK_NATIVE_CAPTURE", "/tmp/grok_native_upstream.sse")
        events = parse_sse(source)
        if not events:
            self.skipTest(f"no native capture at {source}")
        for index, event in enumerate(events):
            normalized = gc._normalize_native_sse_payload(copy.deepcopy(event))
            etype = normalized.get("type")
            if etype in ("response.created", "response.in_progress", "response.completed"):
                response = gc._normalize_native_response({}, normalized.get("response") or {})
                assert_response(self, response, f"capture[{index}].{etype}")
            elif etype in ("response.output_item.added", "response.output_item.done"):
                assert_output_item(self, normalized["item"], f"capture[{index}].{etype}")
            elif etype in ("response.content_part.added", "response.content_part.done"):
                if normalized.get("part", {}).get("type") == "output_text":
                    self.assertIn("annotations", normalized["part"], f"capture[{index}].{etype}")
            elif etype == "response.output_text.annotation.added":
                assert_required(
                    self, normalized["annotation"],
                    ["type", "url", "start_index", "end_index"],
                    f"capture[{index}].{etype}",
                )


if __name__ == "__main__":
    result = unittest.main(argv=sys.argv[:1], exit=False, verbosity=2).result
    if not result.wasSuccessful():
        sys.exit(1)
