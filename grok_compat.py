#!/usr/bin/env python3
"""Single-auth OpenAI Responses compatibility bridge for NewAPI.

The public API stays OpenAI Responses; the upstream API is Chat Completions.
This service deliberately does not authenticate requests itself.  The caller's
Authorization header is passed unchanged to NewAPI, preserving NewAPI quota,
token groups, rate limits and audit semantics.

The wire mapping below is based on the behavior of LiteLLM's open-source
Responses -> Chat Completions bridge (litellm/responses/
litellm_completion_transformation), reduced to a standard-library service.
"""

import codecs
import copy
import http.client
import http.server
import json
import logging
import os
import re
import socket
import threading
import time
import urllib.parse
import uuid
from collections import OrderedDict

LISTEN_HOST = os.getenv("GROK_COMPAT_HOST", "127.0.0.1")
LISTEN_PORT = int(os.getenv("GROK_COMPAT_PORT", "3001"))
UPSTREAM_HOST = os.getenv("NEWAPI_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.getenv("NEWAPI_PORT", "3000"))
UPSTREAM_TIMEOUT = int(os.getenv("GROK_COMPAT_UPSTREAM_TIMEOUT", "3600"))
HISTORY_SIZE = int(os.getenv("GROK_COMPAT_HISTORY_SIZE", "512"))
DEBUG_CAPTURE = os.getenv("GROK_COMPAT_DEBUG_CAPTURE", "0").lower() in ("1", "true", "yes")
DEBUG_DIR = os.getenv("GROK_COMPAT_DEBUG_DIR", "/tmp/grok-compat")

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host",
    "accept-encoding", "content-length",
}

_history_lock = threading.Lock()
_response_history = OrderedDict()


def _iter_chat_sse_chunks(raw_chunks):
    """Decode Chat Completions SSE even when NewAPI omits newline separators."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    json_decoder = json.JSONDecoder()
    buffer = ""
    for raw in raw_chunks:
        buffer += decoder.decode(raw)
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                break
            if not buffer.startswith("data:"):
                logging.warning("ignoring non-SSE upstream prefix")
                buffer = ""
                break
            payload = buffer[5:].lstrip()
            if payload.startswith("[DONE]"):
                return
            try:
                chunk, consumed = json_decoder.raw_decode(payload)
            except json.JSONDecodeError:
                # Wait for the remainder of a JSON object split across reads.
                break
            buffer = payload[consumed:]
            yield chunk


def remember_response(response_id, messages):
    if not response_id or not isinstance(messages, list):
        return
    with _history_lock:
        _response_history.pop(response_id, None)
        _response_history[response_id] = copy.deepcopy(messages)
        while len(_response_history) > HISTORY_SIZE:
            _response_history.popitem(last=False)


def previous_messages(response_id):
    if not response_id:
        return []
    with _history_lock:
        return copy.deepcopy(_response_history.get(response_id, []))


def _rid(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def _text_from_content(value):
    """Extract plain text from OpenAI chat content."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(value)


def _content_to_chat(value):
    """Responses content parts -> Chat Completions content."""
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, list):
        return value

    output = []
    for part in value:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_text", "output_text", "text"):
            output.append({"type": "text", "text": part.get("text", "")})
        elif ptype in ("input_image", "image_url"):
            image_url = part.get("image_url", part.get("url"))
            if isinstance(image_url, dict):
                output.append({"type": "image_url", "image_url": image_url})
            elif isinstance(image_url, str):
                output.append({"type": "image_url", "image_url": {"url": image_url}})
        elif ptype == "refusal":
            output.append({"type": "text", "text": part.get("refusal", "")})
        else:
            output.append(part)
    return output


def _reasoning_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            p.get("text", "") for p in value
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def _tool_output_to_content(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text = "".join(
            p.get("text", "") for p in value
            if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text")
            and isinstance(p.get("text"), str)
        )
        if text:
            return text
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _input_item_to_messages(item):
    """Convert one Responses input item to one or more Chat messages."""
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if not isinstance(item, dict):
        return []

    item_type = item.get("type")
    role = item.get("role", "user")

    # A normal message. Grok may omit type for easy input items.
    if item_type in (None, "message", "easy_input_message"):
        return [{
            "role": role,
            "content": _content_to_chat(item.get("content", "")),
        }]

    # xAI/Grok local shell calls are replayed as a Chat function call named
    # "shell"; the shell_call output becomes the matching tool result.
    if item_type == "shell_call":
        action = item.get("action") or {}
        arguments = {
            "commands": action.get("commands", []),
            "timeout_ms": action.get("timeout_ms"),
            "max_output_length": action.get("max_output_length"),
        }
        return [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": item.get("call_id") or item.get("id") or _rid("call"),
                "type": "function",
                "function": {
                    "name": "shell",
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }],
        }]

    if item_type == "shell_call_output":
        call_id = item.get("call_id") or item.get("id")
        if not call_id:
            return []
        return [{
            "role": "tool",
            "tool_call_id": call_id,
            "content": _tool_output_to_content(item.get("output")),
        }]

    # Prior assistant function call.
    if item_type in ("function_call", "custom_tool_call"):
        call_id = item.get("call_id") or item.get("id") or _rid("call")
        name = item.get("name", "")
        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        return [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }],
        }]

    # A tool result.
    if item_type in ("function_call_output", "custom_tool_call_output", "tool_result"):
        call_id = item.get("call_id") or item.get("id")
        if not call_id:
            return []
        return [{
            "role": "tool",
            "tool_call_id": call_id,
            "content": _tool_output_to_content(item.get("output")),
        }]

    # Keep prior reasoning out of the visible prompt, but preserve it for
    # providers/NewAPI channels that understand reasoning_content.
    if item_type == "reasoning":
        summary = item.get("summary") or item.get("content") or []
        text = _reasoning_text(summary)
        if not text:
            return []
        return [{"role": "assistant", "content": None, "reasoning_content": text}]

    # Unknown role-bearing items are passed through instead of dropping context.
    if role:
        return [{"role": role, "content": _content_to_chat(item.get("content", ""))}]
    return []


def _chat_function_tool(name, description="", parameters=None, strict=None):
    function = {"name": name, "parameters": parameters or {"type": "object", "properties": {}}}
    if description:
        function["description"] = description
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


def _tools_to_chat(tools):
    """Convert Responses tools to Chat tools.

    xAI/Grok's local execution tool is `type: "shell"`.  Since Chat Completions
    has no native local-shell tool, it is exposed to the model as a `shell`
    function and converted back to Responses `shell_call` on output.
    """
    if not isinstance(tools, list):
        return None
    chat_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "shell":
            skills = ((tool.get("environment") or {}).get("skills") or [])
            descriptions = []
            for skill in skills:
                if isinstance(skill, dict) and skill.get("name"):
                    descriptions.append(f"{skill['name']}: {skill.get('description', '')}")
            description = "Execute one or more shell commands in the user's local environment."
            if descriptions:
                description += " Available skills:\n" + "\n".join(descriptions)
            chat_tools.append(_chat_function_tool(
                "shell",
                description,
                {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Shell commands to execute in order.",
                        },
                        "timeout_ms": {"type": "integer", "description": "Optional timeout in milliseconds."},
                        "max_output_length": {"type": "integer", "description": "Optional output character limit."},
                    },
                    "required": ["commands"],
                },
            ))
            continue
        if tool_type == "function":
            function = {
                k: tool[k] for k in ("name", "description", "parameters", "strict")
                if k in tool and tool[k] is not None
            }
            if "parameters" not in function:
                function["parameters"] = tool.get("input_schema") or {"type": "object", "properties": {}}
            chat_tools.append({"type": "function", "function": function})
            continue
        # Tolerate clients that declare callable local tools with another type
        # but still provide a callable name/schema.
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        if name:
            parameters = tool.get("parameters") or tool.get("input_schema") or (
                tool.get("function") or {}
            ).get("parameters") or {"type": "object", "properties": {}}
            chat_tools.append(_chat_function_tool(
                name,
                tool.get("description") or (tool.get("function") or {}).get("description", ""),
                parameters,
                tool.get("strict"),
            ))
    return chat_tools or None


def _response_json_schema(req):
    """Return the schema when the request asks for strict JSON output."""
    text = req.get("text")
    if not isinstance(text, dict):
        return None
    fmt = text.get("format")
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return None
    container = fmt.get("json_schema")
    if not isinstance(container, dict):
        container = fmt
    schema = container.get("schema")
    return schema if isinstance(schema, dict) else None


def _decode_model_json(text):
    """Decode a JSON object emitted directly or inside a markdown fence."""
    if not isinstance(text, str):
        return None
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2:
            value = "\n".join(lines[1:]).strip()
        if value.endswith("```"):
            value = value[:-3].strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, json.JSONDecodeError):
        pass
    start = value.find("{")
    if start < 0:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(value[start:])
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, json.JSONDecodeError):
        return None


def _schema_string_fallback(name, schema, decision):
    """Best-effort value for a string that the provider left empty."""
    minimum = schema.get("minLength", 0)
    if not isinstance(minimum, int) or minimum <= 0:
        return ""
    if name == "next_step":
        return {
            "candidate_complete": "No further implementation step is required.",
            "blocked": "Resolve the reported blocker, then retry.",
        }.get(decision, "Continue with the next actionable step.")
    if name == "evidence":
        return "The model did not provide structured evidence."
    return "N/A"


def _normalize_decision_value(value):
    """Map common decision spellings to the goal evaluator enum."""
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "continue": "continue",
        "candidate_complete": "candidate_complete",
        "candidatecomplete": "candidate_complete",
        "complete": "candidate_complete",
        "blocked": "blocked",
        "block": "blocked",
    }
    return aliases.get(key)


def _normalize_blocker_key(value):
    """Coerce a blocker identity to xAI's lowercase snake_case rule."""
    if not isinstance(value, str):
        value = ""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:128]


def _conform_model_json_to_schema(obj, schema):
    """Repair common provider deviations from a Responses JSON schema."""
    if not isinstance(obj, dict) or not isinstance(schema, dict):
        return obj
    repaired = copy.deepcopy(obj)

    # Older/looser evaluators sometimes return three booleans instead of the
    # required single `decision` enum.
    if "decision" not in repaired:
        if repaired.get("candidate_complete") is True:
            repaired["decision"] = "candidate_complete"
        elif repaired.get("blocked") is True:
            repaired["decision"] = "blocked"
        elif "continue" in repaired:
            repaired["decision"] = "continue" if repaired.get("continue") is True else "candidate_complete"
    if "decision" in repaired:
        normalized = _normalize_decision_value(repaired["decision"])
        if normalized:
            repaired["decision"] = normalized

    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for name in required:
        prop = properties.get(name) if isinstance(properties.get(name), dict) else {}
        value = repaired.get(name)
        invalid_empty_string = isinstance(value, str) and not value.strip()
        if name not in repaired or value is None or invalid_empty_string:
            if name == "decision" and isinstance(prop.get("enum"), list) and prop["enum"]:
                repaired[name] = prop["enum"][0]
            elif prop.get("type") == "string":
                repaired[name] = _schema_string_fallback(
                    name, prop, repaired.get("decision") if isinstance(repaired.get("decision"), str) else ""
                )
            elif prop.get("type") == "array":
                repaired[name] = []
            elif prop.get("type") == "object":
                repaired[name] = {}
            elif prop.get("type") == "integer":
                repaired[name] = 0
            elif prop.get("type") == "number":
                repaired[name] = 0
            elif prop.get("type") == "boolean":
                repaired[name] = False

        # xAI's goal parser is stricter than the JSON schema: it validates
        # semantic relationships between decision and blocker_key as well.
        if name == "blocker_key" and repaired.get("decision") == "blocked":
            repaired[name] = _normalize_blocker_key(repaired.get(name)) or "unknown_blocker"
        elif name == "blocker_key":
            repaired[name] = ""

        enum = prop.get("enum")
        if isinstance(enum, list) and repaired.get(name) not in enum:
            # A legacy mapping may already have set a valid decision above.
            decision = repaired.get("decision")
            if name != "decision" and isinstance(decision, str) and decision in enum:
                repaired[name] = decision
            elif enum:
                repaired[name] = enum[0]

    if schema.get("additionalProperties") is False and isinstance(properties, dict):
        repaired = {key: value for key, value in repaired.items() if key in properties}
    return repaired
def _normalize_structured_output_text(req, text):
    """Make provider JSON conform closely enough for strict Responses clients."""
    schema = _response_json_schema(req)
    if not schema:
        return text
    parsed = _decode_model_json(text)
    if parsed is None:
        return text
    repaired = _conform_model_json_to_schema(parsed, schema)
    return json.dumps(repaired, ensure_ascii=False, separators=(",", ":"))


def _has_shell_tool(req):
    return any(isinstance(t, dict) and t.get("type") == "shell" for t in (req.get("tools") or []))


def _tool_choice_to_chat(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") in ("function", "custom") and tool_choice.get("name"):
            return {"type": "function", "function": {"name": tool_choice["name"]}}
        if isinstance(tool_choice.get("function"), dict):
            return {"type": "function", "function": tool_choice["function"]}
    return "auto"


def _response_text_field(req):
    """Grok requires response.text.format; OpenAI tolerates an empty object."""
    text = req.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        return text
    return {"format": {"type": "text"}}


def _text_to_response_format(text):
    if not isinstance(text, dict):
        return None
    fmt = text.get("format")
    if not isinstance(fmt, dict):
        return None
    fmt_type = fmt.get("type")
    if fmt_type == "json_object":
        return {"type": "json_object"}
    if fmt_type == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.get("name", "response_schema"),
                "schema": fmt.get("schema", {}),
                "strict": fmt.get("strict", False),
            },
        }
    return None


def responses_to_chat_request(req):
    """Map a Responses create request to Chat Completions, LiteLLM style."""
    messages = []
    instructions = req.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})

    input_value = req.get("input", "")
    if isinstance(input_value, list):
        for item in input_value:
            converted = _input_item_to_messages(item)
            # Consecutive Responses function_call items must become one
            # assistant message with multiple tool_calls.
            for message in converted:
                if (
                    message.get("role") == "assistant"
                    and message.get("tool_calls")
                    and messages
                    and messages[-1].get("role") == "assistant"
                    and messages[-1].get("tool_calls")
                ):
                    messages[-1]["tool_calls"].extend(message["tool_calls"])
                else:
                    messages.append(message)
    elif input_value != "":
        messages.append({"role": "user", "content": input_value})

    tools = _tools_to_chat(req.get("tools"))

    chat = {
        "model": req.get("model"),
        "messages": messages,
        "stream": bool(req.get("stream", False)),
    }
    optional = {
        "temperature": "temperature",
        "top_p": "top_p",
        "max_output_tokens": "max_tokens",
        "user": "user",
        "parallel_tool_calls": "parallel_tool_calls",
    }
    for src, dst in optional.items():
        if req.get(src) is not None:
            chat[dst] = req[src]

    if tools:
        chat["tools"] = tools
        chat["tool_choice"] = _tool_choice_to_chat(req.get("tool_choice"))

    response_format = _text_to_response_format(req.get("text"))
    if response_format:
        chat["response_format"] = response_format

    reasoning = req.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        chat["reasoning_effort"] = reasoning["effort"]

    if chat["stream"]:
        chat["stream_options"] = {"include_usage": True}

    # Remove None values; many NewAPI upstreams reject explicit nulls.
    return {k: v for k, v in chat.items() if v is not None}


def _usage_from_chat(usage):
    if not isinstance(usage, dict):
        usage = {}
    input_details = usage.get("prompt_tokens_details") or {}
    output_details = usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "input_tokens_details": {
            "text_tokens": input_details.get("text_tokens", usage.get("prompt_tokens", 0)),
            "audio_tokens": input_details.get("audio_tokens", 0),
            "image_tokens": input_details.get("image_tokens", 0),
            "cached_tokens": input_details.get("cached_tokens", 0),
        },
        "output_tokens": usage.get("completion_tokens", 0),
        "output_tokens_details": {
            "reasoning_tokens": output_details.get("reasoning_tokens", 0),
            "audio_tokens": output_details.get("audio_tokens", 0),
            "accepted_prediction_tokens": output_details.get("accepted_prediction_tokens", 0),
            "rejected_prediction_tokens": output_details.get("rejected_prediction_tokens", 0),
        },
        "total_tokens": usage.get("total_tokens", 0),
        "num_sources_used": usage.get("num_sources_used", 0),
        "num_server_side_tools_used": usage.get("num_server_side_tools_used", 0),
        "server_side_tool_usage_details": usage.get("server_side_tool_usage_details"),
        "context_details": usage.get("context_details"),
        "cost_in_nano_usd": usage.get("cost_in_nano_usd"),
        "cost_in_usd_ticks": usage.get("cost_in_usd_ticks"),
    }


def _status_from_finish_reason(reason):
    if reason == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    if reason == "content_filter":
        return "incomplete", {"reason": "content_filter"}
    return "completed", None


SERVER_TOOL_TYPES = {
    "web_search", "web_search_preview", "x_search", "file_search",
    "code_interpreter", "mcp", "computer_use_preview", "image_generation",
    "tool_search",
}


def _has_native_server_tools(req):
    """True when the request needs NewAPI's native Responses server tools."""
    return any(
        isinstance(tool, dict) and tool.get("type") in SERVER_TOOL_TYPES
        for tool in (req.get("tools") or [])
    )


def _normalize_native_usage(usage):
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}

    # Derive server-side usage counts from the response output when available.
    server_counts = {
        "web_search_calls": 0, "x_search_calls": 0, "x_posts_fetched": 0,
        "x_users_fetched": 0, "code_interpreter_calls": 0,
        "file_search_calls": 0, "mcp_calls": 0,
        "document_search_calls": 0, "image_generation_calls": 0,
    }
    return {
        **usage,
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "text_tokens": input_details.get("text_tokens", input_tokens),
            "audio_tokens": input_details.get("audio_tokens", 0),
            "image_tokens": input_details.get("image_tokens", 0),
            "cached_tokens": input_details.get("cached_tokens", 0),
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": output_details.get("reasoning_tokens", 0),
            "audio_tokens": output_details.get("audio_tokens", 0),
            "accepted_prediction_tokens": output_details.get("accepted_prediction_tokens", 0),
            "rejected_prediction_tokens": output_details.get("rejected_prediction_tokens", 0),
        },
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        "num_sources_used": usage.get("num_sources_used", 0),
        "num_server_side_tools_used": usage.get(
            "num_server_side_tools_used", sum(server_counts.values())
        ),
        "server_side_tool_usage_details": usage.get(
            "server_side_tool_usage_details", server_counts
        ),
        "context_details": usage.get("context_details"),
        "cost_in_nano_usd": usage.get("cost_in_nano_usd"),
        "cost_in_usd_ticks": usage.get("cost_in_usd_ticks"),
    }


def _normalize_native_request(req):
    """Normalize malformed Grok history before native Responses passthrough.

    Older bridge releases emitted streamed function_call items with an empty
    name.  Those items may remain in Grok's existing session history.  xAI's
    FunctionToolCall schema requires name, and NewAPI reports this as
    "missing input.name"; repair only these malformed items rather than
    dropping conversation history.
    """
    if not isinstance(req, dict):
        return req
    req = copy.deepcopy(req)
    input_items = req.get("input")
    if not isinstance(input_items, list):
        # Also remove harmless explicit null names from tool declarations.
        for tool in req.get("tools") or []:
            if isinstance(tool, dict) and tool.get("name") is None and tool.get("type") != "function":
                tool.pop("name", None)
        return req

    known_names = {}
    for item in input_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in ("function_call", "custom_tool_call"):
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            if call_id and isinstance(name, str) and name:
                known_names[str(call_id)] = name

    for item in input_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        # Some Grok builds annotate hosted tools with an explicit null name.
        # NewAPI accepts these tools without the key, while null can fail its
        # strict server-tool decoder.
        if item_type not in ("function_call", "custom_tool_call") and item.get("name") is None:
            item.pop("name", None)

        if item_type in ("function_call", "custom_tool_call"):
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            if not isinstance(name, str) or not name:
                item["name"] = known_names.get(str(call_id), "unknown_tool")
            if item_type == "function_call" and not isinstance(item.get("arguments"), str):
                item["arguments"] = "{}"

        # Repair malformed server-side calls which can also surface as
        # missing input.name in NewAPI's native decoder.
        if item_type == "mcp_call" and not isinstance(item.get("name"), str):
            item["name"] = "unknown_tool"
        if item_type == "mcp_call" and not isinstance(item.get("server_label"), str):
            item["server_label"] = "grok_compat"

    for tool in req.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("name") is None and tool.get("type") != "function":
            tool.pop("name", None)
        fn = tool.get("function")
        if tool.get("type") == "function" and isinstance(fn, dict) and not isinstance(fn.get("name"), str):
            tool.pop("function", None)
            tool.pop("type", None)
    return req


def _normalize_native_annotation(annotation):
    """Grok requires citation indexes even when NewAPI omits them."""
    if isinstance(annotation, dict):
        if not isinstance(annotation.get("start_index"), int):
            annotation["start_index"] = 0
        if not isinstance(annotation.get("end_index"), int):
            annotation["end_index"] = 0
    return annotation


def _normalize_native_output_text_part(part):
    """Grok's strict OutputTextContent requires annotations on every part."""
    if isinstance(part, dict) and part.get("type") == "output_text":
        if not isinstance(part.get("annotations"), list):
            part["annotations"] = []
        part["annotations"] = [_normalize_native_annotation(x) for x in part["annotations"]]
    return part


def _normalize_native_output_item(item):
    """Make in-progress native output items satisfy xAI's item schemas."""
    if not isinstance(item, dict):
        return item
    item_type = item.get("type")
    if item_type == "function_call" and not isinstance(item.get("arguments"), str):
        item["arguments"] = ""
    elif item_type == "custom_tool_call" and not isinstance(item.get("input"), str):
        item["input"] = ""
    elif item_type == "web_search_call" and not isinstance(item.get("action"), dict):
        item["action"] = {"type": "search", "query": ""}
    elif item_type == "shell_call" and not isinstance(item.get("action"), dict):
        action = {"type": "exec", "commands": []}
        if isinstance(item.get("timeout_ms"), int):
            action["timeout_ms"] = item["timeout_ms"]
        if isinstance(item.get("max_output_length"), int):
            action["max_output_length"] = item["max_output_length"]
        item["action"] = action
    elif item_type == "mcp_call":
        if not isinstance(item.get("arguments"), str):
            item["arguments"] = "{}"
        if not isinstance(item.get("output"), str):
            item["output"] = ""
        if not isinstance(item.get("server_label"), str):
            item["server_label"] = "grok_compat"
        if not isinstance(item.get("name"), str):
            item["name"] = "unknown_tool"
    elif item_type == "custom_tool_call":
        if not isinstance(item.get("id"), str) or not item.get("id"):
            item["id"] = item.get("call_id") or _rid("ctc")
    elif item_type == "file_search_call":
        if not isinstance(item.get("queries"), list):
            item["queries"] = []
        if not isinstance(item.get("results"), list):
            item["results"] = []
    elif item_type == "code_interpreter_call":
        if not isinstance(item.get("outputs"), list):
            item["outputs"] = []
    elif item_type in ("tool_search_call", "tool_search_output"):
        if not isinstance(item.get("id"), str) or not item.get("id"):
            item["id"] = _rid("ts")
    return item


def _normalize_native_sse_payload(payload):
    """Materialize xAI-required fields inside native NewAPI SSE events."""
    if not isinstance(payload, dict):
        return payload

    event_type = payload.get("type")
    if event_type in ("response.content_part.added", "response.content_part.done"):
        payload["part"] = _normalize_native_output_text_part(payload.get("part"))
    elif event_type == "response.output_text.done":
        if not isinstance(payload.get("annotations"), list):
            payload["annotations"] = []
        payload["annotations"] = [_normalize_native_annotation(x) for x in payload["annotations"]]
    elif event_type == "response.output_text.annotation.added":
        payload["annotation"] = _normalize_native_annotation(payload.get("annotation"))
    elif event_type in ("response.output_item.added", "response.output_item.done"):
        item = payload.get("item")
        if isinstance(item, dict):
            if item.get("type") == "message":
                content = item.get("content")
                if isinstance(content, list):
                    item["content"] = [_normalize_native_output_text_part(x) for x in content]
            else:
                payload["item"] = _normalize_native_output_item(item)
    return payload


def _normalize_native_response(req, response):
    """Add xAI ModelResponse required fields to NewAPI's native response."""
    if not isinstance(response, dict):
        return response
    response = copy.deepcopy(response)
    status = response.get("status", "completed")
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                if item.get("type") == "message":
                    content = item.get("content")
                    if isinstance(content, list):
                        item["content"] = [_normalize_native_output_text_part(x) for x in content]
                else:
                    _normalize_native_output_item(item)
    response.setdefault("object", "response")
    response.setdefault("error", None)
    response.setdefault("incomplete_details", None)
    response.setdefault("output", [])
    response.setdefault("parallel_tool_calls", bool(req.get("parallel_tool_calls", False)))
    response.setdefault("previous_response_id", req.get("previous_response_id"))
    response.setdefault("reasoning", req.get("reasoning"))
    response.setdefault("store", bool(req.get("store", True)))
    response.setdefault("temperature", req.get("temperature"))
    response.setdefault("text", _response_text_field(req))
    response.setdefault("tool_choice", _response_tool_choice(req))
    response.setdefault("tools", req.get("tools") or [])
    response.setdefault("top_p", req.get("top_p"))
    response.setdefault("max_output_tokens", req.get("max_output_tokens"))
    response.setdefault("metadata", req.get("metadata") or {})
    response.setdefault("instructions", req.get("instructions"))
    response.setdefault("user", req.get("user"))
    response.setdefault("background", bool(req.get("background", False)))
    service_tier = req.get("service_tier")
    if service_tier not in ("default", "priority"):
        service_tier = "default"
    response.setdefault("service_tier", service_tier)
    response.setdefault("truncation", req.get("truncation", "disabled"))
    response.setdefault("top_logprobs", int(req.get("top_logprobs") or 0))
    response.setdefault("presence_penalty", req.get("presence_penalty", 0))
    response.setdefault("frequency_penalty", req.get("frequency_penalty", 0))
    response.setdefault(
        "completed_at", int(time.time()) if status != "in_progress" else None
    )
    response["usage"] = _normalize_native_usage(response.get("usage"))
    return response


def _response_tool_choice(req):
    """xAI ModelResponse requires tool_choice; null is not a valid variant."""
    choice = req.get("tool_choice")
    if choice in ("auto", "none", "required"):
        return choice
    if isinstance(choice, dict):
        if choice.get("type") == "function" and choice.get("name"):
            return {"type": "function", "name": choice["name"]}
        function = choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return {"type": "function", "name": function["name"]}
    return "auto"


def _response_object(
    req, *, response_id, status, output, model, created_at, usage=None,
    incomplete_details=None, completed_at=None,
):
    """Canonical xAI /v1/responses ModelResponse shape.

    Source: https://docs.x.ai/openapi.json, components.schemas.ModelResponse.
    Required response fields omitted by the earlier OpenAI-shaped bridge are
    materialized here so Grok's strict client can deserialize every event.
    """
    service_tier = req.get("service_tier")
    if service_tier not in ("default", "priority"):
        service_tier = "default"
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "incomplete_details": incomplete_details,
        "model": model,
        "output": output,
        "parallel_tool_calls": bool(req.get("parallel_tool_calls", False)),
        "previous_response_id": req.get("previous_response_id"),
        "reasoning": req.get("reasoning"),
        "store": bool(req.get("store", True)),
        "temperature": req.get("temperature"),
        "text": _response_text_field(req),
        "tool_choice": _response_tool_choice(req),
        "tools": req.get("tools") or [],
        "top_p": req.get("top_p"),
        "max_output_tokens": req.get("max_output_tokens"),
        "metadata": req.get("metadata") or {},
        "instructions": req.get("instructions"),
        "user": req.get("user"),
        "usage": usage,
        # Required by xAI's ModelResponse schema and useful for strict clients.
        "background": bool(req.get("background", False)),
        "service_tier": service_tier,
        "truncation": req.get("truncation", "disabled"),
        "top_logprobs": int(req.get("top_logprobs") or 0),
        "presence_penalty": req.get("presence_penalty", 0),
        "frequency_penalty": req.get("frequency_penalty", 0),
        "completed_at": completed_at,
    }


def _nonstream_response_object(req, chat):
    choices = chat.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    status, incomplete_details = _status_from_finish_reason(finish_reason)

    output = []
    reasoning = _reasoning_text(message.get("reasoning_content") or message.get("reasoning"))
    if reasoning:
        output.append({
            "id": _rid("rs"),
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reasoning}],
        })

    content_text = _text_from_content(message.get("content"))
    content_text = _normalize_structured_output_text(req, content_text)
    if content_text or message.get("tool_calls") is None:
        output.append({
            "id": _rid("msg"),
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": content_text,
                "annotations": [],
            }],
        })

    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        call_id = call.get("id") or _rid("call")
        if function.get("name") == "shell" and _has_shell_tool(req):
            try:
                parsed_arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                parsed_arguments = {}
            if not isinstance(parsed_arguments, dict):
                parsed_arguments = {}
            action = {
                "type": "exec",
                "commands": parsed_arguments.get("commands", []),
            }
            if parsed_arguments.get("timeout_ms") is not None:
                action["timeout_ms"] = parsed_arguments["timeout_ms"]
            if parsed_arguments.get("max_output_length") is not None:
                action["max_output_length"] = parsed_arguments["max_output_length"]
            output.append({
                "id": call_id,
                "type": "shell_call",
                "status": status,
                "call_id": call_id,
                "action": action,
            })
            continue
        output.append({
            "id": call.get("id") or _rid("fc"),
            "type": "function_call",
            "status": status,
            "call_id": call_id,
            "name": function.get("name", ""),
            "arguments": function.get("arguments", "{}"),
        })

    return _response_object(
        req,
        response_id="resp_" + str(chat.get("id") or uuid.uuid4().hex),
        status=status,
        output=output,
        model=chat.get("model", req.get("model", "")),
        created_at=chat.get("created", int(time.time())),
        usage=_usage_from_chat(chat.get("usage")),
        incomplete_details=incomplete_details,
        completed_at=int(time.time()),
    )


def chat_output_messages_from_response(response):
    """Create replayable Chat messages from a transformed Responses object."""
    messages = []
    for item in response.get("output", []):
        item_type = item.get("type")
        if item_type == "message":
            text = "".join(
                part.get("text", "") for part in item.get("content", [])
                if isinstance(part, dict) and part.get("type") == "output_text"
            )
            messages.append({"role": "assistant", "content": text})
        elif item_type == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }],
            })
        elif item_type == "shell_call":
            action = item.get("action") or {}
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": json.dumps({
                            "commands": action.get("commands", []),
                            "timeout_ms": action.get("timeout_ms"),
                            "max_output_length": action.get("max_output_length"),
                        }, ensure_ascii=False),
                    },
                }],
            })
    return messages


class StreamBridge:
    """Chat Completions SSE -> OpenAI Responses SSE."""

    def __init__(self, req, request_messages, write_event):
        self.req = req
        self.request_messages = request_messages
        self.write_event = write_event
        self.sequence = 0
        self.response_id = _rid("resp")
        self.created_at = int(time.time())
        self.finish_reason = None
        self.usage = None
        self.model = req.get("model", "")
        self.reasoning_id = None
        self.reasoning_output_index = None
        self.reasoning_text = ""
        self.message_id = None
        self.message_output_index = None
        self.message_text = ""
        self.tools = {}
        self.output_count = 0
        self.completed = False

    def _base_response(self, status="in_progress", output=None):
        return _response_object(
            self.req,
            response_id=self.response_id,
            status=status,
            output=output or [],
            model=self.model,
            created_at=self.created_at,
            usage=_usage_from_chat(self.usage) if self.usage else None,
            incomplete_details=None,
            completed_at=int(time.time()) if status != "in_progress" else None,
        )

    def emit(self, etype, payload):
        self.sequence += 1
        payload = dict(payload)
        payload["type"] = etype
        payload["sequence_number"] = self.sequence
        self.write_event(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def start(self):
        self.emit("response.created", {"response": self._base_response()})
        self.emit("response.in_progress", {"response": self._base_response()})

    def _start_reasoning(self):
        if self.reasoning_id:
            return
        self.reasoning_id = _rid("rs")
        output_index = self.output_count
        self.output_count += 1
        self.reasoning_output_index = output_index
        self.emit("response.output_item.added", {
            "output_index": output_index,
            "item": {
                "id": self.reasoning_id,
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
            },
        })
        self.emit("response.reasoning_summary_part.added", {
            "item_id": self.reasoning_id,
            "output_index": output_index,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": ""},
        })

    def _start_message(self):
        if self.message_id:
            return
        self.message_id = _rid("msg")
        output_index = self.output_count
        self.output_count += 1
        self.message_output_index = output_index
        self.emit("response.output_item.added", {
            "output_index": output_index,
            "item": {
                "id": self.message_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        })
        self.emit("response.content_part.added", {
            "item_id": self.message_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })

    def _start_tool(self, index, call):
        function = call.get("function") or {}
        created = False
        if index not in self.tools:
            item_id = _rid("fc")
            state = {
                "item_id": item_id,
                "call_id": call.get("id") or f"call_{item_id}",
                "name": function.get("name") or "",
                "arguments": "",
                "kind": "shell" if function.get("name") == "shell" and _has_shell_tool(self.req) else "function",
                "output_index": self.output_count,
            }
            self.tools[index] = state
            self.output_count += 1
            created = True
        else:
            state = self.tools[index]
            if call.get("id"):
                state["call_id"] = call["id"]
            if function.get("name"):
                state["name"] = function["name"]
                state["kind"] = "shell" if function["name"] == "shell" and _has_shell_tool(self.req) else "function"

        if not created:
            return

        if state.get("kind") == "shell":
            item = {
                "id": state["item_id"],
                "type": "shell_call",
                "status": "in_progress",
                "call_id": state["call_id"],
                "action": {"type": "exec", "commands": []},
            }
        else:
            item = {
                "id": state["item_id"],
                "type": "function_call",
                "status": "in_progress",
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": "",
            }
        self.emit("response.output_item.added", {
            "output_index": state["output_index"],
            "item": item,
        })

    def process_chunk(self, chunk):
        if not isinstance(chunk, dict):
            return
        if chunk.get("model"):
            self.model = chunk["model"]
        if chunk.get("usage"):
            self.usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        delta = choice.get("delta") or {}
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]

        reasoning = _reasoning_text(delta.get("reasoning_content") or delta.get("reasoning"))
        if reasoning:
            self._start_reasoning()
            self.reasoning_text += reasoning
            self.emit("response.reasoning_summary_text.delta", {
                "item_id": self.reasoning_id,
                "output_index": self.output_count - 1,
                "summary_index": 0,
                "delta": reasoning,
            })

        content = _text_from_content(delta.get("content"))
        if content:
            structured = _response_json_schema(self.req) is not None
            self._start_message()
            self.message_text += content
            if not structured:
                self.emit("response.output_text.delta", {
                    "item_id": self.message_id,
                    "output_index": self.output_count - 1,
                    "content_index": 0,
                    "delta": content,
                })

        for raw_call in delta.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            index = raw_call.get("index", 0)
            self._start_tool(index, raw_call)
            state = self.tools[index]
            function = raw_call.get("function") or {}
            arguments_delta = function.get("arguments") or ""
            if arguments_delta:
                state["arguments"] += arguments_delta
                if state.get("kind") != "shell":
                    self.emit("response.function_call_arguments.delta", {
                        "item_id": state["item_id"],
                        "output_index": state["output_index"],
                        "delta": arguments_delta,
                    })

    def finish_items(self):
        if self.reasoning_id:
            output_index = self.reasoning_output_index
            self.emit("response.reasoning_summary_text.done", {
                "item_id": self.reasoning_id,
                "output_index": output_index,
                "summary_index": 0,
                "text": self.reasoning_text,
            })
            self.emit("response.reasoning_summary_part.done", {
                "item_id": self.reasoning_id,
                "output_index": output_index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": self.reasoning_text},
            })
            self.emit("response.output_item.done", {
                "output_index": output_index,
                "item": {
                    "id": self.reasoning_id,
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": self.reasoning_text}],
                },
            })

        if self.message_id:
            self.message_text = _normalize_structured_output_text(self.req, self.message_text)
            output_index = self.message_output_index
            self.emit("response.output_text.done", {
                "item_id": self.message_id,
                "output_index": output_index,
                "content_index": 0,
                "text": self.message_text,
            })
            self.emit("response.content_part.done", {
                "item_id": self.message_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": self.message_text,
                    "annotations": [],
                },
            })
            self.emit("response.output_item.done", {
                "output_index": output_index,
                "item": {
                    "id": self.message_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": self.message_text,
                        "annotations": [],
                    }],
                },
            })

        for state in self.tools.values():
            if state.get("kind") == "shell":
                try:
                    parsed_arguments = json.loads(state["arguments"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    parsed_arguments = {}
                if not isinstance(parsed_arguments, dict):
                    parsed_arguments = {}
                action = {
                    "type": "exec",
                    "commands": parsed_arguments.get("commands", []),
                }
                if parsed_arguments.get("timeout_ms") is not None:
                    action["timeout_ms"] = parsed_arguments["timeout_ms"]
                if parsed_arguments.get("max_output_length") is not None:
                    action["max_output_length"] = parsed_arguments["max_output_length"]
                self.emit("response.output_item.done", {
                    "output_index": state["output_index"],
                    "item": {
                        "id": state["item_id"],
                        "type": "shell_call",
                        "status": "completed",
                        "call_id": state["call_id"],
                        "action": action,
                    },
                })
            else:
                self.emit("response.function_call_arguments.done", {
                    "item_id": state["item_id"],
                    "output_index": state["output_index"],
                    "arguments": state["arguments"],
                })
                self.emit("response.output_item.done", {
                    "output_index": state["output_index"],
                    "item": {
                        "id": state["item_id"],
                        "type": "function_call",
                        "status": "completed",
                        "call_id": state["call_id"],
                        "name": state["name"],
                        "arguments": state["arguments"],
                    },
                })

    def final_output(self):
        output = []
        if self.reasoning_id:
            output.append({
                "id": self.reasoning_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": self.reasoning_text}],
            })
        if self.message_id:
            output.append({
                "id": self.message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": self.message_text,
                    "annotations": [],
                }],
            })
        for state in self.tools.values():
            if state.get("kind") == "shell":
                try:
                    parsed_arguments = json.loads(state["arguments"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    parsed_arguments = {}
                if not isinstance(parsed_arguments, dict):
                    parsed_arguments = {}
                action = {
                    "type": "exec",
                    "commands": parsed_arguments.get("commands", []),
                }
                if parsed_arguments.get("timeout_ms") is not None:
                    action["timeout_ms"] = parsed_arguments["timeout_ms"]
                if parsed_arguments.get("max_output_length") is not None:
                    action["max_output_length"] = parsed_arguments["max_output_length"]
                output.append({
                    "id": state["item_id"],
                    "type": "shell_call",
                    "status": "completed",
                    "call_id": state["call_id"],
                    "action": action,
                })
            else:
                output.append({
                    "id": state["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": state["call_id"],
                    "name": state["name"],
                    "arguments": state["arguments"],
                })
        return output

    def finish(self):
        if self.completed:
            return
        self.finish_items()
        status, incomplete_details = _status_from_finish_reason(self.finish_reason)
        response = self._base_response(
            status=status,
            output=self.final_output(),
        )
        if response.get("usage") is None:
            # xAI's goal evaluator rejects a terminal response without usage
            # accounting.  Keep the terminal frame valid even if a broken
            # upstream omitted Chat Completions usage.
            response["usage"] = _usage_from_chat({})
        response["incomplete_details"] = incomplete_details
        self.emit("response.completed", {"response": response})
        self.completed = True

        replay = copy.deepcopy(self.request_messages)
        tool_calls = []
        for state in self.tools.values():
            tool_calls.append({
                "id": state["call_id"],
                "type": "function",
                "function": {
                    "name": state["name"],
                    "arguments": state["arguments"] or "{}",
                },
            })
        assistant_message = {
            "role": "assistant",
            "content": self.message_text or None,
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        replay.append(assistant_message)
        remember_response(self.response_id, replay)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        logging.info("%s %s", self.address_string(), fmt % args)

    def _path(self):
        path = self.path
        parsed = urllib.parse.urlsplit(path)
        return parsed.path or "/"

    def _upstream_headers(self, body=None):
        headers = {}
        for name, value in self.headers.items():
            key = name.lower()
            if key in HOP_BY_HOP:
                continue
            headers[name] = value
        headers["Host"] = f"{UPSTREAM_HOST}:{UPSTREAM_PORT}"
        headers["Accept-Encoding"] = "identity"
        headers["Connection"] = "close"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        return headers

    def _read_request_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return b""
        return self.rfile.read(length) if length else b""

    def _send_json(self, status, obj, headers=None):
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status, message):
        self._send_json(status, {
            "error": {"message": message, "type": "grok_compat_error"},
        })

    def _begin_upstream_response(self, response, content_type):
        self.send_response(response.status, response.reason)
        for name, value in response.getheaders():
            key = name.lower()
            if key in HOP_BY_HOP or key == "content-length":
                continue
            self.send_header(name, value)
        if content_type and "text/event-stream" in content_type:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

    def _proxy_passthrough(self, path, body=b""):
        method = self.command
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=UPSTREAM_TIMEOUT)
            conn.request(method, path, body=body if body else None, headers=self._upstream_headers(body if body else None))
            response = conn.getresponse()
            content_type = response.getheader("Content-Type", "").lower()
            self._begin_upstream_response(response, content_type)
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, http.client.HTTPException) as exc:
            logging.warning("upstream error: %s", exc)
            try:
                self._send_error_json(502, f"NewAPI unavailable: {exc}")
            except Exception:
                pass
        finally:
            try:
                response.close()
                conn.close()
            except Exception:
                pass

    def _proxy_native_responses(self, path, raw_body, req):
        """Pass Responses requests through to NewAPI and normalize its wire."""
        try:
            # Diagnostic captures are opt-in because they contain request and
            # response content, but never the Authorization header.
            if DEBUG_CAPTURE:
                try:
                    os.makedirs(DEBUG_DIR, exist_ok=True)
                    with open(os.path.join(DEBUG_DIR, "native_request.json"), "wb") as f:
                        f.write(raw_body)
                except OSError:
                    logging.warning("could not save native request body")

            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=UPSTREAM_TIMEOUT)
            conn.request("POST", path, body=raw_body, headers=self._upstream_headers(raw_body))
            response = conn.getresponse()
            content_type = response.getheader("Content-Type", "").lower()

            if response.status < 200 or response.status >= 300:
                payload = response.read()
                if DEBUG_CAPTURE:
                    try:
                        os.makedirs(DEBUG_DIR, exist_ok=True)
                        debug_body = json.dumps(
                            _normalize_native_request(req),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        with open(os.path.join(DEBUG_DIR, "native_failed.json"), "wb") as f:
                            f.write(debug_body)
                        logging.info("native error body saved to %s", DEBUG_DIR)
                    except Exception:
                        logging.warning("could not save native error body")
                self.send_response(response.status, response.reason)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                return

            if "text/event-stream" not in content_type:
                payload = response.read()
                try:
                    native = json.loads(payload.decode("utf-8"))
                    normalized = _normalize_native_response(req, native)
                    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
                self._begin_upstream_response(response, content_type)
                self.wfile.write(payload)
                self.wfile.flush()
                return

            self._begin_upstream_response(response, content_type)
            debug_upstream = None
            debug_forwarded = None
            if DEBUG_CAPTURE:
                try:
                    os.makedirs(DEBUG_DIR, exist_ok=True)
                    debug_upstream = open(os.path.join(DEBUG_DIR, "native_upstream.sse"), "wb")
                    debug_forwarded = open(os.path.join(DEBUG_DIR, "native_forwarded.sse"), "wb")
                except OSError:
                    logging.warning("could not save native SSE capture")
            sequence = 0
            while True:
                line = response.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                if debug_upstream:
                    try:
                        debug_upstream.write(line)
                        debug_upstream.flush()
                    except OSError:
                        pass
                if text.startswith("event:"):
                    self.wfile.write(line)
                    continue
                if not text.startswith("data:"):
                    self.wfile.write(line)
                    continue
                raw = text[5:].strip()
                if raw == "[DONE]":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    self.wfile.write(line)
                    continue
                if isinstance(payload, dict):
                    payload["sequence_number"] = sequence
                    sequence += 1
                    if payload.get("type") == "response.reasoning_summary_part.added":
                        part = payload.get("part")
                        if isinstance(part, dict) and part.get("type") == "summary_text" and not part.get("text"):
                            part["text"] = ""
                    payload = _normalize_native_sse_payload(payload)
                    if payload.get("type") in ("response.created", "response.in_progress", "response.completed"):
                        native_response = payload.get("response")
                        if isinstance(native_response, dict):
                            payload["response"] = _normalize_native_response(req, native_response)
                forwarded = b"data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"
                if debug_forwarded:
                    try:
                        debug_forwarded.write(forwarded)
                        debug_forwarded.flush()
                    except OSError:
                        pass
                self.wfile.write(forwarded)
                self.wfile.flush()
            try:
                if debug_upstream:
                    debug_upstream.close()
                if debug_forwarded:
                    debug_forwarded.close()
            except Exception:
                pass
            return

        except (BrokenPipeError, ConnectionResetError):
            logging.info("client disconnected")
        except (OSError, http.client.HTTPException) as exc:
            logging.warning("native responses upstream error: %s", exc)
            try:
                self._send_error_json(502, f"NewAPI unavailable: {exc}")
            except Exception:
                pass
        finally:
            try:
                response.close()
                conn.close()
            except Exception:
                pass

    def _handle_responses(self, path):
        raw_body = self._read_request_body()
        try:
            req = json.loads(raw_body.decode("utf-8") or "{}")
            if not isinstance(req, dict):
                raise ValueError("request body must be a JSON object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send_error_json(400, f"invalid Responses request: {exc}")
            return

        # NewAPI's native /v1/responses implements server-side tools such as
        # web_search.  Forcing these through Chat Completions would lose them.
        if _has_native_server_tools(req):
            normalized_req = _normalize_native_request(req)
            normalized_body = json.dumps(
                normalized_req, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self._proxy_native_responses("/v1/responses", normalized_body, normalized_req)
            return

        previous_id = req.get("previous_response_id")
        chat = responses_to_chat_request(req)
        request_messages = copy.deepcopy(chat.get("messages", []))
        if previous_id:
            chat["messages"] = previous_messages(previous_id) + chat["messages"]

        incoming_tools = []
        for tool in req.get("tools") or []:
            if isinstance(tool, dict):
                incoming_tools.append({
                    "type": tool.get("type"),
                    "name": tool.get("name") or (tool.get("function") or {}).get("name"),
                })
        outgoing_tools = [
            tool.get("function", {}).get("name") for tool in chat.get("tools") or []
        ]
        if incoming_tools:
            logging.info("tools incoming=%s outgoing=%s", incoming_tools, outgoing_tools)

        body = json.dumps(chat, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        upstream_path = "/v1/chat/completions"

        # Goal evaluation uses a smaller Chat-only request. Capture it separately
        # so ordinary large native Responses captures are not overwritten.
        goal_debug_base = None
        goal_upstream = None
        request_id = self.headers.get("X-Grok-Req-Id", "")
        if DEBUG_CAPTURE and request_id.startswith("xai-goal-eval"):
            try:
                goal_debug_dir = os.path.join(DEBUG_DIR, "goal")
                os.makedirs(goal_debug_dir, exist_ok=True)
                stamp = str(int(time.time() * 1000))
                safe_req_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in request_id)
                goal_debug_base = os.path.join(goal_debug_dir, f"{stamp}-{safe_req_id}")
                with open(goal_debug_base + "-responses-request.json", "wb") as f:
                    f.write(raw_body)
                with open(goal_debug_base + "-chat-request.json", "wb") as f:
                    f.write(body)
                logging.info("goal eval debug captured to %s", goal_debug_dir)
            except OSError:
                logging.warning("could not save goal eval request capture")
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=UPSTREAM_TIMEOUT)
            conn.request("POST", upstream_path, body=body, headers=self._upstream_headers(body))
            response = conn.getresponse()
            content_type = response.getheader("Content-Type", "").lower()

            # Preserve NewAPI's own authentication/validation errors.
            if response.status < 200 or response.status >= 300:
                payload = response.read()
                self.send_response(response.status, response.reason)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                return

            if req.get("stream"):
                if goal_debug_base:
                    try:
                        goal_upstream = open(goal_debug_base + "-upstream.sse", "wb")
                    except OSError:
                        logging.warning("could not save goal eval upstream capture")
                self._begin_upstream_response(response, content_type)
                bridge = StreamBridge(req, request_messages, lambda data: self._write_sse(data))
                bridge.start()

                def upstream_chunks():
                    while True:
                        raw = response.read(64 * 1024)
                        if not raw:
                            return
                        if goal_upstream:
                            try:
                                goal_upstream.write(raw)
                                goal_upstream.flush()
                            except OSError:
                                pass
                        yield raw

                for chunk in _iter_chat_sse_chunks(upstream_chunks()):
                    bridge.process_chunk(chunk)
                bridge.finish()
                if goal_upstream:
                    try:
                        goal_upstream.close()
                    except Exception:
                        pass
                self.wfile.flush()
                return

            payload = response.read()
            try:
                chat_response = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(502, {"error": {"message": "NewAPI returned invalid JSON"}})
                return
            transformed = _nonstream_response_object(req, chat_response)
            messages = copy.deepcopy(request_messages)
            messages.extend(chat_output_messages_from_response(transformed))
            remember_response(transformed["id"], messages)
            self._send_json(200, transformed)
        except (BrokenPipeError, ConnectionResetError):
            logging.info("client disconnected")
        except (OSError, http.client.HTTPException) as exc:
            logging.warning("upstream error: %s", exc)
            try:
                self._send_error_json(502, f"NewAPI unavailable: {exc}")
            except Exception:
                pass
        finally:
            try:
                response.close()
                conn.close()
            except Exception:
                pass

    def _write_sse(self, data):
        self.wfile.write(b"data: " + data.encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def do_GET(self):
        self._proxy_passthrough(self._path())

    def do_POST(self):
        path = self._path()
        if path in ("/v1/responses", "/responses", "/openai/v1/responses"):
            self._handle_responses(path)
        else:
            self._proxy_passthrough(path, self._read_request_body())

    def do_PUT(self):
        self._proxy_passthrough(self._path(), self._read_request_body())

    def do_PATCH(self):
        self._proxy_passthrough(self._path(), self._read_request_body())

    def do_DELETE(self):
        self._proxy_passthrough(self._path(), self._read_request_body())

    def do_OPTIONS(self):
        self._proxy_passthrough(self._path(), self._read_request_body())

    def do_HEAD(self):
        self._proxy_passthrough(self._path())


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        super().server_bind()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("GROK_COMPAT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info(
        "grok Responses bridge listening on %s:%d -> NewAPI %s:%d (auth passthrough)",
        LISTEN_HOST, LISTEN_PORT, UPSTREAM_HOST, UPSTREAM_PORT,
    )
    Server((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()
