#!/usr/bin/env python3
# Harness: tool dispatch -- expanding what the model can reach (OpenAI version).
"""
s02_tool_use.py - Tool dispatch + message normalization (OpenAI SDK)
The agent loop is the same shape as the Anthropic version. We adapt:
1. Client and API call to OpenAI's chat.completions
2. Tool schemas to OpenAI's function-calling format
3. Response parsing from message.tool_calls instead of content blocks
4. Tool result messages use role="tool" with tool_call_id
Key insight: "The loop didn't change at all. I just added tools."
"""

import os
import logging
import subprocess
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()

cli = OpenAI(
    base_url=os.getenv("MOONSHOT_BASE_URL"),
    api_key=os.getenv("MOONSHOT_API_KEY"),
)

MODEL = os.getenv("MOONSHOT_LATEST_MODEL")

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Concurrency safety classification --
CONCURRENCY_SAFE = {"read_file"}
CONCURRENCY_UNSAFE = {"write_file", "edit_file"}

# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# OpenAI expects tools in the function-calling format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]


def normalize_messages(messages: list) -> list:
    """Clean up messages before sending to the API.
    Three jobs:
    1. Strip internal metadata fields the API doesn't understand
    2. Ensure every tool_call has a matching tool result (insert placeholder if missing)
    3. Merge consecutive same-role messages (API requires strict alternation)
    """
    cleaned = []
    for msg in messages:
        clean = {"role": msg["role"]}
        if isinstance(msg.get("content"), str):
            clean["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            clean["content"] = [
                {k: v for k, v in block.items() if not k.startswith("_")}
                for block in msg["content"]
                if isinstance(block, dict)
            ]
        else:
            clean["content"] = msg.get("content", "")
        # Preserve tool_calls, tool_call_id, and reasoning_content if present
        if "tool_calls" in msg:
            clean["tool_calls"] = msg["tool_calls"]
        if "tool_call_id" in msg:
            clean["tool_call_id"] = msg["tool_call_id"]
        if "reasoning_content" in msg:
            clean["reasoning_content"] = msg["reasoning_content"]
        cleaned.append(clean)

    # Collect existing tool result IDs
    existing_results = set()
    for msg in cleaned:
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            existing_results.add(msg["tool_call_id"])

    # Find orphaned tool_calls blocks and insert placeholder results
    for msg in cleaned:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            if tc.get("id") not in existing_results:
                cleaned.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "(cancelled)",
                    }
                )

    # Merge consecutive same-role messages (OpenAI also requires alternating roles)
    if not cleaned:
        return cleaned
    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        if msg["role"] == merged[-1]["role"] and msg.get("role") != "tool":
            prev = merged[-1]
            prev_c = prev["content"] if isinstance(prev.get("content"), str) else str(prev.get("content", ""))
            curr_c = msg["content"] if isinstance(msg.get("content"), str) else str(msg.get("content", ""))
            prev["content"] = prev_c + "\n" + curr_c
        else:
            merged.append(msg)
    return merged


def agent_loop(messages: list):
    while True:
        response = cli.chat.completions.create(
            model=MODEL,
            # messages=[{"role": "system", "content": SYSTEM}] + normalize_messages(messages),
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )
        choice = response.choices[0]
        msg = choice.message

        # # Build the assistant message to append
        # assistant_msg = {"role": "assistant", "content": msg.content or ""}
        # # Preserve reasoning_content for APIs that require it on thinking models
        # if getattr(msg, "reasoning_content", None):
        #     assistant_msg["reasoning_content"] = msg.reasoning_content

        # if msg.tool_calls:
        #     assistant_msg["tool_calls"] = [
        #         {
        #             "id": tc.id,
        #             "type": tc.type,
        #             "function": {
        #                 "name": tc.function.name,
        #                 "arguments": tc.function.arguments,
        #             },
        #         }
        #         for tc in msg.tool_calls
        #     ]

        assistant_msg = msg
        messages.append(assistant_msg)

        if choice.finish_reason != "tool_calls":
            return

        results = []
        for tc in msg.tool_calls:
            if tc.type != "function":
                continue
            func = tc.function
            handler = TOOL_HANDLERS.get(func.name)
            try:
                args = json.loads(func.arguments)
            except json.JSONDecodeError:
                args = {}
            output = (handler(**args) if handler else f"Unknown tool: {func.name}")

            print(f"> {func.name}:")
            print(output[:200])
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                }
            )
        messages.extend(results)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the final assistant text response
        last_msg = history[-1]
        # # If the last message is tool results, find the preceding assistant message
        # if last_msg.get("role") == "tool":
        #     for msg in reversed(history[:-1]):
        #         if msg.get("role") == "assistant" and msg.get("content"):
        #             print(msg["content"])
        #             break
        # elif last_msg.get("role") == "assistant" and last_msg.get("content"):
        #     print(last_msg["content"])
        print(last_msg.content)
        print()
