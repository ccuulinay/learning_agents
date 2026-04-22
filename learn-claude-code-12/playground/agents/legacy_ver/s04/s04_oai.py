#!/usr/bin/env python3
"""
s04_subagent_oai.py - Subagents (OpenAI version)

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
"""

import os
import logging
import subprocess
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()

cli = OpenAI(
    base_url=os.getenv("MOONSHOT_BASE_URL"),
    api_key=os.getenv("MOONSHOT_API_KEY")
)

MODEL = os.getenv("MOONSHOT_LATEST_MODEL")

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""

SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."

# Define tool handlers with name
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()

    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path traversal detected: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
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
            lines = lines[:limit] + [f"... ({len(lines)-limit} more lines)"]
        # Truncate to 50k chars to avoid token limits
        return "\n".join(lines)[:50000]
    except Exception as e:
        logging.exception(e)
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        logging.exception(e)
        return f"Error: {e}"


def run_edit(path: str, ori_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        if fp.exists():
            current = fp.read_text()
            if ori_text not in current:
                return "Error: Text to replace not found in {path}."
            new_content = current.replace(ori_text, new_text, 1)
        else:
            new_content = new_text
            fp.parent.mkdir(parents=True, exist_ok=True)

        fp.write_text(new_content)
        return f"Edited {path} with new content ({len(new_text)} chars)"
    except Exception as e:
        logging.exception(e)
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# Child gets all base tools except task (no recursive spawning)
CHILD_TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
]

# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context
    for _ in range(30):  # safety limit
        response = cli.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SUBAGENT_SYSTEM}] + sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        message = response.choices[0].message
        sub_messages.append(message)
        if not message.tool_calls:
            break
        # results = []
        for tool_call in message.tool_calls:
            handler = TOOL_HANDLERS.get(tool_call.function.name)
            import json
            args = json.loads(tool_call.function.arguments)
            output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
            # results.append({
            #     "role": "tool",
            #     "tool_call_id": tool_call.id,
            #     "content": str(output)[:50000]
            # })
            sub_messages.append({"role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output)[:50000]})
    # Only the final text returns to the parent -- child context is discarded
    return response.choices[0].message.content or "(no summary)"

# -- Parent tools: base tools + task dispatcher --
PARENT_TOOLS = CHILD_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "Short description of the task",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]

def agent_loop(messages: list):
    while True:
        response = cli.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=PARENT_TOOLS,
            max_tokens=8000,
        )
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            return
        # results = []
        for tool_call in message.tool_calls:
            if tool_call.function.name == "task":
                import json
                args = json.loads(tool_call.function.arguments)
                desc = args.get("description", "subtask")
                print(f"> task ({desc}): {args['prompt'][:80]}")
                output = run_subagent(args["prompt"])
            else:
                handler = TOOL_HANDLERS.get(tool_call.function.name)
                import json
                args = json.loads(tool_call.function.arguments)
                output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
            print(f"  {str(output)[:200]}")
            # results.append({
            #     "role": "tool",
            #     "tool_call_id": tool_call.id,
            #     "content": str(output)
            # })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04_oai >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        message = history[-1]
        if hasattr(message, "content") and message.content:
            print(message.content)
        print()
