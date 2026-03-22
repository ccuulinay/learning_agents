#!/usr/bin/env python3
"""
s03_todo_write.py - TodoWrite
The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.
    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>
Key insight: "The agent can track its own progress -- and I can see it."
"""
import os
import logging
import subprocess
from pathlib import Path
from numpy import block
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

# -- TodoManager: structured state the LLM writes to --
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[
                item["status"]
            ]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()

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
    "todo": lambda **kw: TODO.update(kw["items"]),
}

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
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's content. Use 'limit' to read only first N lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "description": "Max lines to read"}
                },
                "required": ["path"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text", "new_text"],
            },
        }

    },
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "Update task list. Track progress on multi-step tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["id", "text", "status"],
                        }
                    }
                },
                "required": ["items"],
            },
        }
    }
]

def agent_loop(user_msg: list):
    rounds_since_todo = 0
    while True:
        response = cli.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + user_msg,
            tools=TOOLS,
            max_tokens=8000,
        )

        # Append assistant turn
        assistant_message = response.choices[0].message
        user_msg.append(assistant_message)

        # If the model didn't call a tool, we're done
        if not assistant_message.tool_calls:
            return

        # Execute each tool call, collect results
        for tool_call in assistant_message.tool_calls:
            import json
            args = json.loads(tool_call.function.arguments)
            handler = TOOL_HANDLERS.get(tool_call.function.name)
            output = handler(**args) if handler else f"Error: No handler for {block.name}, unknown tool."
            print(f"\033[33m$ {tool_call.function.name}({args})\033[0m")
            print(output[:200])
            # Append tool result
            user_msg.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })
            if tool_call.function.name == "todo":
                rounds_since_todo = 0
            else:
                rounds_since_todo += 1
                
            if rounds_since_todo >= 3:
                user_msg.append({
                    "role": "assistant",
                    "content": "<reminder>Update your todos.</reminder>",
                })
            


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03_oai >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]
        if hasattr(response_content, "content") and response_content.content:
            print(response_content.content)
        print()