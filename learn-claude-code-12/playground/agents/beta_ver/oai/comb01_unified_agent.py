#!/usr/bin/env python3
# Unified coding agent: tool use + todo + subagent + skill loading + context compact.
"""
s07_unified_agent.py - OpenAI SDK
Composes the five harness chapters into one agent loop:
1. Tool use      -> bash, read_file, write_file, edit_file
2. Todo          -> session planning with reminder nudges
3. Subagent      -> task tool spawns isolated child with fresh context
4. Skill loading -> on-demand knowledge via load_skill
5. Context compact -> micro-compaction, persistence, and history summarization
"""

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
WORKDIR = Path.cwd()

client = OpenAI(
    base_url=os.getenv("MOONSHOT_BASE_URL"),
    api_key=os.getenv("MOONSHOT_API_KEY"),
)

MODEL = os.getenv("MOONSHOT_LATEST_MODEL")

PLAN_REMINDER_INTERVAL = 3
CONTEXT_LIMIT = 50000
KEEP_RECENT_TOOL_RESULTS = 3
PERSIST_THRESHOLD = 30000
PREVIEW_CHARS = 2000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
SKILLS_DIR = WORKDIR / "playground" / "skills"


# ---------------------------------------------------------------------------
# 1. Todo Manager
# ---------------------------------------------------------------------------
@dataclass
class PlanItem:
    content: str
    status: str = "pending"
    active_form: str = ""


@dataclass
class PlanningState:
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0


class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list) -> str:
        if len(items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")
        normalized = []
        in_progress_count = 0
        for index, raw in enumerate(items):
            content = str(raw.get("content", "")).strip()
            status = str(raw.get("status", "pending")).lower()
            active_form = str(raw.get("activeForm", "")).strip()
            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            normalized.append(PlanItem(content=content, status=status, active_form=active_form))
        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")
        self.state.items = normalized
        self.state.rounds_since_update = 0
        return self.render()

    def note_round_without_update(self) -> None:
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>"

    def render(self) -> str:
        if not self.state.items:
            return "No session plan yet."
        lines = []
        for item in self.state.items:
            marker = {"pending": "[ ]", "in_progress": ">", "completed": "[x]"}[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)
        completed = sum(1 for i in self.state.items if i.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# ---------------------------------------------------------------------------
# 2. Skill Registry
# ---------------------------------------------------------------------------
@dataclass
class SkillManifest:
    name: str
    description: str
    path: Path


@dataclass
class SkillDocument:
    manifest: SkillManifest
    body: str


class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.documents: dict[str, SkillDocument] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            meta, body = self._parse_frontmatter(path.read_text())
            name = meta.get("name", path.parent.name)
            description = meta.get("description", "No description")
            manifest = SkillManifest(name=name, description=description, path=path)
            self.documents[name] = SkillDocument(manifest=manifest, body=body.strip())

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        return meta, match.group(2)

    def describe_available(self) -> str:
        if not self.documents:
            return "(no skills available)"
        lines = [f"- {doc.manifest.name}: {doc.manifest.description}" for doc in self.documents.values()]
        return "\n".join(sorted(lines))

    def load_full_text(self, name: str) -> str:
        doc = self.documents.get(name)
        if not doc:
            known = ", ".join(sorted(self.documents)) or "(none)"
            return f"Error: Unknown skill '{name}'. Available skills: {known}"
        return f'<skill name="{doc.manifest.name}">\n{doc.body}\n</skill>'


SKILL_REGISTRY = SkillRegistry(SKILLS_DIR)


# ---------------------------------------------------------------------------
# 3. Context Compaction State & Helpers
# ---------------------------------------------------------------------------
@dataclass
class CompactState:
    has_compacted: bool = False
    last_summary: str = ""
    recent_files: list[str] = field(default_factory=list)


def estimate_context_size(messages: list) -> int:
    return len(str(messages))


def track_recent_file(state: CompactState, path: str) -> None:
    if path in state.recent_files:
        state.recent_files.remove(path)
    state.recent_files.append(path)
    if len(state.recent_files) > 5:
        state.recent_files[:] = state.recent_files[-5:]


def persist_large_output(tool_use_id: str, output: str) -> str:
    if len(output) <= PERSIST_THRESHOLD:
        return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not stored_path.exists():
        stored_path.write_text(output)
    preview = output[:PREVIEW_CHARS]
    rel_path = stored_path.relative_to(WORKDIR)
    return (
        "<persisted-output>\n"
        f"Full output saved to: {rel_path}\n"
        "Preview:\n"
        f"{preview}\n"
        "</persisted-output>"
    )


def collect_tool_result_blocks(messages: list) -> list[tuple[int, dict]]:
    return [(i, m) for i, m in enumerate(messages) if m.get("role") == "tool"]


def micro_compact(messages: list) -> list:
    tool_results = collect_tool_result_blocks(messages)
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
        return messages
    for _, block in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        content = block.get("content", "")
        if not isinstance(content, str) or len(content) <= 120:
            continue
        block["content"] = "[Earlier tool result compacted. Re-run the tool if you need full detail.]"
    return messages


def write_transcript(messages: list) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    return path


def summarize_history(messages: list) -> str:
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve:\n"
        "1. The current goal\n"
        "2. Important findings and decisions\n"
        "3. Files read or changed\n"
        "4. Remaining work\n"
        "5. User constraints and preferences\n"
        "Be compact but concrete.\n\n"
        f"{conversation}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def compact_history(messages: list, state: CompactState, focus: str | None = None) -> list:
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)
    if focus:
        summary += f"\n\nFocus to preserve next: {focus}"
    if state.recent_files:
        recent_lines = "\n".join(f"- {p}" for p in state.recent_files)
        summary += f"\n\nRecent files to reopen if needed:\n{recent_lines}"
    state.has_compacted = True
    state.last_summary = summary
    return [
        {
            "role": "user",
            "content": (
                "This conversation was compacted so the agent can continue working.\n\n"
                f"{summary}"
            ),
        }
    ]


# ---------------------------------------------------------------------------
# 4. Base Tool Implementations
# ---------------------------------------------------------------------------
def safe_path(path_str: str) -> Path:
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path


def run_bash(command: str, tool_use_id: str = "") -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    output = (result.stdout + result.stderr).strip() or "(no output)"
    if tool_use_id:
        return persist_large_output(tool_use_id, output)
    return output


def run_read(path: str, tool_use_id: str = "", state: CompactState | None = None, limit: int | None = None) -> str:
    try:
        if state:
            track_recent_file(state, path)
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        output = "\n".join(lines)
        if tool_use_id:
            return persist_large_output(tool_use_id, output)
        return output
    except Exception as exc:
        return f"Error: {exc}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        content = file_path.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        file_path.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"


# Child dispatch: base tools only (no persistence)
CHILD_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], limit=kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

CHILD_TOOLS = [
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
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once.",
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


# ---------------------------------------------------------------------------
# 5. Subagent
# ---------------------------------------------------------------------------
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SUBAGENT_SYSTEM}] + sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        msg = response.choices[0].message
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
            "reasoning_content": getattr(msg, "reasoning_content", None) or "",
        }
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        sub_messages.append(assistant_msg)
        if not msg.tool_calls:
            break
        results = []
        for tc in msg.tool_calls:
            handler = CHILD_HANDLERS.get(tc.function.name)
            try:
                args = json.loads(tc.function.arguments)
                output = handler(**args) if handler else f"Unknown tool: {tc.function.name}"
            except Exception as exc:
                output = f"Error: {exc}"
            results.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)[:50000]})
        sub_messages.extend(results)
    last = sub_messages[-1]
    if last.get("role") == "assistant":
        return last.get("content", "") or "(no summary)"
    return "(no summary)"


# ---------------------------------------------------------------------------
# 6. Parent Tool Schemas
# ---------------------------------------------------------------------------
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use tools to solve tasks. Act, don't explain.
Use the todo tool for multi-step work. Keep exactly one step in_progress.
Use the task tool to delegate exploration or subtasks.
Use load_skill when a task needs specialized instructions before you act.
Use compact if the conversation gets too long.
Skills available:
{SKILL_REGISTRY.describe_available()}
"""

PARENT_TOOLS = CHILD_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "Rewrite the current session plan for multi-step work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                                "activeForm": {"type": "string", "description": "Optional present-continuous label."},
                            },
                            "required": ["content", "status"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {"type": "string", "description": "Short description of the task"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load the full body of a named skill into the current context.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compact",
            "description": "Summarize earlier conversation so work can continue in a smaller context.",
            "parameters": {
                "type": "object",
                "properties": {"focus": {"type": "string"}},
            },
        },
    },
]


def extract_text(message: dict) -> str:
    content = message.get("content", "")
    return content.strip() if isinstance(content, str) else ""


def execute_tool(name: str, args: dict, tool_id: str, state: CompactState) -> str:
    if name == "bash":
        return run_bash(args["command"], tool_id)
    if name == "read_file":
        return run_read(args["path"], tool_id, state, args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "todo":
        return TODO.update(args["items"])
    if name == "load_skill":
        return SKILL_REGISTRY.load_full_text(args["name"])
    if name == "compact":
        return "Compacting conversation..."
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# 7. Agent Loop
# ---------------------------------------------------------------------------
def agent_loop(messages: list, state: CompactState) -> None:
    while True:
        # --- compaction layer ---
        messages[:] = micro_compact(messages)
        if estimate_context_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages, state)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=PARENT_TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )
        message = response.choices[0].message

        assistant_msg = {
            "role": "assistant",
            "content": message.content or "",
            "reasoning_content": getattr(message, "reasoning_content", None) or "",
        }
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            return

        results = []
        manual_compact = False
        compact_focus = None
        used_todo = False

        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            if tc.function.name == "task":
                desc = args.get("description", "subtask")
                prompt = args.get("prompt", "")
                print(f"> task ({desc}): {prompt[:80]}")
                output = run_subagent(prompt)
            else:
                output = execute_tool(tc.function.name, args, tc.id, state)

            print(f"  {str(output)[:200]}")
            results.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)})

            if tc.function.name == "todo":
                used_todo = True
            if tc.function.name == "compact":
                manual_compact = True
                compact_focus = args.get("focus")

        messages.extend(results)

        # --- todo reminder layer ---
        if used_todo:
            TODO.state.rounds_since_update = 0
        else:
            TODO.note_round_without_update()
            reminder = TODO.reminder()
            if reminder:
                messages.append({"role": "user", "content": reminder})

        # --- manual compaction ---
        if manual_compact:
            print("[manual compact]")
            messages[:] = compact_history(messages, state, focus=compact_focus)


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    history = []
    compact_state = CompactState()
    while True:
        try:
            query = input("\033[36magent >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history, compact_state)
        last = history[-1]
        if last.get("role") == "assistant":
            txt = extract_text(last)
            if txt:
                print(txt)
        print()
