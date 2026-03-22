#!/usr/bin/env python3
"""
s02_adk.py - Tools (Google ADK version)
The agent loop from s01 didn't change. We just added tools.
ADK key insight: tools are plain Python functions — no schema dicts needed.
    +----------+      +-------+      +------------------+
    |   User   | ---> | Agent | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash           |
                          ^          |   read_file      |
                          |          |   write_file     |
                          +----------+   edit_file      |
                          tool_result| }                |
                                     +------------------+
Key insight: "The loop didn't change at all. I just added tools."
ADK advantage: Python functions ARE the tool definitions. No JSON schema boilerplate.
"""

import os
import asyncio
import logging
import subprocess
from pathlib import Path

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()

ds_llm = LiteLlm(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

MODEL = ds_llm
APP_NAME = "s02_adk_agent"
USER_ID = "user"
SESSION_ID = "session"


# --- Tool helpers ---

def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path traversal detected: {p}")
    return path


# --- Tools: plain Python functions, ADK infers schemas from type hints + docstrings ---

def bash(command: str) -> str:
    """Run a shell command and return stdout+stderr. Dangerous commands are blocked."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=os.getcwd(),
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        print(f"\033[33m$ {command}\033[0m")
        result = out[:50000] if out else "(no output)"
        print(result[:200])
        return result
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def read_file(path: str, limit: int = 0) -> str:
    """Read a file's content. Use limit > 0 to read only the first N lines."""
    try:
        text = _safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        logging.exception(e)
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        logging.exception(e)
        return f"Error: {e}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace the first occurrence of old_text with new_text in a file."""
    try:
        fp = _safe_path(path)
        if fp.exists():
            current = fp.read_text()
            if old_text not in current:
                return f"Error: Text to replace not found in {path}."
            new_content = current.replace(old_text, new_text, 1)
        else:
            new_content = new_text
            fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(new_content)
        return f"Edited {path} ({len(new_text)} chars replaced)"
    except Exception as e:
        logging.exception(e)
        return f"Error: {e}"


# --- Agent ---

root_agent = Agent(
    model=MODEL,
    name="coding_agent",
    description="A coding agent with file and shell tools",
    instruction=f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain.",
    tools=[bash, read_file, write_file, edit_file],
)


# --- Runner (shared across turns for persistent conversation) ---

session_service = InMemorySessionService()


async def init_session():
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )


runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


async def agent_turn(user_query: str) -> str:
    """Send one user message and return the final agent response."""
    content = types.Content(role="user", parts=[types.Part(text=user_query)])

    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_response = part.text

    return final_response


async def main():
    await init_session()
    print(f"Agent : {root_agent.name}")
    print(f"Model : {MODEL}")
    print(f"Workdir: {WORKDIR}")
    print("Type 'q', 'exit', or Ctrl+C to quit\n")

    while True:
        try:
            query = input("\033[36ms02_adk >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        response = await agent_turn(query)
        if response:
            print(response)
        print()


if __name__ == "__main__":
    asyncio.run(main())
