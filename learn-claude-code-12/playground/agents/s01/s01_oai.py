#!/usr/bin/env python3
"""
s01_oai.py - The Agent Loop (OpenAI version)
The entire secret of an AI coding agent in one pattern:
    while stop_reason == "tool_calls":
        response = LLM(messages, tools)
        execute tools
        append results
    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)
This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)


cli = OpenAI(
    base_url=os.getenv("MOONSHOT_BASE_URL"),
    api_key=os.getenv("MOONSHOT_API_KEY")
)

MODEL = os.getenv("MOONSHOT_LATEST_MODEL")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."
TOOLS = [{
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
}]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def agent_loop(user_msg: list):
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
            command = args.get("command", "")
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command)
            print(output[:200])

            # Append tool result
            user_msg.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01_oai >> \033[0m")
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
