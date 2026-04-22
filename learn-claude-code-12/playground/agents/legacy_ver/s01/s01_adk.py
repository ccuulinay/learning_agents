#!/usr/bin/env python3
"""
s01_adk.py - The Agent Loop (Google ADK version)
The entire secret of an AI coding agent in one pattern:
    while state != "completed":
        response = agent.run()
        execute tools
        append results
    +----------+      +-------+      +---------+
    |   User   | ---> | Agent | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)
This is the core loop: feed tool results back to the agent
until the agent decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess
import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from dotenv import load_dotenv
load_dotenv(override=True)

# Define litellm
ds_llm = LiteLlm(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
MODEL = ds_llm
APP_NAME = "s01_adk_agent"
USER_ID = "user"
SESSION_ID = "session"


def run_bash(command: str) -> str:
    """Run a shell command and return the output."""
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


# Create the agent with the bash tool
root_agent = Agent(
    model=MODEL,
    name="coding_agent",
    description="A coding agent that executes bash commands",
    instruction=f"You are a coding agent at {os.getcwd()}. Use the bash tool to solve tasks. Act, don't explain.",
    tools=[run_bash],
)


async def agent_loop(user_query: str):
    """Run the agent loop with the user query."""
    # Set up session service and runner
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Create the user message using proper ADK types
    content = types.Content(role='user', parts=[types.Part(text=user_query)])

    # Run the agent - run_async returns an async generator
    events = runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    )

    # Collect and display results
    final_response = ""
    async for event in events:
        # Print tool calls
        function_calls = event.get_function_calls()
        if function_calls:
            for call in function_calls:
                args = call.args if hasattr(call, 'args') else {}
                cmd = args.get('command', '') if isinstance(args, dict) else str(args)
                print(f"\033[33m$ {cmd}\033[0m")

        # Print tool responses
        function_responses = event.get_function_responses()
        if function_responses:
            for resp in function_responses:
                output = resp.response if hasattr(resp, 'response') else str(resp)
                print(str(output)[:200])

        # Collect final text response
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_response = part.text

    return final_response


async def main():
    """Main interactive loop."""
    print(f"Agent: {root_agent.name}")
    print(f"Model: {MODEL}")
    print("Type 'q', 'exit', or press Ctrl+C to quit\n")

    while True:
        try:
            query = input("\033[36ms01_adk >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        response = await agent_loop(query)
        if response:
            print(response)
        print()


if __name__ == "__main__":
    asyncio.run(main())
