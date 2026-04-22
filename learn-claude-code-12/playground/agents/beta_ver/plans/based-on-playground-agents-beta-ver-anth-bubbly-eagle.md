# Plan: Create OpenAI version of beta s01_agent_loop.py

## Context
The user wants an OpenAI SDK version of `playground/agents/beta_ver/anth/s01_agent_loop.py`. The beta version introduces a structured `LoopState` dataclass and separates the loop into `run_one_turn()` and `agent_loop()` functions. There is already an empty file at `playground/agents/beta_ver/oai/s01_agent_loop.py` and a legacy OpenAI reference at `playground/agents/legacy_ver/s01/s01_oai.py`.

## Approach
Port the beta anthropic version to OpenAI while preserving its structural improvements (`LoopState`, `run_one_turn`, `agent_loop` separation). Use the legacy OpenAI file for SDK-specific patterns.

### Changes to make in `playground/agents/beta_ver/oai/s01_agent_loop.py`:

1. **Replace Anthropic SDK import with OpenAI**
   - `from openai import OpenAI`
   - `cli = OpenAI(base_url=..., api_key=...)`
   - Keep `load_dotenv` and env var handling.

2. **Convert tool definition format**
   - Anthropic: `{name, description, input_schema}`
   - OpenAI: `{"type": "function", "function": {name, description, parameters}}`

3. **Convert API call in `run_one_turn()`**
   - Use `cli.chat.completions.create(model=MODEL, messages=..., tools=TOOLS, max_tokens=8000)`
   - Prepend system message: `[{"role": "system", "content": SYSTEM}] + state.messages`
   - Store `assistant_message = response.choices[0].message`

4. **Adapt tool call detection and execution**
   - Check `assistant_message.tool_calls` instead of `response.stop_reason == "tool_use"`
   - Parse arguments with `json.loads(tool_call.function.arguments)`
   - Build tool results as `{"role": "tool", "tool_call_id": tool_call.id, "content": output}`

5. **Adapt message appending**
   - Append `assistant_message` object directly to history
   - Append tool results as dicts with `role: "tool"`

6. **Adapt text extraction**
   - OpenAI message objects have `.content` attribute directly; simplify `extract_text()` to handle the final assistant message object.

7. **Preserve all other beta features**
   - `LoopState` dataclass with `turn_count` and `transition_reason`
   - `run_bash` with dangerous command filtering
   - Readline UTF-8 fixes
   - Interactive REPL loop in `__main__`

## Verification
Run the script and issue a simple bash query (e.g., "list files" or "pwd") to verify the loop executes tool calls and returns results correctly.
