"""
Agent loop for the Climate / Air-Quality assignment.

Connects to mcp_server.py over stdio and asks Gemini to:
  For each Indian city:
    1. fetch_aqi_now(city)
    2. fetch_aqi_24h(city)
    3. fetch_weather_now(city)
    4. save_city_report(...)              ← merges values from steps 1-3
  Then:
    5. show_dashboard(focus_city=<best>)
    6. FINAL_ANSWER with the cleanest city + its peak pollution hour

Run:
    cd "/Users/bavyabalakrishnan/EAG V3/APP4/assignment_eco"
    ~/.local/bin/uv run --with prefab-ui --with fastmcp \
        --with requests --with python-dotenv --with google-genai \
        --with "mcp[cli]" python agent.py

Then, in another terminal:
    cd "/Users/bavyabalakrishnan/EAG V3/APP4/assignment_eco"
    ~/.local/bin/uv run --with prefab-ui --with fastmcp --with requests \
        --with python-dotenv fastmcp dev apps mcp_server.py
And click `show_dashboard` to see the live UI.
"""

import asyncio
import os
from concurrent.futures import TimeoutError
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
# Fall back to ../assignment/.env so we don't need to copy GEMINI_API_KEY.
load_dotenv(HERE.parent / "assignment" / ".env")


# Preview model. May return 503 UNAVAILABLE under load — the retry loop
# below handles that with exponential backoff.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
MAX_ITERATIONS = 60
# 20s pause before each LLM call -> ~3 req/min, well under the
# gemini-2.5-flash-lite 15 RPM free-tier cap even with retries.
LLM_SLEEP_SECONDS = 20
LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 8
LLM_BACKOFF_CAP = 60

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def generate_with_timeout(prompt: str, timeout: int = LLM_TIMEOUT,
                                model: str = MODEL):
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: client.models.generate_content(model=model, contents=prompt),
        ),
        timeout=timeout,
    )


def describe_tools(tools) -> str:
    lines = []
    for i, t in enumerate(tools, 1):
        props = (t.inputSchema or {}).get("properties", {})
        params = ", ".join(
            f"{n}: {p.get('type', '?')}" for n, p in props.items()
        ) or "no params"
        lines.append(f"{i}. {t.name}({params}) — {t.description or ''}")
    return "\n".join(lines)


def coerce(value: str, schema_type: str):
    if schema_type == "integer":
        try: return int(value)
        except ValueError: return int(float(value))
    if schema_type == "number":
        return float(value)
    if schema_type == "array":
        # Agent passes lists as Python literals, e.g. "[10.0, 12.5, ...]".
        return eval(value)
    if schema_type == "boolean":
        return value.lower() in ("true", "1", "yes")
    return value


async def main():
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "--with", "fastmcp",
            "--with", "prefab-ui",
            "--with", "requests",
            "--with", "python-dotenv",
            "python", str(HERE / "mcp_server.py"),
        ],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to ClimateTrackerServer")

            tools = (await session.list_tools()).tools
            tools_desc = describe_tools(tools)
            print(f"Loaded {len(tools)} tools:\n{tools_desc}\n")

            system_prompt = f"""You are a climate / air-quality agent for Indian
cities. You solve tasks by calling MCP tools ONE AT A TIME and using each
result to choose your next call.

Available tools:
{tools_desc}

Respond with EXACTLY ONE line, in one of these two formats:
  FUNCTION_CALL: tool_name|arg1|arg2|...
  FINAL_ANSWER: <short answer>

Rules:
- Pass arguments in the exact order of the tool's parameters (left to right).
- All tools return JSON strings. Parse the values you need from the previous
  result before constructing the next FUNCTION_CALL.
- For each city, prefer the FAST batch path:
    build_city_report|<city>
  This single call fetches AQI + 24h history + weather IN PARALLEL and
  saves the merged report to eco_log.json. One MCP call per city.
- Only fall back to the 4-step sequence (fetch_aqi_now, fetch_aqi_24h,
  fetch_weather_now, save_city_report) if build_city_report errors.
- After ALL cities are saved, call show_dashboard ONCE (no arguments —
  every tab now shows every city).
- Then emit FINAL_ANSWER summarising: cleanest city + worst pollution
  hour observed across all cities (e.g. "FINAL_ANSWER: Bangalore is
  cleanest (AQI 38). Worst hour: Delhi at 09:00.").
- Do not invent tools that are not listed above.
- Never repeat a successful FUNCTION_CALL; always advance to the next step.
"""

            DEFAULT_TASK = (
                "Build a climate & air-quality report for these 3 Indian "
                "cities: Delhi, Mumbai, Bangalore. Use build_city_report "
                "once per city (fast parallel path). After all 3 are saved, "
                "call show_dashboard once, then FINAL_ANSWER summarising "
                "the cleanest city and the worst pollution hour observed."
            )

            async def run_task(task: str) -> None:
                """Run one user-provided task to completion (FINAL_ANSWER or
                MAX_ITERATIONS). Reuses the live MCP session, so changes to
                eco_log.json persist across tasks just like a real REPL."""
                history: list[str] = []
                for iteration in range(1, MAX_ITERATIONS + 1):
                    print(f"\n--- Iteration {iteration} ---")

                    context = "\n".join(history) if history else "(no prior steps)"
                    prompt = (
                        f"{system_prompt}\n"
                        f"Task: {task}\n\n"
                        f"Previous steps:\n{context}\n\n"
                        f"What is your next single action?"
                    )

                    await asyncio.sleep(LLM_SLEEP_SECONDS)

                    response = None
                    quota_exhausted = False
                    for attempt in range(1, LLM_MAX_RETRIES + 1):
                        try:
                            response = await generate_with_timeout(prompt)
                            break
                        except (TimeoutError, asyncio.TimeoutError):
                            print(f"LLM timed out (attempt {attempt}/{LLM_MAX_RETRIES}, model={MODEL})")
                        except Exception as e:
                            msg = str(e)
                            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                                print(f"LLM daily quota exhausted (model={MODEL}). "
                                      "Stopping immediately.")
                                quota_exhausted = True
                                break
                            transient = any(s in msg for s in ("503", "UNAVAILABLE", "500"))
                            print(f"LLM error (attempt {attempt}/{LLM_MAX_RETRIES}, model={MODEL}): {msg[:200]}")
                            if not transient:
                                break
                        backoff = min(2 ** attempt, LLM_BACKOFF_CAP)
                        print(f"  retrying in {backoff}s…")
                        await asyncio.sleep(backoff)

                    if response is None:
                        if quota_exhausted:
                            print("Stopped: daily Gemini quota exhausted.")
                        else:
                            print("LLM unavailable after retries — stopping this task.")
                        return

                    text = (response.text or "").strip().splitlines()[0].strip()
                    print(f"LLM: {text}")

                    if text.startswith("FINAL_ANSWER:"):
                        print("\n=== Task done ===")
                        print(text)
                        return

                    if not text.startswith("FUNCTION_CALL:"):
                        print("Unexpected response format — ending this task.")
                        return

                    _, call = text.split(":", 1)
                    parts = [p.strip() for p in call.split("|")]
                    func_name, raw_args = parts[0], parts[1:]

                    tool = next((t for t in tools if t.name == func_name), None)
                    if tool is None:
                        msg = f"Unknown tool {func_name!r}"
                        print(msg)
                        history.append(f"Iteration {iteration}: {msg}")
                        continue

                    props = (tool.inputSchema or {}).get("properties", {})
                    arguments = {}
                    for (name, info), val in zip(props.items(), raw_args):
                        try:
                            arguments[name] = coerce(val, info.get("type", "string"))
                        except Exception as e:
                            arguments[name] = val
                            print(f"  (coerce warning for {name}={val!r}: {e})")

                    print(f"→ {func_name}({list(arguments)[:6]}{'...' if len(arguments) > 6 else ''})")
                    try:
                        result = await session.call_tool(func_name, arguments=arguments)
                        payload = (
                            result.content[0].text
                            if result.content and hasattr(result.content[0], "text")
                            else str(result)
                        )
                    except Exception as e:
                        payload = f"ERROR: {e}"

                    short = payload if len(payload) < 1200 else payload[:1200] + "…"
                    print(f"← {short}")
                    history.append(
                        f"Iteration {iteration}: called {func_name} → {short}"
                    )
                else:
                    print("\nReached MAX_ITERATIONS without FINAL_ANSWER.")

            # ---- REPL: keep prompting for tasks until the user quits ----
            print("\n" + "=" * 60)
            print("Eco-tracker REPL — type a task and watch the agent run it.")
            print("Each task reuses the same MCP session, so eco_log.json")
            print("persists across tasks (Create / Update / Delete really stick).")
            print("Commands: 'quit' / 'exit' to leave, blank line = default task.")
            print("=" * 60)
            print(f"\nDefault task: {DEFAULT_TASK}\n")

            task_num = 0
            while True:
                try:
                    user_task = input("\nTask> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break
                if user_task.lower() in {"quit", "exit", "q"}:
                    print("Exiting.")
                    break
                task = user_task or DEFAULT_TASK
                task_num += 1
                print(f"\n========== Task {task_num} ==========")
                print(f"Using task: {task}\n")
                try:
                    await run_task(task)
                except Exception as e:
                    print(f"\nTask failed with: {e}")
                print("\n(open the dashboard at `fastmcp dev apps mcp_server.py` "
                      "or refresh the browser to see updated data)")


if __name__ == "__main__":
    asyncio.run(main())
