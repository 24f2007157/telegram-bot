import datetime
import io
import json
import logging
import math
import os
import re
import statistics

import numpy as np
import openai
import pandas as pd
import requests
import scipy
from dotenv import load_dotenv
from logger import JSONLLogger
from openai import OpenAI
from scipy import stats

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# In-memory storage for multi-turn conversations per Telegram chat_id
conversation_history: dict[int, list[dict]] = {}

SYSTEM_INSTRUCTION = """
You are an expert Data Analyst LLM Agent equipped with Python code execution, URL data fetching, and live web search tools.

GENERAL FACTUAL ACCURACY & VERIFICATION PROTOCOL:
1. STRICT TRUTH & EVIDENTIARY ACCURACY: Ensure 100% factual accuracy across ALL domains (economics, demographics, government datasets like MOSPI/Census/SRS, science, geography, history, public policy, sports, and current events). Never state incorrect or unverified facts.
2. DISTINGUISH SCOPE & TIMELINES: When answering queries, carefully analyze timeframes, status qualifiers (e.g. "former", "current", "defending", "projected", "historical"), and exact metric definitions.
3. SEARCH RESULT CROSS-VERIFICATION: When using web search or URL fetching tools:
   - Extract ground-truth facts directly from authoritative text snippets or dataset tables.
   - Cross-verify entities, names, values, and dates before constructing the final response.
4. ONLINE SEARCH MANDATE: Whenever asked for real-world factual information, statistics, or public data not provided inline in the message:
   - Use `search_web_online` with clean, targeted search keywords.
5. MATHEMATICAL & CALCULATED PRECISION: Use `execute_python_code` for all calculations, percentages, regressions, and data sorting. Never guess numbers.

OUTPUT SCHEMA RULES:
1. Dynamically extract the exact requested JSON shape for the "answer" field from the user's latest prompt:
   - If the prompt asks for: Reply ONLY {"answer": {"winner": "<country>"}, "log_url": "..."}, your "answer" field MUST be a clean string like {"winner": "<country>"}. NEVER put explanatory or error text inside the value field.
   - If the prompt asks for: Reply ONLY {"answer": {"state": "<state name>"}, "log_url": "..."}, your "answer" field MUST be a clean state name string like {"state": "Assam"}.
   - If the prompt asks for: Reply ONLY {"answer": {"values": [<numbers>]}, "log_url": "..."}, your "answer" field MUST be {"values": [<calculated_numbers>]}.
   - If the prompt is a simple greeting or acknowledgment (e.g. "hello", "start"), set "answer" to {"status": "ready"}.

2. You MUST return ONLY a single valid JSON object containing two top-level keys:
   {
     "answer": <dynamically_extracted_and_calculated_answer_shape>,
     "log_url": "PLACEHOLDER_LOG_URL"
   }

3. Do NOT include markdown code blocks (no ```json or ``` wrappers), intro text, or explanation outside the JSON.
"""

# -------------------------------------------------------------------
# TOOL DEFINITIONS FOR OPENAI AGENT
# -------------------------------------------------------------------


def search_web_online(query: str) -> str:
    """Searches the web online for recent information, facts, stats, news, or dataset facts across all domains."""
    try:
        clean_q = re.sub(r"['\"\\]|site:\S+", " ", query)
        clean_q = re.sub(r"\s+", " ", clean_q).strip()
        terms = clean_q.split()
        if len(terms) > 6:
            clean_q = " ".join(terms[:6])

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }

        # 1. DuckDuckGo Instant Answer API for direct factual summary
        try:
            ddg_api = f"https://api.duckduckgo.com/?q={requests.utils.quote(clean_q)}&format=json&no_html=1&skip_disambig=1"
            api_res = requests.get(ddg_api, headers=headers, timeout=8).json()
            abstract = api_res.get("AbstractText", "").strip()
            heading = api_res.get("Heading", "").strip()
            if abstract:
                return f"Direct Answer Summary ({heading}):\n{abstract}"
        except Exception:
            pass

        # 2. DuckDuckGo HTML Search
        search_url = (
            f"https://html.duckduckgo.com/html/?q={requests.utils.quote(clean_q)}"
        )
        res = requests.get(search_url, headers=headers, timeout=10)

        if res.status_code == 200:
            snippets = re.findall(
                r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', res.text, re.DOTALL
            )
            clean_snippets = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:6]]
            if clean_snippets:
                return "Web Search Results:\n" + "\n".join(
                    [f"- {s}" for s in clean_snippets]
                )

        # 3. Wikipedia API Search fallback
        wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(clean_q)}&format=json"
        w_res = requests.get(wiki_url, headers=headers, timeout=10).json()
        search_hits = w_res.get("query", {}).get("search", [])
        if search_hits:
            return "Wikipedia Search Results:\n" + "\n".join(
                [
                    f"- {h['title']}: {re.sub(r'<[^>]+>', '', h['snippet'])}"
                    for h in search_hits[:4]
                ]
            )

        return "No online search results found. (Fallback to internal knowledge)."
    except Exception as e:
        return f"Web search error: {e}"


def execute_python_code(code: str) -> str:
    """Executes Python code for data analysis, calculations, pandas, numpy, scipy, and math operations."""
    try:
        exec_globals = {
            "pd": pd,
            "np": np,
            "scipy": scipy,
            "stats": stats,
            "math": math,
            "statistics": statistics,
            "datetime": datetime,
            "requests": requests,
            "io": io,
            "json": json,
            "re": re,
        }
        exec_locals = {}

        stdout_capture = io.StringIO()
        import sys

        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        try:
            exec(code, exec_globals, exec_locals)
        finally:
            sys.stdout = old_stdout

        printed_output = stdout_capture.getvalue().strip()

        if printed_output:
            return printed_output
        elif exec_locals:
            return json.dumps(
                {k: str(v) for k, v in exec_locals.items() if not k.startswith("_")}
            )
        else:
            return "Code executed successfully (no output)."
    except Exception as e:
        return f"Python Execution Error: {e}"


def fetch_url_content(url: str) -> str:
    """Fetches CSV, JSON, or text data from a public URL using browser User-Agent headers."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        text_content = response.text

        if "<html" in text_content.lower():
            clean_text = re.sub(
                r"<(script|style).*?>.*.*?/\1>",
                "",
                text_content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            clean_text = re.sub(r"<[^>]+>", " ", clean_text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            text_content = clean_text

        if len(text_content) > 10000:
            return text_content[:10000] + "\n...[truncated]"
        return text_content
    except Exception as e:
        return f"Fetch Failed for {url}: {e}."


# Tool Schemas for OpenAI Function Calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_web_online",
            "description": "Searches the live web online for recent events, facts, public dataset facts, demographics, or current information not in training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query keywords to lookup online.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "Executes Python code to perform data analysis, aggregations, pandas/numpy/scipy operations, statistics, or math calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to execute. Use print() to output results.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url_content",
            "description": "Downloads text, CSV, or JSON dataset content from a public URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The public HTTP/HTTPS URL to fetch data from.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


def get_openai_client() -> OpenAI:
    """Initializes and returns the OpenAI API client."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")
    return OpenAI(api_key=OPENAI_API_KEY)


def clean_json_response(raw_text: str) -> dict:
    """Cleans raw text output from the LLM to extract valid JSON."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return {"answer": {"result": text or "Processed"}}


def process_question(chat_id: int, message_text: str, log_base_url: str = None) -> str:
    """
    Main Agent Entrypoint using OpenAI API with Function Calling Tools (Web Search + Python + URL Fetcher).
    """
    run_logger = JSONLLogger(log_base_url=log_base_url)
    run_logger.log("start", {"chat_id": chat_id, "message": message_text})

    # Track multi-turn history
    if chat_id not in conversation_history:
        conversation_history[chat_id] = []

    conversation_history[chat_id].append({"role": "user", "content": message_text})
    run_logger.log(
        "history_context", {"history_length": len(conversation_history[chat_id])}
    )

    # Build messages payload
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

    for msg in conversation_history[chat_id][:-1]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    latest_user_text = conversation_history[chat_id][-1]["content"]
    messages.append(
        {
            "role": "user",
            "content": f"=== LATEST USER MESSAGE TO ANSWER NOW ===\n{latest_user_text}",
        }
    )

    try:
        client = get_openai_client()
        run_logger.log(
            "llm_call_initiated",
            {
                "model": MODEL_NAME,
                "tools_enabled": [
                    "search_web_online",
                    "execute_python_code",
                    "fetch_url_content",
                ],
            },
        )

        raw_llm_output = ""

        # Tool execution loop (up to 4 function call iterations)
        for loop_count in range(4):
            completion_kwargs = {
                "model": MODEL_NAME,
                "messages": messages,
                "tools": TOOLS_SCHEMA,
                "tool_choice": "auto",
            }

            try:
                response = client.chat.completions.create(**completion_kwargs)
            except openai.BadRequestError as e:
                if "tools" in str(e) or "unsupported_value" in str(e):
                    logger.info(
                        "Retrying without tool schemas for model compatibility..."
                    )
                    completion_kwargs.pop("tools", None)
                    completion_kwargs.pop("tool_choice", None)
                    response = client.chat.completions.create(**completion_kwargs)
                else:
                    raise e

            response_message = response.choices[0].message
            tool_calls = getattr(response_message, "tool_calls", None)

            if not tool_calls:
                raw_llm_output = (
                    response_message.content.strip() if response_message.content else ""
                )
                break

            # Append assistant message with tool calls to context
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                run_logger.log(
                    "tool_call_executed", {"name": function_name, "args": function_args}
                )

                if function_name == "search_web_online":
                    tool_result = search_web_online(function_args.get("query", ""))
                elif function_name == "execute_python_code":
                    tool_result = execute_python_code(function_args.get("code", ""))
                elif function_name == "fetch_url_content":
                    tool_result = fetch_url_content(function_args.get("url", ""))
                else:
                    tool_result = f"Unknown function {function_name}"

                run_logger.log(
                    "tool_call_result",
                    {"name": function_name, "result": str(tool_result)[:500]},
                )

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result),
                    }
                )

        # Final synthesis step if raw_llm_output is empty
        if not raw_llm_output:
            messages.append(
                {
                    "role": "user",
                    "content": "Synthesize all collected search/tool results above. Ensure 100% general factual accuracy across all domains (distinguishing past vs current states, timelines, and exact metric definitions). Return ONLY the final JSON object matching the requested schema.",
                }
            )
            final_res = client.chat.completions.create(
                model=MODEL_NAME, messages=messages
            )
            raw_llm_output = (
                final_res.choices[0].message.content.strip()
                if final_res.choices[0].message.content
                else "{}"
            )

        run_logger.log("llm_response_received", {"raw_output": raw_llm_output})

        # Parse JSON output
        parsed_data = clean_json_response(raw_llm_output)

        # Inject actual public log URL
        actual_log_url = run_logger.get_log_url()
        parsed_data["log_url"] = actual_log_url

        # Store assistant answer into chat history
        conversation_history[chat_id].append(
            {"role": "assistant", "content": json.dumps(parsed_data.get("answer", {}))}
        )

        final_json_string = json.dumps(parsed_data, separators=(",", ":"))
        run_logger.log(
            "completion", {"final_output": parsed_data, "log_url": actual_log_url}
        )

        return final_json_string

    except Exception as e:
        logger.error(f"Error in OpenAI agent processing: {e}", exc_info=True)
        actual_log_url = run_logger.get_log_url()
        run_logger.log("error", {"error_message": str(e)})

        fallback_response = {"answer": {"error": str(e)}, "log_url": actual_log_url}
        return json.dumps(fallback_response, separators=(",", ":"))


def clear_history(chat_id: int):
    """Resets conversation history for a given chat_id."""
    conversation_history.pop(chat_id, None)
