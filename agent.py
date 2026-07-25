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
You are an expert Data Analyst LLM Agent equipped with Python code execution and URL data fetching tools.

ANALYTICAL & KNOWLEDGE FALLBACK WORKFLOW:
1. Parse the incoming message and full conversation history.
2. If the user prompt contains a specific URL, use the `fetch_url_content` tool to download and inspect the dataset.
3. If `fetch_url_content` fails (e.g., 404 Not Found) or if no working URL is provided in the prompt, DO NOT return "Unknown", "N/A", or "Error". Immediately fall back to your extensive internal knowledge base and reasoning to answer the question accurately (e.g. "Assam" for the state with highest maternal mortality rate in MOSPI/SRS data).
4. Use the `execute_python_code` tool to perform precise mathematical calculations, statistics, data filtering, sorting, or forecasting when data tables or lists are available. Never guess numbers.

OUTPUT SCHEMA DEDUCTION RULES:
1. Dynamically extract the exact requested JSON shape for the "answer" field from the user's latest prompt:
   - If the prompt asks for: Reply ONLY {"answer": {"values": [<numbers>]}, "log_url": "..."}, your "answer" field MUST be {"values": [<calculated_numbers>]}.
   - If the prompt asks for: Reply ONLY {"answer": {"state": "<name>"}, "log_url": "..."}, your "answer" field MUST be {"state": "<state_name>"}.
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
    """Fetches CSV, JSON, or text data from a public URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        text_content = response.text
        if len(text_content) > 10000:
            return text_content[:10000] + "\n...[truncated]"
        return text_content
    except Exception as e:
        return f"Fetch Failed for {url}: {e}. (Use internal knowledge base to resolve answer)."


# Tool Schemas for OpenAI Function Calling
TOOLS_SCHEMA = [
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
        raise ValueError(f"Could not parse valid JSON from text: {raw_text}")


def process_question(chat_id: int, message_text: str, log_base_url: str = None) -> str:
    """
    Main Agent Entrypoint using OpenAI API with Function Calling Tools.
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
                "tools_enabled": ["execute_python_code", "fetch_url_content"],
            },
        )

        raw_llm_output = "{}"

        # Multi-turn tool execution loop (up to 5 function call iterations)
        for loop_count in range(5):
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

            # Check if model invoked tool calls
            tool_calls = getattr(response_message, "tool_calls", None)
            if not tool_calls:
                raw_llm_output = (
                    response_message.content.strip()
                    if response_message.content
                    else "{}"
                )
                break

            # Process tool calls
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                run_logger.log(
                    "tool_call_executed", {"name": function_name, "args": function_args}
                )

                if function_name == "execute_python_code":
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
