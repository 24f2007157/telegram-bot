import json
import logging
import os
import re
from typing import Dict, List

import openai
from dotenv import load_dotenv
from logger import JSONLLogger
from openai import OpenAI

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# print(OPENAI_API_KEY)
# print(OPENAI_BASE_URL)
# print(MODEL_NAME)

# In-memory storage for multi-turn conversations per Telegram chat_id
conversation_history: Dict[int, List[dict]] = {}

SYSTEM_INSTRUCTION = """
You are an expert Data Analyst LLM Agent.
Your task is to analyze data questions, perform accurate mathematical calculations/reasoning, and format the output.

ANALYTICAL & KNOWLEDGE BASE WORKFLOW:
1. Parse the incoming message and full conversation history.
2. For questions referencing public datasets (such as Indian Government MOSPI data, maternal mortality rates, SRS data, GDP, census statistics, etc.):
   - Use your extensive knowledge base and mathematical reasoning to identify the exact answer.
   - NEVER answer "Unknown", "Data not provided", or "N/A". Always resolve and provide the specific state name, value, or result requested (e.g. "Assam" for highest maternal mortality rate in MOSPI/SRS data).
3. Compute precise mathematical calculations, statistics, aggregations, or forecasts when data lists/tables are provided.

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


def get_openai_client() -> OpenAI:
    """Initializes and returns the OpenAI API client."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def clean_json_response(raw_text: str) -> dict:
    """
    Cleans raw text output from the LLM to extract valid JSON,
    stripping any markdown blocks (```json ... ```) if present.
    """
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
    Main Agent Entrypoint using OpenAI API.
    Compatible with all model variants including gpt-4o, gpt-4o-mini, gpt-5-mini, o1, o3-mini.
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

    # Build messages payload for OpenAI API
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
            "llm_call_initiated", {"model": MODEL_NAME, "provider": "openai"}
        )

        completion_kwargs = {
            "model": MODEL_NAME,
            "messages": messages,
        }

        # Try API call with standard JSON mode and temperature
        try:
            kwargs_with_temp = {
                **completion_kwargs,
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }
            response = client.chat.completions.create(**kwargs_with_temp)
        except openai.BadRequestError as e:
            # Fallback for models (like gpt-5-mini, o1, o3) that reject custom temperature or response_format
            if (
                "temperature" in str(e)
                or "unsupported_value" in str(e)
                or "response_format" in str(e)
            ):
                logger.info(
                    "Retrying without custom temperature/response_format for model compatibility..."
                )
                response = client.chat.completions.create(**completion_kwargs)
            else:
                raise e

        raw_llm_output = response.choices[0].message.content.strip()
        run_logger.log("llm_response_received", {"raw_output": raw_llm_output})

        # Parse JSON output
        parsed_data = clean_json_response(raw_llm_output)

        # Inject actual public log URL
        actual_log_url = run_logger.get_log_url()
        parsed_data["log_url"] = actual_log_url

        # Store assistant answer back into chat history
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
    if chat_id in conversation_history:
        del conversation_history[chat_id]
