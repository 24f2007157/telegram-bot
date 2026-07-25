import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
from agent import clear_history, process_question
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from logger import LOG_DIR

# 1. Load Environment Variables from .env
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# 2. Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("main_webhook")


# 3. Modern FastAPI Lifespan Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that automatically registers the Telegram Webhook
    via HTTP POST on server startup.
    """
    logger.info("=" * 50)
    logger.info("🚀 FASTAPI WEBHOOK SERVER STARTING UP...")
    logger.info(f"📌 BOT_TOKEN: {'FOUND' if BOT_TOKEN else '❌ MISSING'}")
    logger.info(f"📌 WEBHOOK_URL: {WEBHOOK_URL if WEBHOOK_URL else '❌ MISSING'}")

    if BOT_TOKEN and WEBHOOK_URL:
        full_webhook_endpoint = WEBHOOK_URL.rstrip("/")
        if not full_webhook_endpoint.endswith("/webhook"):
            full_webhook_endpoint += "/webhook"

        telegram_api_url = f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook"
        payload = {
            "url": full_webhook_endpoint,
            "drop_pending_updates": True,  # Clears stale test messages on startup
        }

        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(telegram_api_url, json=payload, timeout=10.0)
                logger.info(f"✅ TELEGRAM WEBHOOK REGISTRATION RESULT: {res.json()}")
        except Exception as e:
            logger.error(f"❌ Failed to register webhook: {e}")
    else:
        logger.warning("⚠️ SKIPPED WEBHOOK REGISTRATION: Token or URL missing in .env!")

    logger.info("=" * 50)

    yield  # Server handles incoming HTTP requests here

    logger.info("🛑 FASTAPI WEBHOOK SERVER SHUTTING DOWN...")


# 4. Initialize FastAPI App
app = FastAPI(
    title="Data Analyst Telegram Bot Webhook API",
    description="FastAPI Webhook Server integrated with Gemini LLM Agent for Project 1",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static logs directory so log_url is publicly wget-able by the grader
os.makedirs(LOG_DIR, exist_ok=True)
app.mount("/logs", StaticFiles(directory=LOG_DIR), name="logs")


# 5. Helper function to send Telegram reply
async def send_telegram_message(chat_id: int, text: str):
    """Sends a message back to Telegram user via Telegram Bot API POST request."""
    if not BOT_TOKEN:
        logger.error("Cannot send Telegram message: BOT_TOKEN missing!")
        return

    send_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(send_url, json=payload, timeout=15.0)
            if res.status_code != 200:
                logger.error(f"Telegram sendMessage failed: {res.text}")
    except Exception as e:
        logger.error(f"Error sending message to Telegram: {e}")


# 6. Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run / load balancers."""
    return {"status": "ok", "bot_configured": BOT_TOKEN is not None}


@app.get("/webhook/info")
async def webhook_info():
    """Queries Telegram API for active Webhook status and error diagnostics."""
    if not BOT_TOKEN:
        raise HTTPException(status_code=500, detail="BOT_TOKEN missing.")

    info_url = f"https://api.telegram.org/bot{BOT_TOKEN}/getWebhookInfo"
    async with httpx.AsyncClient() as client:
        res = await client.get(info_url)
        return res.json()


@app.delete("/webhook")
async def delete_webhook():
    """Manually unregisters the webhook from Telegram."""
    if not BOT_TOKEN:
        raise HTTPException(status_code=500, detail="BOT_TOKEN missing.")

    del_url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook"
    async with httpx.AsyncClient() as client:
        res = await client.post(del_url)
        return res.json()


@app.post("/webhook")
async def receive_telegram_webhook(request: Request):
    """
    Main Webhook Handler:
    Receives POST updates from Telegram, passes the prompt to the Gemini LLM Agent,
    and returns the strict JSON response back to Telegram.
    """
    try:
        data = await request.json()
        logger.info(f"Incoming Webhook Payload: {data}")

        # Check if update contains a message
        if "message" in data and "text" in data["message"]:
            chat_id = data["message"]["chat"]["id"]
            user_text = data["message"]["text"]

            # Reset history on /start command
            if user_text.strip() == "/start":
                clear_history(chat_id)
                welcome_reply = '{"answer": {"status": "ready"}, "log_url": ""}'
                await send_telegram_message(chat_id, welcome_reply)
                return {"status": "ok"}

            # Infer public base URL for log_url
            log_base_url = os.getenv("LOG_BASE_URL")
            if not log_base_url and WEBHOOK_URL:
                base_domain = (
                    WEBHOOK_URL.rsplit("/", 1)[0]
                    if "/webhook" in WEBHOOK_URL
                    else WEBHOOK_URL
                )
                log_base_url = f"{base_domain}/logs"

            # -------------------------------------------------------------
            # Execute LLM Agent Logic (Gemini 2.5 Flash + Multi-Turn + Logging)
            # -------------------------------------------------------------
            try:
                # Wrap agent execution with a 240s safety timeout guard
                reply_json_string = await asyncio.wait_for(
                    asyncio.to_thread(
                        process_question, chat_id, user_text, log_base_url
                    ),
                    timeout=240.0,  # 4-minute maximum safety guard
                )
            except asyncio.TimeoutError:
                logger.error(f"Agent timed out after 240s for chat {chat_id}")
                reply_json_string = json.dumps(
                    {
                        "answer": {"error": "Processing timed out"},
                        "log_url": f"{log_base_url}/timeout.jsonl",
                    }
                )

            logger.info(f"Agent Reply for Chat {chat_id}: {reply_json_string}")

            # Send exact JSON response string back to Telegram user/grader
            await send_telegram_message(chat_id, reply_json_string)

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
