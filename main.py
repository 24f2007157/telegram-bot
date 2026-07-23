import json
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Load environment variables from .env
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Setup basic logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Send me a data analysis question.")


# Message handler (Processes incoming user text)
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    logging.info(f"Received message: {user_text}")

    # -------------------------------------------------------------
    # TODO: Place your LLM / Data Analyst Agent logic here
    # For now, we return a mock JSON response as required by Project 1
    # -------------------------------------------------------------
    response_payload = {
        "answer": {"result": "Data analyzed successfully"},
        "log_url": "https://your-host/run.jsonl",
    }

    # Convert payload to strict JSON string (no additional text)
    json_response = json.dumps(response_payload)

    # Send reply back to Telegram user
    await update.message.reply_text(json_response)


def main():
    if not BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in environment variables.")

    # Initialize the bot application
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start polling for incoming messages
    logging.info("Bot is starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
