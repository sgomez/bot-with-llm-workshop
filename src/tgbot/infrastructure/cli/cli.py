import json
import logging
import sys

import typer
from langchain.text_splitter import MarkdownHeaderTextSplitter
from rich import print, print_json

from tgbot.infrastructure.bot.model import general_question
from tgbot.infrastructure.bot.model_with_memory import general_question_with_memory
from tgbot.infrastructure.cli.AsyncTyper import AsyncTyper

from ..bot import bot
from ..chroma import database
from ..config import settings

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

chunk_size = 1000
chunk_overlap = 0


app = AsyncTyper()


@app.command()
def train() -> None:
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    documents = [
        "documents/reglamento-tfg-epsc/01-capitulo-01.md",
        "documents/reglamento-tfg-epsc/01-capitulo-02.md",
        "documents/reglamento-tfg-epsc/01-capitulo-03.md",
        "documents/reglamento-tfg-epsc/01-capitulo-04.md",
        "documents/reglamento-tfg-epsc/01-capitulo-05.md",
        "documents/reglamento-tfg-epsc/01-capitulo-06.md",
        "documents/reglamento-tfg-epsc/01-capitulo-07.md",
    ]

    for document in documents:
        with open(document) as fd:
            md_file = fd.read()
            md_header_splits = markdown_splitter.split_text(md_file)
            database.add_documents(md_header_splits)

    database.persist()


@app.command()
def ask(question: str) -> None:
    result = general_question(question)
    print(result)


@app.command()
def askm(chat_id: str, question: str) -> None:
    result = general_question_with_memory(chat_id, question)
    print(result)


@app.command()
def about() -> None:
    typer.echo("This is a bot created from aulasoftwarelibre/telegram-bot-template")


@app.async_command()
async def info() -> None:
    """Returns information about the bot."""
    result = await bot.get_me()
    print("Bot me information")
    print_json(result.to_json())
    result = await bot.get_webhook_info()
    print("Bot webhook information")
    print_json(
        json.dumps(
            {
                "url": result.url,
                "has_custom_certificate": result.has_custom_certificate,
                "pending_update_count": result.pending_update_count,
                "ip_address": result.ip_address,
                "last_error_date": result.last_error_date,
                "last_error_message": result.last_error_message,
                "last_synchronization_error_date": result.last_synchronization_error_date,
                "max_connections": result.max_connections,
                "allowed_updates": result.allowed_updates,
            }
        )
    )
    await bot.close_session()


@app.async_command()
async def install() -> None:
    """Install bot webhook"""
    # Remove webhook, it fails sometimes the set if there is a previous webhook
    await bot.remove_webhook()

    WEBHOOK_URL_BASE = f"https://{settings.webhook_host}:{443}"
    WEBHOOK_URL_PATH = f"/{settings.secret_token}/"

    # Set webhook
    result = await bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH)

    print(f"Set webhook to {WEBHOOK_URL_BASE + WEBHOOK_URL_PATH}: {result}")

    await bot.close_session()


@app.async_command()
async def serve() -> None:
    """Run polling bot version."""
    logging.info("Starting...")

    await bot.remove_webhook()
    await bot.infinity_polling(logger_level=logging.INFO, restart_on_change=True)

    await bot.close_session()


@app.async_command()
async def uninstall() -> None:
    """Uninstall bot webhook."""
    await bot.remove_webhook()

    await bot.close_session()


if __name__ == "__main__":
    app()
