import logging

import telebot
from telebot.async_telebot import AsyncTeleBot
from telebot.types import Message

from ..config import settings
from .model import general_question  # noqa: F401, RUF100
from .model_with_memory import general_question_with_memory  # noqa: F401, RUF100
from .model_with_rag import tfg_question  # noqa: F401, RUF100

bot = AsyncTeleBot(settings.bot_token)
telebot.logger.setLevel(logging.INFO)


@bot.message_handler(commands=["start"])  # type: ignore
async def start(message: Message) -> None:
    await bot.reply_to(message, "Hello, " + message.from_user.first_name)


@bot.message_handler(func=lambda message: True)  # type: ignore # noqa: ARG005
async def echo_message(message: Message) -> None:
    await bot.send_chat_action(message.chat.id, action="typing")
    response = general_question(message.text)
    await bot.send_message(message.chat.id, response)
