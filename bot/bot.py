import utils
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

from aiogram import Bot, types
from aiogram.utils.markdown import text
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton


from config import TOKEN, DATA_PATH


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


# state
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
df = pd.read_parquet(DATA_PATH)
user_ids = {u: i for i, u in enumerate(df.user_id.unique())}
df.user_id = df.user_id.map(user_ids)
user_ids = df.user_id.unique()
state = defaultdict(lambda: defaultdict(str))


# /sample keyboards
btn_sample_new_user = InlineKeyboardButton('new user', callback_data='btn_sample_pred_new_user')
btn_sample_new_h3 = InlineKeyboardButton('new h3', callback_data='btn_sample_pred_new_h3')
btn_sample_top_k = InlineKeyboardButton('top_k', callback_data='btn_sample_pred_top_k')
btn_sample_als = InlineKeyboardButton('als', callback_data='btn_sample_pred_als')
btn_sample_top_rec = InlineKeyboardButton('top_rec', callback_data='btn_sample_pred_top_rec')
btn_sample_lightfm = InlineKeyboardButton('lightfm', callback_data='btn_sample_pred_lightfm')
kb_sample_pred = InlineKeyboardMarkup(row_width=3, resize_keyboard=True).row(btn_sample_new_user, btn_sample_new_h3, btn_sample_top_k)
kb_sample_pred.row(btn_sample_als, btn_sample_top_rec, btn_sample_lightfm)

# /predict keyboards


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("TODO: start")


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply("TODO: help")


@dp.message_handler(commands=['sample'])
async def process_command_sample(message: types.Message):
    state_id = message['from'].id
    user_id = utils.get_user_id(df, user_ids)
    h3 = utils.get_h3(df, user_id)
    msg = utils.get_user_hist(df, user_id, h3)
    state[state_id]['user_id'] = user_id
    state[state_id]['h3'] = h3
    state[state_id]['model'] = 'als'
    state[state_id]['top_k'] = 5
    logger.info('/sample started from:')
    logger.info(message)
    logger.info(f'new state of msg {state_id}:')
    logger.info(state[state_id])
    await message.reply(msg, reply_markup=kb_sample_pred)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith('btn_sample'))
async def process_callback_sample(callback_query: types.CallbackQuery):
    state_id = callback_query.message.reply_to_message['from'].id
    user_id = int(state[state_id]['user_id'])
    h3 = str(state[state_id]['h3'])
    top_k = int(state[state_id]['top_k'])
    model = str(state[state_id]['model'])
    data = callback_query.data
    if data.startswith('btn_sample_pred_new_user'):
        user_id = utils.get_user_id(df, user_ids)
        if h3 not in set(df[df['user_id'] == user_id]['h3'].unique()):
            h3 = utils.get_h3(df, user_id)
        msg = utils.get_user_hist(df, user_id, h3)
        markup = kb_sample_pred

    elif data.startswith('btn_sample_pred_new_h3'):
        h3 = utils.get_h3(df, user_id)
        msg = utils.get_user_hist(df, user_id, h3)
        markup = kb_sample_pred

    elif data.startswith('btn_sample_pred_top_k'):
        top_k = utils.next_top_k(top_k)
        msg = utils.get_user_hist(df, user_id, h3) + '\n'
        msg += utils.get_user_pred(df, user_id, h3, model, top_k)
        markup = kb_sample_pred

    else:
        model = data[len('btn_sample_pred_'):]
        msg = utils.get_user_hist(df, user_id, h3) + '\n'
        msg += utils.get_user_pred(df, user_id, h3, model, top_k)
        markup = kb_sample_pred
    
    state[state_id]['user_id'] = str(user_id)
    state[state_id]['h3'] = str(h3)
    state[state_id]['top_k'] = str(top_k)
    state[state_id]['model'] = str(model)
    msg += '\n'

    logger.info('/sample keyboard callback:')
    logger.info(callback_query)
    logger.info(f'new state of msg {state_id}:')
    logger.info(state[state_id])
    logger.info(f'reply msg: {msg}')

    await bot.edit_message_text(
        chat_id=callback_query.message.chat.id,
        message_id=callback_query.message.message_id,
        reply_markup=markup,
        text=msg)


if __name__ == '__main__':
    executor.start_polling(dp)