import utils
import logging
import pandas as pd
import numpy as np
from h3 import geo_to_h3
from collections import defaultdict

from aiogram import Bot, types
from aiogram.utils.markdown import text
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types.message import ContentType
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton


from config import TOKEN, DATA_PATH, H3_RESOLUTION, \
    H3_TO_CHAINS_PATH, H3_TO_CITY_ID_PATH, CITY_ID_TO_NAME_PATH


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# sample state
sample_df = pd.read_parquet(DATA_PATH)
sample_user_ids = {u: i for i, u in enumerate(sample_df.user_id.unique())}
sample_df.user_id = sample_df.user_id.map(sample_user_ids)
sample_user_ids = sample_df.user_id.unique()
sample_state = defaultdict(lambda: defaultdict(str))

# predict state
h3_to_chains = pd.read_pickle(H3_TO_CHAINS_PATH)
h3_to_city_id = pd.read_pickle(H3_TO_CITY_ID_PATH)
city_id_to_name = {city_id: name for name, city_id in pd.read_pickle(CITY_ID_TO_NAME_PATH).items()}
demo_user_state = defaultdict(lambda: defaultdict(str))

# /sample keyboards
btn_sample_new_user = InlineKeyboardButton('new user', callback_data='btn_sample_pred_new_user')
btn_sample_new_h3 = InlineKeyboardButton('new h3', callback_data='btn_sample_pred_new_h3')
btn_sample_top_k = InlineKeyboardButton('top_k', callback_data='btn_sample_pred_top_k')
btn_sample_als = InlineKeyboardButton('als', callback_data='btn_sample_pred_als')
btn_sample_top_rec = InlineKeyboardButton('top_rec', callback_data='btn_sample_pred_top_rec')
btn_sample_lightfm = InlineKeyboardButton('lightfm', callback_data='btn_sample_pred_lightfm')
kb_sample_pred = InlineKeyboardMarkup(row_width=3, resize_keyboard=True).row(btn_sample_new_user, btn_sample_new_h3, btn_sample_top_k)
kb_sample_pred.row(btn_sample_als, btn_sample_top_rec, btn_sample_lightfm)

# /recommend keyboards
kb_loc_request = ReplyKeyboardMarkup(resize_keyboard=True).add(
    KeyboardButton('Отправить свою локацию 🗺️', request_location=True)
)


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("TODO: start")


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.reply("TODO: help")


@dp.message_handler(commands=['sample'])
async def process_command_sample(message: types.Message):
    state_id = message['from'].id
    user_id = utils.get_user_id(sample_df, sample_user_ids)
    h3 = utils.get_h3(sample_df, user_id)
    msg = utils.get_user_hist(sample_df, user_id, h3)
    sample_state[state_id]['user_id'] = user_id
    sample_state[state_id]['h3'] = h3
    sample_state[state_id]['model'] = 'als'
    sample_state[state_id]['top_k'] = 5
    logger.info('/sample started from:')
    logger.info(message)
    logger.info(f'new state of msg {state_id}:')
    logger.info(sample_state[state_id])
    await message.reply(msg, reply_markup=kb_sample_pred)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith('btn_sample'))
async def process_callback_sample(callback_query: types.CallbackQuery):
    state_id = callback_query.message.reply_to_message['from'].id
    user_id = int(sample_state[state_id]['user_id'])
    h3 = str(sample_state[state_id]['h3'])
    top_k = int(sample_state[state_id]['top_k'])
    model = str(sample_state[state_id]['model'])
    data = callback_query.data
    if data.startswith('btn_sample_pred_new_user'):
        user_id = utils.get_user_id(sample_df, sample_user_ids)
        if h3 not in set(sample_df[sample_df['user_id'] == user_id]['h3'].unique()):
            h3 = utils.get_h3(sample_df, user_id)
        msg = utils.get_user_hist(sample_df, user_id, h3)
        markup = kb_sample_pred

    elif data.startswith('btn_sample_pred_new_h3'):
        h3 = utils.get_h3(sample_df, user_id)
        msg = utils.get_user_hist(sample_df, user_id, h3)
        markup = kb_sample_pred

    elif data.startswith('btn_sample_pred_top_k'):
        top_k = utils.next_top_k(top_k)
        msg = utils.get_user_hist(sample_df, user_id, h3) + '\n'
        msg += utils.get_user_pred(sample_df, user_id, h3, model, top_k)
        markup = kb_sample_pred

    else:
        model = data[len('btn_sample_pred_'):]
        msg = utils.get_user_hist(sample_df, user_id, h3) + '\n'
        msg += utils.get_user_pred(sample_df, user_id, h3, model, top_k)
        markup = kb_sample_pred
    
    sample_state[state_id]['user_id'] = str(user_id)
    sample_state[state_id]['h3'] = str(h3)
    sample_state[state_id]['top_k'] = str(top_k)
    sample_state[state_id]['model'] = str(model)
    msg += '\n'

    logger.info('/sample keyboard callback:')
    logger.info(callback_query)
    logger.info(f'new state of msg {state_id}:')
    logger.info(sample_state[state_id])
    logger.info(f'reply msg: {msg}')

    await bot.edit_message_text(
        chat_id=callback_query.message.chat.id,
        message_id=callback_query.message.message_id,
        reply_markup=markup,
        text=msg)


@dp.message_handler(commands=['recommend'])
async def process_recommend_command(message: types.Message):
    logger.info('/recommend started from:')
    logger.info(message)
    await message.reply(
        "Для получения рекомендации отправьте локацию", 
        reply_markup=kb_loc_request)


@dp.message_handler(content_types=ContentType.LOCATION)
async def process_location(message: types.Message):
    logger.info('location parsed from:')
    logger.info(message)
    lat = message['location'].latitude
    lng = message['location'].longitude
    h3 = geo_to_h3(lat=lat,
                   lng=lng,
                   resolution=H3_RESOLUTION)

    demo_user_id = message['from'].id
    msg = f'lat: {lat},\nlng: {lng}\n'
    if h3 in h3_to_chains:
        logger.info(f'h3 cached for demo user {demo_user_id}')
        demo_user_state[demo_user_id] = defaultdict()
        demo_user_state[demo_user_id]['h3'] = h3
        msg += f'h3: {h3}\ncity: {city_id_to_name[h3_to_city_id[h3]]}'
    else:
        msg += 'h3: unknown,\nпопробуйте еще раз'
    logger.info(f'reply msg: {msg}')
    await bot.send_message(demo_user_id, msg)

if __name__ == '__main__':
    executor.start_polling(dp)
