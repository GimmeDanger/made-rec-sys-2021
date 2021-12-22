import utils
import logging
import pandas as pd
import numpy as np
from h3 import geo_to_h3
from model import Model
from random import sample
from collections import defaultdict

from aiogram import Bot, types
from aiogram.utils.markdown import text
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types.message import ContentType
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton


from config import TOKEN, H3_RESOLUTION, \
    USER_CHOICE_SIZE, RECOMMEND_TOP_K, HIST_POOL_TOP_K


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
model = Model()

# sample state
sample_df = pd.read_parquet('data/results_30.parquet')
sample_user_ids = {u: i for i, u in enumerate(sample_df.user_id.unique())}
sample_df.user_id = sample_df.user_id.map(sample_user_ids)
sample_user_ids = sample_df.user_id.unique()
sample_state = defaultdict(lambda: defaultdict(str))

# recommend state
h3_to_chains = pd.read_pickle('data/h3_to_chains.pkl')
h3_to_city_id = pd.read_pickle('data/h3_to_city_id.pkl')
chain_id_to_name = pd.read_pickle('data/chain_id_to_name.pkl')
city_id_to_name = {city_id: name for name, city_id in pd.read_pickle('data/city_name_to_id.pkl').items()}
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


def generate_kb_top_rest(chains, max_btn_in_row=2, max_btn_in_kb=6):
    kb = InlineKeyboardMarkup(row_width=max_btn_in_row, resize_keyboard=True)
    btns = []
    for chain_id in sample(chains, min(len(chains), max_btn_in_kb)):
        chain_name = chain_id_to_name[chain_id]
        btns.append(InlineKeyboardButton(chain_name, callback_data=f'btn_kb_top_rest_{chain_id}'))
        if len(btns) % max_btn_in_row == 0:
            kb.row(*btns, )
            btns = []
    if len(btns) != 0:
        kb.row(*btns, )
    kb.add(InlineKeyboardButton('Обновить 🔄', callback_data='btn_kb_top_rest_refresh'))
    return kb


def drop_same_names(top_rec, top_k=30):
    selected_chains = set()
    h3_top_chains = []
    for id in top_rec:
        chain_name = chain_id_to_name[id]
        if chain_name not in selected_chains:
            h3_top_chains.append(id)
            selected_chains.add(chain_name)
    return h3_top_chains[:top_k]


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    msg = 'Бот для рекомендации ресторанов. Нажмите /help для дополнительной информации'
    await message.reply(msg)


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    msg = 'Функции бота\n'
    msg += '\n\n* /recommend - получение рекомендации для свой локации и любимых ресторанов'
    msg += '\n\n* /sample - пример работы на случайном пользователе из выборки'
    await message.reply(msg)


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
    demo_user_id = message['chat'].id
    
    msg = f'lat: {lat}'
    msg += f'\nlng: {lng}'
    if h3 in h3_to_chains:   
        top_rec = model.top_rec(h3, top_k=HIST_POOL_TOP_K)
        h3_top_chains = drop_same_names(top_rec, top_k=RECOMMEND_TOP_K)
        demo_user_state[demo_user_id] = defaultdict()
        demo_user_state[demo_user_id]['h3'] = h3
        demo_user_state[demo_user_id]['h3_top_chains'] = set(h3_top_chains)
        demo_user_state[demo_user_id]['h3_hist_chains'] = set()
        
        logger.info(f'h3 cached for demo user {demo_user_id}')
        logger.info(demo_user_state[demo_user_id])

        msg += f'\nh3: {h3}'
        msg += f'\ncity: {city_id_to_name[h3_to_city_id[h3]]}'
        msg += '\n\nВыберите 5 ресторанов из списка:'
        ids = list(demo_user_state[demo_user_id]['h3_top_chains'])
        markup = generate_kb_top_rest(ids)
    else:
        msg += '\nh3: неизвестный'
        msg += '\n\nПопробуйте еще раз!'
        markup = None

    logger.info(f'reply msg: {msg}')
    await bot.send_message(demo_user_id, msg, reply_markup=markup)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith('btn_kb_top_rest'))
async def process_callback_top_rest(callback_query: types.CallbackQuery):

    data = callback_query.data
    demo_user_id = callback_query.message.chat.id
    user_h3_top_chains = demo_user_state[demo_user_id]['h3_top_chains']
    user_h3_hist_chains = demo_user_state[demo_user_id]['h3_hist_chains']

    logger.info('/recommend top rest keyboard callback:')
    logger.info(callback_query)
    logger.info(f'demo user id {demo_user_id} state:')
    logger.info([(id, chain_id_to_name[id]) for id in user_h3_top_chains])
    logger.info([(id, chain_id_to_name[id]) for id in user_h3_hist_chains])

    if data.startswith('btn_kb_top_rest_refresh'):
        logger.info('btn refresh')
        ids = list(user_h3_top_chains - user_h3_hist_chains)
        text = callback_query.message.text
        markup = generate_kb_top_rest(ids)

    else:
        chain_id = int(data[len('btn_kb_top_rest_'):])
        user_h3_hist_chains.add(chain_id)
        demo_user_state[demo_user_id]['h3_hist_chains'] = user_h3_hist_chains
        logger.info(f'btn {chain_id}: {chain_id_to_name[chain_id]}')
        if USER_CHOICE_SIZE <= len(user_h3_hist_chains):
            logger.info('enough history gathered for prediction')
            preds = model.predict(h3=demo_user_state[demo_user_id]['h3'],
                                  user_orders_history=user_h3_hist_chains,
                                  top_k=2*RECOMMEND_TOP_K)
            preds = drop_same_names(preds, top_k=RECOMMEND_TOP_K)
            preds = [chain_id_to_name[chain_id] for chain_id in preds]
            hist = [chain_id_to_name[chain_id] for chain_id in user_h3_hist_chains]
            text = 'Выбранные рестораны: ' + ', '.join([s for s in hist])
            text += '\n\nРекомендации:\n' + '\n'.join([f'    {i+1}. {s}' for i, s in enumerate(preds)])
            markup = None
            
        else:
            logger.info('continue adding hist chains from kb')
            ids = list(user_h3_top_chains - user_h3_hist_chains)
            text = callback_query.message.text
            markup = generate_kb_top_rest(ids)
    
    logger.info(f'reply msg: {text}')
    await bot.edit_message_text(
        chat_id=callback_query.message.chat.id,
        message_id=callback_query.message.message_id,
        reply_markup=markup,
        text=text)

if __name__ == '__main__':
    executor.start_polling(dp)
