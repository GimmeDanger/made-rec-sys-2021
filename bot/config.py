import os

TOKEN = os.getenv('REC_SYS_BOT_TOKEN')
DATA_PATH = 'data/results_30.parquet'

H3_RESOLUTION = 9
H3_TO_CHAINS_PATH = 'data/h3_to_chains.pkl'
H3_TO_CITY_ID_PATH = 'data/h3_to_city_id.pkl'
CITY_ID_TO_NAME_PATH = 'data/city_id_to_name.pkl'