import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from typing import Tuple


def reduce_mem_usage(props: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Convert data types of a pandas DataFrame
    """
    
    if verbose:
        start_mem_usg = props.memory_usage().sum() / 1024**2 
        print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if not is_datetime(props[col]):  # Exclude datetime
            
            if verbose:
                # Print current column type
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",props[col].dtype)
            
            if props[col].dtype == object: 
                try:
                    props[col] = props[col].astype(np.float32)
                except Exception as e:
                    if verbose:
                        print(e)
                    continue           
            
            # make variables for Int, max and min
            IsInt = True
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                IsInt = False
#                 props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
#             asint = props[col].fillna(0).astype(np.int64)
#             result = (props[col] - asint)
#             result = result.sum()
#             if result > -0.01 and result < 0.01:
#                 IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
#             else:
#                 props[col] = props[col].astype(np.float32)
            if verbose:
                # Print new column type
                print("dtype after: ",props[col].dtype)
                print("******************************")
    
    if verbose:
        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = props.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    
    return props, NAlist


def preprocess_orders_and_clicks(
    path_to_orders: str = "./data/raw",
    path_to_clicks: str = "./data/raw",
    save_path: str = "./data",
    is_save: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Чтение данных заказов и кликов, конвертация типов, небольшая фильтрация
    """
    
    orders = pd.concat(
        [pd.read_pickle(f'{path_to_orders}/orders{i}.pkl') for i in [1, 2, 3]],
        ignore_index=True
    )
    orders = orders.drop(columns=['index'])
    orders, _ = reduce_mem_usage(orders, verbose=verbose)
    for col in orders.select_dtypes(include=[np.float64]).columns:
        if col not in 'delivery_distance':
            orders[col] = orders[col].astype(np.float32)
    
    orders['discount_percent'] = orders["discount_value"] / orders["initial_product_sum"]
    orders['fee_percent'] = (orders["delivery_fee"] + orders["service_fee"]) / orders["total_value"]
    
    if verbose:
        print(orders.dtypes)
    
    # фильтрация
    # только заказы из ресторанов
    orders = orders[orders.category_id.isin([1])]
    
    clicks = pd.concat(
        [pd.read_pickle(f'{path_to_clicks}/click_{i}.pkl') for i in [1, 2]],
        ignore_index=True
    )
    clicks, _ = reduce_mem_usage(clicks, verbose=verbose)

    if verbose:
        print(orders.dtypes)
    
    # фильтрация
    # только клики по ресторанам
    clicks = clicks[clicks.chain_id.isin(orders.chain_id.unique())]
    
    if is_save:
        orders.to_parquet(save_path + "/orders.parquet")
        clicks.to_parquet(save_path + "/clicks.parquet")
    
    return orders, clicks


def additional_filtration_orders_and_clicks(
    orders: pd.DataFrame,
    clicks: pd.DataFrame,
    order_cnt: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Дополнительная фильтрация, после генерации фичей
    (есть фичи завязанные на успешность/неуспешность заказа; 
    отфильтровываются рестораны с малым числом заказов),
    но до построения матрицы интеракций
    
    orders - предобработанные заказы
    clicks - предобработанные клики
    order_cnt - минимальное количество заказов которое должно быть у ресторана
    """
    # только Мск
    orders = orders[orders.city_id.isin([1])]
    # отсекаем рестораны без заказов
    orders = orders[orders.groupby("chain_id")['chain_id'].transform('size') >= order_cnt]
    # успешный заказ и перезаказ
    orders = orders[orders.status_id.isin([11, 18])] 
    
    clicks = clicks[clicks.city_id.isin([1])]
    clicks = clicks[clicks.chain_id.isin(orders.chain_id.unique())]
    
    return orders, clicks
