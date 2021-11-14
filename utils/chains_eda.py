import numpy as np
import pandas as pd
from collections import Counter
from process_data import reduce_mem_usage

PATH_TO_CHAIN = '../data/chains.pkl'
PATH_TO_ORDERS = '../data/orders'


def chains_eda() -> pd.DataFrame:
    path_to_files = PATH_TO_ORDERS
    tables = []
    for num in range(1, 3):
        filename = f'{path_to_files}/orders{num}.pkl'
        print(filename)
        tables.append(pd.read_pickle(filename))

    orders_df = pd.concat(tables)
    orderes_df, _ = reduce_mem_usage(orders_df)
    orders_group_df = orders_df \
        .groupby('chain_id') \
        .agg(
        {'discount_value': np.mean,
         'total_value': np.mean,
         'initial_product_sum': np.mean,
         'index': len
         }
    )
    orders_group_df.rename({'index': 'num_orders'}, axis=1, inplace=True)
    orders_group_df.loc[:, 'discount_value'] = orders_group_df.discount_value / orders_group_df.initial_product_sum
    chains_df = pd.read_pickle(PATH_TO_CHAIN)
    chains_df, _ = reduce_mem_usage(chains_df)
    chains_df.drop(
        chains_df[chains_df.chain_id.isna()].index,
        axis=0,
        inplace=True,
    )
    chains_df.drop(
        chains_df[chains_df.is_test_chain == 1].index,
        axis=0,
        inplace=True,
    )

    def count_val(col: pd.Series):
        return pd.Series([i for lst in col for i in lst])

    top_product_dict = Counter(count_val(chains_df.product_group_ids.dropna()).value_counts().to_dict(dict))
    top_cousine_dict = Counter(count_val(chains_df.product_group_ids.dropna()).value_counts().to_dict(dict))
    chains_df.product_group_ids = chains_df.product_group_ids.apply(lambda d: d if isinstance(d, list) else [])
    chains_df.loc[:, 'top3_products'] = chains_df.product_group_ids.map(
        lambda x: any([True for i in x if i in [i[0] for i in top_product_dict.most_common(3)]]))
    chains_df.loc[:, 'top1_products'] = chains_df.product_group_ids.map(
        lambda x: any([True for i in x if i in [i[0] for i in top_product_dict.most_common(1)]]))
    chains_df.loc[:, 'top10_products'] = chains_df.product_group_ids.map(
        lambda x: any([True for i in x if i in [i[0] for i in top_product_dict.most_common(10)]]))
    chains_df.loc[:, 'defaults_in_top3'] = chains_df.default_product_group_id.map(
        lambda x: True if x in [i[0] for i in top_product_dict.most_common(3)] else False)
    chains_df.loc[:, 'vendors_in_chain'] = chains_df.groupby('chain_id')['vendor_id'].transform(len)
    chains_df.drop([
        'vendor_id', 'chain_name', 'chain_name_en',
        'chain_created_at', 'vendor_created_at',
        'category_id', 'default_product_group_id', 'product_group_ids',
        'cuisine_ids', 'food_for_points_marker', 'takeaway_support',
        'is_test_chain', 'citymobil_support',
        'deliveryprovider_id', 'lat', 'lon', 'chain_active',
        'chain_deleted', 'vendor_deleted', 'city_id', 'is_qsr',
    ], axis=1, inplace=True)
    chains_group_df = chains_df.groupby('chain_id').agg(np.mean)
    df = chains_group_df.join(orders_group_df, how='inner')
    df.loc[:, 'order_per_vendor'] = df.num_orders / df.vendors_in_chain
    return df

# df = chains_eda()