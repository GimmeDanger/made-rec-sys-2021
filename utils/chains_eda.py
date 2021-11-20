import numpy as np
import pandas as pd
from collections import Counter
from process_data import reduce_mem_usage
import os

def chains_eda(orders_df, chains_df) -> pd.DataFrame:
    
    orders = orders_df[orders_df.status_id == 11]
    chains = chains_df.copy()
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
    orders_group_df = orders_group_df[orders_group_df.num_orders != 1]
    orders_group_df.loc[:, 'discount_value'] = orders_group_df.discount_value / orders_group_df.initial_product_sum

    chains.drop(
        chains[chains.chain_id.isna()].index,
        axis=0,
        inplace=True,
    )
    chains.drop(
        chains[chains.is_test_chain == 1].index,
        axis=0,
        inplace=True,
    )
    chains = chains[chains.category_id == 1]
    
    
    def count_val(col: pd.Series):
        return pd.Series([i for lst in col for i in lst])

    top_product_dict = Counter(count_val(chains.product_group_ids.dropna()).value_counts().to_dict(dict))
    top_cousine_dict = Counter(count_val(chains.product_group_ids.dropna()).value_counts().to_dict(dict))
    chains.product_group_ids = chains.product_group_ids.apply(lambda d: d if isinstance(d, list) else [])
    chains.loc[:, 'top3_products'] = chains.product_group_ids.map(
        lambda x: any([True for i in x if i in [i[0] for i in top_product_dict.most_common(3)]]))
    chains.loc[:, 'top1_products'] = chains.product_group_ids.map(
        lambda x: any([True for i in x if i in [i[0] for i in top_product_dict.most_common(1)]]))
    chains.loc[:, 'top10_products'] = chains.product_group_ids.map(
        lambda x: any([True for i in x if i in [i[0] for i in top_product_dict.most_common(10)]]))
    chains.loc[:, 'defaults_in_top3'] = chains.default_product_group_id.map(
        lambda x: True if x in [i[0] for i in top_product_dict.most_common(3)] else False)
    chains.loc[:, 'vendors_in_chain'] = chains.groupby('chain_id')['vendor_id'].transform(len)
    chains.drop([
        'vendor_id', 'chain_name', 'chain_name_en',
        'chain_created_at', 'vendor_created_at',
        'category_id', 'default_product_group_id', 'product_group_ids',
        'cuisine_ids', 'food_for_points_marker', 'takeaway_support',
        'is_test_chain', 'citymobil_support',
        'deliveryprovider_id', 'lat', 'lon', 'chain_active',
        'chain_deleted', 'vendor_deleted', 'city_id', 'is_qsr',
    ], axis=1, inplace=True)
    chains_group_df = chains.groupby('chain_id').agg(np.mean)
    df = chains_group_df.join(orders_group_df, how='inner')
    df.loc[:, 'order_per_vendor'] = df.num_orders / df.vendors_in_chain
    return df
