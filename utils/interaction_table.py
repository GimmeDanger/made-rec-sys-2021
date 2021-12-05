import random
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype


def ncf_clicks_weigher(clicks_df):
    """
    (chain, user) -> interaction weight
    """
    clicks_df['weight'] = 1
    clicks_df = clicks_df[['user_id', 'chain_id', 'weight']]
    return clicks_df


def ncf_orders_weigher(orders_df):
    """
    (chain, user) -> interaction weight
    """
    orders_df['weight'] = 1
    orders_df = orders_df[['user_id', 'chain_id', 'weight']]
    return orders_df


def orders_weigher_sum(orders_df, normalize=False):
    """
    (chain, user) -> interaction weight
    """
    print('Orders weighter: use user avg orders per chain as weight')
    orders_df['weight'] = 1
    orders_df = orders_df[['user_id', 'chain_id', 'weight']]
    
    total_user_stats = orders_df.groupby(['user_id']).sum()
    total_user_stats = total_user_stats.reset_index()[['user_id', 'weight']]
    assert total_user_stats.weight.isnull().sum() == 0

    user_chain_stats = orders_df.groupby(['user_id', 'chain_id']).sum()
    user_chain_stats = user_chain_stats.reset_index()[['user_id', 'chain_id', 'weight']]
    assert sorted(total_user_stats.user_id.unique()) == sorted(user_chain_stats.user_id.unique())
    assert user_chain_stats.weight.isnull().sum() == 0
    
    user_chain_stats = user_chain_stats.merge(total_user_stats, left_on='user_id',
                                              right_on='user_id', suffixes=('_per_chain', '_total'))
    if normalize:
        user_chain_stats['weight_per_chain'] /= user_chain_stats['weight_total']

    user_chain_stats = user_chain_stats.rename(columns={'user_id_per_chain': 'user_id', 'weight_per_chain': 'weight'})
    orders_df = user_chain_stats[['user_id', 'chain_id', 'weight']]
    assert len(orders_df) == len(user_chain_stats)
    assert orders_df.weight.isnull().sum() == 0
    print(orders_df.describe())
    print(f'Orders df weighted: size={len(orders_df)},',
          f'uniq_users={len(orders_df.user_id.unique())},',
          f'uniq_chains={len(orders_df.chain_id.unique())}')
    return orders_df


class InteractionTable:
    
    """
    weight = alpha * click_weight + (1 - alpha) * orders_weight
    alpha in [0, 1], click_weight in [0, 1], orders_weight in [0, 1]
    so final weight in (0, 1]
    """
    def __init__(self, orders_df, clicks_df, alpha=0, test_slice=None):

        if alpha < 0 or alpha > 1:
            raise RuntimeError("Invalid input: alpha must be in [0, 1]")
            
        self.alpha = alpha
        self.clicks_df = pd.DataFrame()
        self.orders_df = pd.DataFrame()
        
        if clicks_df is not None:
            self.clicks_df = clicks_weigher(clicks_df, normalize=False)
            self.clicks_df['weight'] *= self.alpha
        
        if orders_df is not None:
            self.orders_df = orders_weigher(orders_df, normalize=False)
            self.orders_df['weight'] *= (1 - self.alpha)
        
        self.interaction_df = pd.concat([self.clicks_df, self.orders_df], ignore_index=True)

        if test_slice is not None:
            uniq_users = self.interaction_df.user_id.unique()
            random.Random(42).shuffle(uniq_users)
            test_users = uniq_users[:test_slice]
            self.interaction_df = self.interaction_df[self.interaction_df["user_id"].isin(test_users)]
            print('Interaction df len for test: ', len(self.interaction_df))
        
        self.chain_to_index = self.get_uniqs_index(self.interaction_df.chain_id)
        self.r_chain_index = list(self.chain_to_index.keys())
        self.index_to_chain = {v: k for k, v in self.chain_to_index.items()}
        
        self.user_to_index = self.get_uniqs_index(self.interaction_df.user_id)
        self.r_user_index = list(self.user_to_index.keys())
        self.index_to_user = {v: k for k, v in self.user_to_index.items()}
        
        self.sparse_interaction_matrix = self.get_sparse_interaction_matrix(self.interaction_df)
    
#     def load(self, df, weigher, label):
#         df = getter()
#         print(f'{label} df loaded: size={len(df)},',
#               f' uniq_users={len(df.user_id.unique())},',
#               f' uniq_chains={len(df.chain_id.unique())}')
#         df = weigher(df)
#         print(f'{label} df weighted: size={len(df)},',
#               f'uniq_users={len(df.user_id.unique())},',
#               f'uniq_chains={len(df.chain_id.unique())}')
#         return df
        
    def get_sparse_interaction_matrix(self, df):
        """
        https://stackoverflow.com/questions/31661604/efficiently-create-sparse-pivot-tables-in-pandas
        user-chain pivot sparse matrix
        """
        chain_c = CategoricalDtype(sorted(df.chain_id.unique()), ordered=True)
        user_c = CategoricalDtype(sorted(df.user_id.unique()), ordered=True)
        
        row = df.chain_id.astype(chain_c).cat.codes
        col = df.user_id.astype(user_c).cat.codes
        assert row.min() >= 0 and col.min() >= 0
        
        sparse_matrix = csr_matrix((df.weight, (row, col)), \
                                   shape=(chain_c.categories.size, user_c.categories.size))
        return sparse_matrix
    
    def get_uniqs_index(self, df_column):
        """
        mapping ids (user or chain) -> uniq id starting from 0
        """
        uniqs = sorted(df_column.unique())
        uniqs_index = dict(zip(uniqs, [x for x in range(len(uniqs))]))
        return uniqs_index
