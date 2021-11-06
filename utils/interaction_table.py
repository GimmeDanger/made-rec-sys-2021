import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

# TODO: replace with filtered datasets

def get_clicks():
    path = '../data/clicks/click.pkl'
    return pd.read_pickle(path)


def get_orders():
    path = '../data/orders/orders.pkl'
    return pd.read_pickle(path)


def clicks_weigher(clicks_df, max_clicks_per_session = 100):
    """
    (chain, user) -> interaction weight
    """
    print('Clicks weighter: use user clicks per chain as weight')
    clicks_df = clicks_df[['user_id', 'session_id', 'chain_id']]
    
    # erase sessions with fraud click count
    clicks_df['cnt'] = 1
    by_session = clicks_df[['session_id', 'cnt']].groupby(['session_id']).sum()
    assert len(by_session) == len(by_session.index.unique())
    by_session = by_session.drop(by_session[by_session.cnt > max_clicks_per_session].index)
    clicks_df = clicks_df.join(by_session, on='session_id', how='inner', lsuffix='_caller', rsuffix='_other')
    clicks_df = clicks_df[['user_id', 'chain_id']]
    
    # create table (user_id, chain_id) -> weight
    clicks_df['weight'] = 1
    clicks_df = clicks_df.groupby(['user_id', 'chain_id']).sum()
    return clicks_df.reset_index()[['user_id', 'chain_id', 'weight']]


def orders_weigher(orders_df, max_orders_per_chain = 50):
    """
    (chain, user) -> interaction weight
    """
    print('Orders weighter: use successful user orders per chain as weight')
    # TODO: consider other statuses in weight function!
    
    SUCCESS_STATUS = 11
    orders_df = orders_df[['user_id', 'status_id', 'chain_id']]
    orders_df = orders_df.drop(orders_df[orders_df.status_id != SUCCESS_STATUS].index)

    orders_df['weight'] = 1
    orders_df = orders_df.groupby(['user_id', 'chain_id']).sum()
    orders_df = orders_df.reset_index()[['user_id', 'chain_id', 'weight']]
    orders_df = orders_df.drop(orders_df[orders_df.weight > max_orders_per_chain].index)
    orders_df = orders_df.drop(orders_df[orders_df.user_id < 0].index)
    return orders_df


class InteractionTable:
    
    def __init__(self, clicks_getter, orders_getter, clicks_weigher, orders_weigher):
        clicks_df = self.load(clicks_getter, clicks_weigher, 'Clicks')
        orders_df = self.load(orders_getter, orders_weigher, 'Orders')
        self.interaction_df = pd.concat([clicks_df, orders_df], ignore_index=True)
        self.sparse_interaction_matrix = self.get_sparse_interaction_matrix(self.interaction_df)
    
    def load(self, getter, weigher, label):
        df = getter()
        if 'user_id' not in df.columns:
            df['user_id'] = df['customer_id'].astype('int64')
        print(f'{label} df loaded: size={len(df)},',
              f' uniq_users={len(df.user_id.unique())},',
              f' uniq_chains={len(df.chain_id.unique())}')
        df = weigher(df)
        print(f'{label} df weighted: size={len(df)},',
              f'uniq_users={len(df.user_id.unique())},',
              f'uniq_chains={len(df.chain_id.unique())}')
        return df
        
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
