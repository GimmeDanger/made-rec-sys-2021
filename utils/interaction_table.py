import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype


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
    clicks_df = clicks_df.reset_index()
    clicks_df = clicks_df[['user_id', 'chain_id', 'weight']]
    clicks_df['weight'] /= clicks_df['weight'].max()
    
    if clicks_df['weight'].min() < 0 or clicks_df['weight'].max() > 1:
        raise RuntimeError("Invalid input: clicks weight must be in [0, 1]")
    return clicks_df


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
    orders_df['weight'] /= orders_df['weight'].max()
    
    if orders_df['weight'].min() < 0 or orders_df['weight'].max() > 1:
        raise RuntimeError("Invalid input: orders weight must be in [0, 1]")
    return orders_df


class InteractionTable:
    
    """
    weight = alpha * click_weight + (1 - alpha) * orders_weight
    alpha in [0, 1], click_weight in [0, 1], orders_weight in [0, 1]
    so final weight in (0, 1]
    """
    def __init__(self, clicks_getter, orders_getter, clicks_weigher, orders_weigher, alpha=0):

        if alpha < 0 or alpha > 1:
            raise RuntimeError("Invalid input: alpha must be in [0, 1]")
            
        self.alpha = alpha
        self.clicks_df = pd.DataFrame()
        self.orders_df = pd.DataFrame()
        
        if clicks_getter is not None:
            self.clicks_df = self.load(clicks_getter, clicks_weigher, 'Clicks')
            self.clicks_df['weight'] *= self.alpha
        
        if orders_getter is not None:
            self.orders_df = self.load(orders_getter, orders_weigher, 'Orders')
            self.orders_df['weight'] *= (1 - self.alpha)
        
        self.interaction_df = pd.concat([self.clicks_df, self.orders_df], ignore_index=True)
        
        self.chain_index = self.get_uniqs_index(self.interaction_df.chain_id)
        self.r_chain_index = sorted(self.interaction_df.chain_id.unique())
        
        self.user_index = self.get_uniqs_index(self.interaction_df.user_id)
        self.r_user_index = sorted(self.interaction_df.user_id.unique())
        
        self.sparse_interaction_matrix = self.get_sparse_interaction_matrix(self.interaction_df)
    
    def load(self, getter, weigher, label):
        df = getter()
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
    
    def get_uniqs_index(self, df_column):
        """
        mapping ids (user or chain) -> uniq id starting from 0
        """
        uniqs = sorted(df_column.unique())
        uniqs_index = dict(zip(uniqs, [x for x in range(len(uniqs))]))
        return uniqs_index
