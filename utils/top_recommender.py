from typing import List
from collections import defaultdict
import pandas as pd

class TopRecommender:
    def __init__(self, status_id: List[int] = [11, 18]):
        self.status_id = status_id
        
    def fit(self, orders: pd.DataFrame):
        self.chains_to_cnt = (
            orders[orders.status_id.isin(self.status_id)]
            .groupby("chain_id", sort=False)["order_id"]
            .size()
            .to_dict()
        )
        self.chains_to_cnt = defaultdict(int, self.chains_to_cnt)
        
        return self

    def predict(self, chain_ids: List[int]):
        
        return [self.chains_to_cnt[chain_id] for chain_id in chain_ids]

    def __getstate__(self):
        # print("I'm being pickled")
        return self.__dict__
    
    def __setstate__(self, d):
        # print("I'm being unpickled with these values: " + repr(d))
        self.__dict__ = d