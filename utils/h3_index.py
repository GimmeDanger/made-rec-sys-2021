import pandas as pd

class H3Index:
    def __init__(self, h3_to_chains_path):
        self.h3_to_chains = pd.read_pickle(h3_to_chains_path)
        self.valid = set([x for x in self.h3_to_chains.keys()])
    
    def filter_by(self, h3, chains):
        return self.h3_to_chains[h3].intersection(chains)
