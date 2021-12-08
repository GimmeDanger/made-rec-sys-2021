import pandas as pd

class H3Index:
    def __init__(self, h3_to_chains_path):
        self.h3_to_chains = pd.read_pickle(h3_to_chains_path)
        self.valid = set([x for x in self.h3_to_chains.keys()])
        self.h3_to_index = {h3: i for i, h3 in enumerate(self.h3_to_chains.keys())}
        self.r_h3_to_index = {i: h3 for i, h3 in enumerate(self.h3_to_chains.keys())}
    
    def filter_by(self, h3, chains):
        return self.h3_to_chains[h3].intersection(chains)
