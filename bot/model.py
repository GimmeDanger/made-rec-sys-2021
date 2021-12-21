import pickle


class Model:
    def __init__(self):
        with open('data/h3_to_chains.pkl', 'rb') as f:
            self.h3_to_chains = pickle.load(f)
        with open('data/h3_to_city_id.pkl', 'rb') as f:
            self.h3_to_city_id = pickle.load(f)
        with open('data/lightfm_moscow.pkl', 'rb') as f:
            self.lightfm_moscow = pickle.load(f)
        with open('data/interactions_moscow.pkl', 'rb') as f:
            self.interactions_moscow = pickle.load(f)
        with open('data/user_features_sparse_moscow.pkl', 'rb') as f:
            self.user_features_sparse_moscow = pickle.load(f)
        with open('data/lightfm_piter.pkl', 'rb') as f:
            self.lightfm_piter = pickle.load(f)
        with open('data/interactions_piter.pkl', 'rb') as f:
            self.interactions_piter = pickle.load(f)
        with open('data/user_features_sparse_piter.pkl', 'rb') as f:
            self.user_features_sparse_piter = pickle.load(f)
        with open('data/lightfm_other.pkl', 'rb') as f:
            self.lightfm_other = pickle.load(f)
        with open('data/interactions_other.pkl', 'rb') as f:
            self.interactions_other = pickle.load(f)
        with open('data/user_features_sparse_other.pkl', 'rb') as f:
            self.user_features_sparse_other = pickle.load(f)
        with open('data/top_rec_moscow.pkl', 'rb') as f:
            self.top_rec_moscow = pickle.load(f)
        with open('data/top_rec_piter.pkl', 'rb') as f:
            self.top_rec_piter = pickle.load(f)
        with open('data/top_rec_other.pkl', 'rb') as f:
            self.top_rec_other = pickle.load(f)
    
    def _get_lightfm(self, city_id):
        if city_id == 1:
            return self.lightfm_moscow
        elif city_id == 2:
            return self.lightfm_piter
        else:
            return self.lightfm_other
    
    def _get_interactions(self, city_id):
        if city_id == 1:
            return self.interactions_moscow
        elif city_id == 2:
            return self.interactions_piter
        else:
            return self.interactions_other
    
    def _get_user_features(self, city_id):
        if city_id == 1:
            return self.user_features_sparse_moscow
        elif city_id == 2:
            return self.user_features_sparse_piter
        else:
            return self.user_features_sparse_other
    
    def _get_top_rec(self, city_id):
        if city_id == 1:
            return self.top_rec_moscow
        elif city_id == 2:
            return self.top_rec_piter
        else:
            return self.top_rec_other

    def top_rec(self, h3, top_k=30):
        city_id = self.h3_to_city_id[h3]
        model = self._get_top_rec(city_id)
        valid_chains = self.h3_to_chains[h3]
        pred = model.predict(valid_chains)
        return [
            x for _, x in sorted(
                zip(pred, valid_chains),
                reverse=True
            )
        ][:top_k]
