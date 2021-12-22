import pickle
import logging

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


class Model:
    def __init__(self):
        logger.info('Model initializing')
        with open('data/h3_to_chains.pkl', 'rb') as f:
            self.h3_to_chains = pickle.load(f)
            self.h3_valid = set([x for x in self.h3_to_chains.keys()])
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

    def _user_id_by_history(self, h3, filter_set):
        logger.info('Searching user id by history')
        city_id = self.h3_to_city_id[h3]
        interactions = self._get_interactions(city_id).interaction_df
        interactions = interactions.query('chain_id in @filter_set')
        logger.info(f'Interactions len after sparsing query: {len(interactions)}')
        interactions = interactions[['user_id', 'weight']]
        interactions['weight'] = 1 # we need count of chain match 
        interactions = interactions.groupby('user_id').sum()
        interactions = interactions.reset_index()[['user_id', 'weight']]
        interactions = interactions.sort_values(by=['weight'], ascending=False)
        logger.info('Top look alike users')
        logger.info(interactions.head())
        return interactions.user_id[0]
    
    def _predict(self, h3, user_id, filter_out_set, top_k=30):

        logger.info('Prediction start')

        city_id = self.h3_to_city_id[h3]
        lightfm = self._get_lightfm(city_id)
        top_rec = self._get_top_rec(city_id)
        interactions = self._get_interactions(city_id)
        user_features_sparse = self._get_user_features(city_id)

        if h3 in self.h3_valid:
            logger.info('h3 is valid')
            valid_chains = self.h3_to_chains[h3]
            valid_chain_index = [v for k, v in interactions.chain_to_index.items() if k in valid_chains]
            if user_id in interactions.user_to_index and len(valid_chain_index) > 9:
                logger.info('user_id is valid, run lightfm')
                user_index = interactions.user_to_index[user_id]
                pred = lightfm.predict(user_index, valid_chain_index, user_features=user_features_sparse)
                top_chain_index = [x for _, x in sorted(zip(pred, valid_chain_index), reverse=True)]
                top = [interactions.index_to_chain[k] for k in top_chain_index]
                top = [k for k in top if k not in filter_out_set][:top_k]
                
            else:
                logger.info('user_id is not valid, run top_rec for h3 chains')
                pred = top_rec.predict(valid_chains)
                top = [
                    x for _, x in sorted(
                        zip(pred, valid_chains),
                        reverse=True
                    )
                ]
                top = [k for k in top if k not in filter_out_set][:top_k]
        else:
            logger.info('h3 is not valid, run top_rec for all chains')
            top = [
                k for k, v in sorted(
                    top_rec.chains_to_cnt.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
            ]
            top = [k for k in top if k not in filter_out_set][:top_k]
        return top

    def predict(self, h3, user_orders_history, top_k=30):
        filter_set = set(user_orders_history)
        user_id = self._user_id_by_history(h3, filter_set)
        return self._predict(h3, user_id, filter_set, top_k)
