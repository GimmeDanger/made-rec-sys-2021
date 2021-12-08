# Copyright (c) 2018, deepakn94, codyaustun, robieta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
import pandas as pd
from load import implicit_load
from feature_spec import FeatureSpec
from neumf_constants import USER_CHANNEL_NAME, ITEM_CHANNEL_NAME, LABEL_CHANNEL_NAME, TEST_SAMPLES_PER_SERIES
import torch
import pickle
import os
import tqdm

TEST_1 = 'test_data_1.pt'
TEST_0 = 'test_data_0.pt'
TRAIN_1 = 'train_data_1.pt'
TRAIN_0 = 'train_data_0.pt'
PRED_0 = 'pred_data_0.pt'

USER_MAPPING = 'user_to_index.pkl'
ITEM_MAPPING = 'chain_to_index.pkl'

USER_COLUMN = 'user_id'
ITEM_COLUMN = 'chain_id'
WEIGHT_COLUMN = 'weight'
LABEL_COLUMN = 'label'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train_path', type=str, help='Path to CSV file of train interactions')
    parser.add_argument('--pred_path', type=str, help='Path to CSV file of interactions to predict')
    parser.add_argument('--output', type=str, help='Output directory for train/test files')
    parser.add_argument('--valid_negative', type=int, default=100, help='Number of negative samples for each positive test example')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Manually set random seed for torch')
    return parser.parse_args()


class _TestNegSampler:
    def __init__(self, train_ratings, nb_neg):
        self.nb_neg = nb_neg
        self.nb_users = int(train_ratings[:, 0].max()) + 1
        self.nb_items = int(train_ratings[:, 1].max()) + 1

        # compute unique ids for quickly created hash set and fast lookup
        ids = (train_ratings[:, 0] * self.nb_items) + train_ratings[:, 1]
        self.set = set(ids)

    def generate(self, batch_size=128 * 1024):
        users = torch.arange(0, self.nb_users).reshape([1, -1]).repeat([self.nb_neg, 1]).transpose(0, 1).reshape(-1)

        items = [-1] * len(users)

        random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
        print('Generating validation negatives...')
        for idx, u in enumerate(tqdm.tqdm(users.tolist())):
            if not random_items:
                random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
            j = random_items.pop()
            while u * self.nb_items + j in self.set:
                if not random_items:
                    random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
                j = random_items.pop()

            items[idx] = j
        items = torch.LongTensor(items)
        return items


def save_feature_spec(args, user_cardinality, item_cardinality, dtypes,
                      user_feature_name='user',
                      item_feature_name='item',
                      label_feature_name='label'):

    pred_input = args.pred_path != ''
    test_negative_samples = args.valid_negative
    output_path = args.output + '/feature_spec.yaml'

    feature_spec = {
        user_feature_name: {
            'dtype': dtypes[user_feature_name],
            'cardinality': int(user_cardinality)
        },
        item_feature_name: {
            'dtype': dtypes[item_feature_name],
            'cardinality': int(item_cardinality)
        },
        label_feature_name: {
            'dtype': dtypes[label_feature_name],
        }
    }
    metadata = {
        TEST_SAMPLES_PER_SERIES: test_negative_samples + 1
    }
    train_mapping = [
        {
            'type': 'torch_tensor',
            'features': [
                user_feature_name,
                item_feature_name
            ],
            'files': [TRAIN_0]
        },
        {
            'type': 'torch_tensor',
            'features': [
                label_feature_name
            ],
            'files': [TRAIN_1]
        }
    ]
    test_mapping = [
        {
            'type': 'torch_tensor',
            'features': [
                user_feature_name,
                item_feature_name
            ],
            'files': [TEST_0],
        },
        {
            'type': 'torch_tensor',
            'features': [
                label_feature_name
            ],
            'files': [TEST_1],
        }
    ]
    pred_mapping = [
        {
            'type': 'torch_tensor',
            'features': [
                user_feature_name,
                item_feature_name,
                label_feature_name
            ],
            'files': [PRED_0],
        }
    ]
    channel_spec = {
        USER_CHANNEL_NAME: [user_feature_name],
        ITEM_CHANNEL_NAME: [item_feature_name],
        LABEL_CHANNEL_NAME: [label_feature_name]
    }
    source_spec = {'train': train_mapping, 'test': test_mapping}
    if pred_input:
        source_spec['pred'] = pred_mapping
    feature_spec = FeatureSpec(feature_spec=feature_spec, metadata=metadata, source_spec=source_spec,
                               channel_spec=channel_spec, base_directory="")
    feature_spec.to_yaml(output_path=output_path)


def prepare_train_test(args):

    print("Loading raw data from {}".format(args.train_path))
    df = pd.read_parquet(args.train_path)

    print("Mapping original user and item IDs to new sequential IDs")
    user_codes, user_uniques = pd.factorize(df[USER_COLUMN], sort=True)
    df[USER_COLUMN] = user_codes
    user_uniques = { user_id : i for i, user_id in enumerate(user_uniques)}
    with open(USER_MAPPING, 'wb') as f:
        pickle.dump(user_uniques, f)

    item_codes, item_uniques = pd.factorize(df[ITEM_COLUMN], sort=True)
    df[ITEM_COLUMN] = item_codes
    item_uniques = { item_id: i for i, item_id in enumerate(item_uniques)}
    with open(ITEM_MAPPING, 'wb') as f:
        pickle.dump(item_uniques, f)

    # clean up data
    del df[WEIGHT_COLUMN]
    df = df.drop_duplicates()  # assuming it keeps order

    # Test set is the last interaction for a given user
    grouped_sorted = df.groupby(USER_COLUMN, group_keys=False)
    test_data = grouped_sorted.tail(1).sort_values(by=USER_COLUMN)
    # Train set is all interactions but the last one
    train_data = grouped_sorted.apply(lambda x: x.iloc[:-1])
    print(len(train_data.user_id.unique()), len(test_data.user_id.unique()))

    sampler = _TestNegSampler(train_data.values, args.valid_negative)
    test_negs = sampler.generate().cuda()
    test_negs = test_negs.reshape(-1, args.valid_negative)

    # Reshape train set into user,item,label tabular and save
    train_ratings = torch.from_numpy(train_data.values).cuda()
    train_labels = torch.ones_like(train_ratings[:, 0:1], dtype=torch.float32)
    torch.save(train_ratings, os.path.join(args.output, TRAIN_0))
    torch.save(train_labels, os.path.join(args.output, TRAIN_1))

    # Reshape test set into user,item,label tabular and save
    # All users have the same number of items, items for a given user appear consecutively
    test_ratings = torch.from_numpy(test_data.values).cuda()
    test_users_pos = test_ratings[:, 0:1]  # slicing instead of indexing to keep dimensions
    test_items_pos = test_ratings[:, 1:2]
    test_users = test_users_pos.repeat_interleave(args.valid_negative + 1, dim=0)
    test_items = torch.cat((test_items_pos.reshape(-1, 1), test_negs), dim=1).reshape(-1, 1)
    positive_labels = torch.ones_like(test_users_pos, dtype=torch.float32)
    negative_labels = torch.zeros_like(test_users_pos, dtype=torch.float32).repeat(1, args.valid_negative)
    test_labels = torch.cat((positive_labels, negative_labels), dim=1).reshape(-1, 1)
    dtypes = {'user': str(test_users.dtype), 'item': str(test_items.dtype), 'label': str(test_labels.dtype)}
    test_tensor = torch.cat((test_users, test_items), dim=1)
    torch.save(test_tensor, os.path.join(args.output, TEST_0))
    torch.save(test_labels, os.path.join(args.output, TEST_1))

    return user_uniques, item_uniques, dtypes


def prepare_pred(args, user_to_index, item_to_index):
    print("Loading pred data from {}".format(args.pred_path))
    pred = pd.read_parquet(args.pred_path)
    print("pred len, uniq users, uniq items:", len(pred), len(pred[USER_COLUMN].unique()), len(pred[ITEM_COLUMN].unique()))
    pred = pred.query('user_id in @user_to_index')
    print("filtering by user_to_index")
    print("pred len, uniq users, uniq items:", len(pred), len(pred[USER_COLUMN].unique()), len(pred[ITEM_COLUMN].unique()))
    pred = pred.query('chain_id in @item_to_index')
    print("filtering by item_to_index")
    print("pred len, uniq users, uniq items:", len(pred), len(pred[USER_COLUMN].unique()), len(pred[ITEM_COLUMN].unique()))
    pred[USER_COLUMN] = pred[USER_COLUMN].map(user_to_index)
    pred[ITEM_COLUMN] = pred[ITEM_COLUMN].map(item_to_index)
    pred_data = torch.from_numpy(pred[[USER_COLUMN, ITEM_COLUMN, LABEL_COLUMN]].values).cuda()
    torch.save(pred_data, os.path.join(args.output, PRED_0))


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    user_to_index, item_to_index, dtypes = prepare_train_test(args)
    prepare_pred(args, user_to_index, item_to_index)
    save_feature_spec(args, user_cardinality=len(user_to_index), item_cardinality=len(item_to_index), dtypes=dtypes)


if __name__ == '__main__':
    main()
