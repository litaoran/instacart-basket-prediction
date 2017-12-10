import os

import numpy as np
import pandas as pd


product_df = pd.read_csv('../../data/processed/product_data.csv', usecols=['user_id', 'product_id', 'label'])
products = pd.read_csv('../../data/raw/products.csv')
product_df = product_df.merge(products, how='left', on='product_id')

orders = pd.read_csv('../../data/raw/orders.csv')
orders = orders[orders['eval_set'].isin({'train', 'test'})]

# add the last order of the user(either is a train or test)
product_df = product_df.merge(orders[['user_id', 'order_id']], how='left', on='user_id').reset_index(drop=True)
product_df['is_none'] = (product_df['product_id'] == 0).astype(int)


# feature representations
prefix = 'rnn_product'
h_df = pd.DataFrame(np.load('../rnn_product/predictions/final_states.npy')).add_prefix('{}_h'.format(prefix))
h_df['user_id'] = np.load('../rnn_product/predictions/user_ids.npy')
h_df['product_id'] = np.load('../rnn_product/predictions/product_ids.npy')
product_df = product_df.merge(h_df, how='left', on=['user_id', 'product_id'])

prefix = 'rnn_aisle'
h_df = pd.DataFrame(np.load('../rnn_aisle/predictions/final_states.npy')).add_prefix('{}_h'.format(prefix))
h_df['user_id'] = np.load('../rnn_aisle/predictions/user_ids.npy')
h_df['aisle_id'] = np.load('../rnn_aisle/predictions/aisle_ids.npy')
h_df['{}_prediction'.format(prefix)] = np.load('../rnn_aisle/predictions/predictions.npy')
product_df = product_df.merge(h_df, how='left', on=['user_id', 'aisle_id']).fillna(-1)

prefix = 'rnn_department'
h_df = pd.DataFrame(np.load('../rnn_department/predictions/final_states.npy')).add_prefix('{}_h'.format(prefix))
h_df['user_id'] = np.load('../rnn_department/predictions/user_ids.npy')
h_df['department_id'] = np.load('../rnn_department/predictions/department_ids.npy')
h_df['{}_prediction'.format(prefix)] = np.load('../rnn_department/predictions/predictions.npy')
product_df = product_df.merge(h_df, how='left', on=['user_id', 'department_id']).fillna(-1)


drop_cols = [
    'label',
    'user_id',
    'product_id',
    'order_id',
    'product_name',
    'aisle_id',
    'department_id',
]

user_id = product_df['user_id']
product_id = product_df['product_id']
order_id = product_df['order_id']
label = product_df['label']

product_df.drop(drop_cols, axis=1, inplace=True)
features = product_df.values
feature_names = product_df.columns.values
feature_maxs = features.max(axis=0)
feature_mins = features.min(axis=0)
feature_means = features.mean(axis=0)

if not os.path.isdir('data'):
    os.makedirs('data')

np.save('data/user_id.npy', user_id)
np.save('data/product_id.npy', product_id)
np.save('data/order_id.npy', order_id)
np.save('data/features.npy', features)
np.save('data/feature_names.npy', product_df.columns)
np.save('data/feature_maxs.npy', feature_maxs)
np.save('data/feature_mins.npy', feature_mins)
np.save('data/feature_means.npy', feature_means)
np.save('data/label.npy', label)
