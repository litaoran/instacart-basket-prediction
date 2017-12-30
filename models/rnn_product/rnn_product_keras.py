
from pandas import DataFrame
from pandas import concat

import tensorflow as tf

import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Masking

from keras.utils import to_categorical


data_cols = [
    'user_id',
    'product_id',
    'aisle_id',
    'department_id',
    'is_ordered_history',
    'index_in_order_history',
    'order_dow_history',
    'order_hour_history',
    'days_since_prior_order_history',
    'order_size_history',
    'reorder_size_history',
    'order_number_history',
    'history_length',
    'product_name',
    'product_name_length',
    'eval_set',
    'label'
]
data_dir = './data'
data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]
product_data = DataFrame(data=data).transpose()
product_data.columns = data_cols

print product_data.dtypes
print 'read!!!!'

# train_dataset = product_data.loc[product_data['label'] == 'train']
# test_dataset = product_data.loc[product_data['label'] == 'test']


num_rows = product_data.shape[0]

# product_name = np.zeros(shape=[num_rows, 30], dtype=np.int32)
# product_name_length = np.zeros(shape=[num_rows], dtype=np.int8)
# history_length = np.zeros(shape=[num_rows], dtype=np.int8)

is_ordered_history = to_categorical(product_data['is_ordered_history'], 2)
index_in_order_history = to_categorical(product_data['index_in_order_history'], 20)
order_dow_history = to_categorical(product_data['order_dow_history'], 8)
order_hour_history = to_categorical(product_data['order_hour_history'], 25)
days_since_prior_order_history = to_categorical(product_data['days_since_prior_order_history'], 31)
order_size_history = to_categorical(product_data['order_size_history'], 60)
reorder_size_history = to_categorical(product_data['reorder_size_history'], 50)
order_number_history = to_categorical(product_data['order_number_history'], 101)


x_history = tf.concat([
    is_ordered_history,
    index_in_order_history,
    order_dow_history,
    order_hour_history,
    days_since_prior_order_history,
    order_size_history,
    reorder_size_history,
    order_number_history

    # ,
    # index_in_order_history_scalar,
    # order_dow_history_scalar,
    # order_hour_history_scalar,
    # days_since_prior_order_history_scalar,
    # order_size_history_scalar,
    # reorder_size_history_scalar,
    # order_number_history_scalar,
], axis=2)

# x = tf.concat([x_history, x_product, x_user], axis=2)



print 'aaaaaaaaaaaaaaaaaaa'



timesteps = 100
model = Sequential()

model.add(Masking(mask_value=0., input_shape=(timesteps, x_history.shape[1] )))
model.add(LSTM(128))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='binary_crossentropy', optimizer='adam')
print(model.summary())


