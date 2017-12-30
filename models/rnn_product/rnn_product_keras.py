
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

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
product_data = DataFrame(columns=data_cols, data=data)

train_dataset = product_data.loc[product_data['label'] == 'train']
test_dataset = product_data.loc[product_data['label'] == 'test']


num_rows = product_data.shape[0]

# product_name = np.zeros(shape=[num_rows, 30], dtype=np.int32)
# product_name_length = np.zeros(shape=[num_rows], dtype=np.int8)
# history_length = np.zeros(shape=[num_rows], dtype=np.int8)

is_ordered_history = to_categorical(product_data['is_ordered_history'].value)
index_in_order_history = to_categorical(product_data['index_in_order_history'].value)
order_dow_history = to_categorical(product_data['order_dow_history'].value)
order_hour_history = to_categorical(product_data['order_hour_history'].value)
days_since_prior_order_history = to_categorical(product_data['days_since_prior_order_history'].value)
order_size_history = to_categorical(product_data['order_size_history'].value)
reorder_size_history = to_categorical(product_data['reorder_size_history'].value)
order_number_history = to_categorical(product_data['order_number_history'].value)


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
# trainX =np.zeros((test_df.shape[0],100, None))

print 'bbbbbbbbbbbbbbbbbb'


timesteps = 100
model = Sequential()
# # # time step should be 100 for sequence

model.add(Masking(mask_value=0., input_shape=(timesteps, x_history.shape[1] )))
model.add(LSTM(512))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='binary_crossentropy', optimizer='adam')


