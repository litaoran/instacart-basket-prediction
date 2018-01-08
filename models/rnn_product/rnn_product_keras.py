from pandas import DataFrame
from pandas import concat

import tensorflow as tf

import pandas as pd
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
    'history_length',
    # 'product_name_length',
    'eval_set',
    'label'
]

data_cols2 = [
    'is_ordered_history',
    'index_in_order_history',
    'order_dow_history',
    'order_hour_history',
    'days_since_prior_order_history',
    'order_size_history',
    'reorder_size_history',
    # 'product_name',
    'order_number_history'
]

data_dir = './data'
# data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]
#
# product_data = DataFrame()
# for i in data_cols:
#     product_data[i] = np.load(os.path.join(data_dir, '{}.npy'.format(i)))
#
# for i in data_cols2:
#     product_data[i] = np.load(os.path.join(data_dir, '{}.npy'.format(i))).tolist()
#
# for i in data_cols2:
#     tmp_data = np.load(os.path.join(data_dir, '{}.npy'.format(i)))
#     for j in range(tmp_data.shape[1]):
#         col_name = i + '_' + str(j)
#         product_data[col_name] = tmp_data[:,j]
#
# # print 'before transpose'
# # product_data = DataFrame(data=data).transpose()
# # print 'after transpose'
#
# product_data.columns = data_cols
#
# print product_data.shape
# print product_data.dtypes
#
#
# product_data['user_id'].astype('int')
# product_data['product_id'].astype('int')
# product_data['aisle_id'].astype('int')
# product_data['department_id'].astype('int')
# product_data['is_ordered_history'].astype('int')
# product_data['index_in_order_history'].astype('int')
# product_data['order_dow_history'].astype('int')
# product_data['order_hour_history'].astype('int')
# product_data['days_since_prior_order_history'].astype('int')
# product_data['order_size_history'].astype('int')
# product_data['reorder_size_history'].astype('int')
# product_data['order_number_history'].astype('int')



user_id = np.load(os.path.join(data_dir, '{}.npy'.format('user_id')), mmap_mode='r')
product_id = np.load(os.path.join(data_dir, '{}.npy'.format('product_id')), mmap_mode='r')
aisle_id = np.load(os.path.join(data_dir, '{}.npy'.format('aisle_id')), mmap_mode='r')
department_id = np.load(os.path.join(data_dir, '{}.npy'.format('department_id')), mmap_mode='r')
is_ordered_history = np.load(os.path.join(data_dir, '{}.npy'.format('is_ordered_history')), mmap_mode='r')
index_in_order_history = np.load(os.path.join(data_dir, '{}.npy'.format('index_in_order_history')), mmap_mode='r')
order_dow_history = np.load(os.path.join(data_dir, '{}.npy'.format('order_dow_history')), mmap_mode='r')
order_hour_history = np.load(os.path.join(data_dir, '{}.npy'.format('order_hour_history')), mmap_mode='r')
days_since_prior_order_history = np.load(os.path.join(data_dir, '{}.npy'.format('days_since_prior_order_history')), mmap_mode='r')
order_size_history = np.load(os.path.join(data_dir, '{}.npy'.format('order_size_history')), mmap_mode='r')
reorder_size_history = np.load(os.path.join(data_dir, '{}.npy'.format('reorder_size_history')), mmap_mode='r')
order_number_history = np.load(os.path.join(data_dir, '{}.npy'.format('order_number_history')), mmap_mode='r')
eval_set = np.load(os.path.join(data_dir, '{}.npy'.format('eval_set')), mmap_mode='r')
label = np.load(os.path.join(data_dir, '{}.npy'.format('label')), mmap_mode='r')


print '************* finish reading data'


is_ordered_history = to_categorical(is_ordered_history, 2)
index_in_order_history = to_categorical(index_in_order_history, 20)
order_dow_history = to_categorical(order_dow_history, 8)
order_hour_history = to_categorical(order_hour_history, 25)
days_since_prior_order_history = to_categorical(days_since_prior_order_history, 31)
order_size_history = to_categorical(order_size_history, 60)
reorder_size_history = to_categorical(reorder_size_history, 50)
order_number_history = to_categorical(order_number_history, 101)




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


print type(x_history)
print x_history.shape()

y_history = product_data['label']

print x_history


print '************* finish concatenatiing data'



timesteps = 100
model = Sequential()

model.add(Masking(mask_value=0., input_shape=(x_history.shape[1], x_history.shape[2])))
model.add(LSTM(128))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='binary_crossentropy', optimizer='adam')
print(model.summary())
print '************* finish building model'



