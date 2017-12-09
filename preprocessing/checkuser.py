import os

import pandas as pd


orders = pd.read_csv('../raw/orders.csv')


print(orders.loc[orders['order_id'] == 803273])



user_data = pd.read_csv('./user_data.csv')

print(user_data.loc[user_data['user_id']==206208])


