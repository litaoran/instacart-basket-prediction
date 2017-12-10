import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

order_id = np.load('data/order_id.npy')
product_id = np.load('data/product_id.npy')
features = np.load('data/features.npy')
feature_names = np.load('data/feature_names.npy')
label = np.load('data/label.npy')

df = pd.DataFrame(data=features, columns=feature_names)
df['order_id'] = order_id
df['product_id'] = product_id
df['label'] = label

training_df = df.loc[df['label'] != -1]
test_df = df.loc[df['label'] == -1]


# training
train_df, val_df = train_test_split(training_df, train_size=0.9)

X_train = train_df[feature_names].values
y_train = train_df['label'].values

X_test = val_df[feature_names].values
y_test = val_df['label'].values


# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=2000)


print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')

# predict
y_pred = gbm.predict(test_df[feature_names].values, num_iteration=gbm.best_iteration)



dirname = 'predictions_gbm'
if not os.path.isdir(dirname):
    os.makedirs(dirname)

np.save(os.path.join(dirname, 'order_ids.npy'), test_df['order_id'])
np.save(os.path.join(dirname, 'product_ids.npy'), test_df['product_id'])
np.save(os.path.join(dirname, 'predictions.npy'), y_pred)
np.save(os.path.join(dirname, 'labels.npy'), test_df['label'])
