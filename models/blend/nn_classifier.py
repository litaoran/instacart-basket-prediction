from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import pandas as pd
import os

class nn_model():
    def __init__(self):

        self.df = None
        self.training_df = None
        self.test_df = None
        self.data_cols = None
        self.data_dim = 0
        self.model = None

        self.get_data()

    def get_data(self):
        self.data_cols =np.load(os.path.join(data_dir, 'feature_names.npy'))
        self.data_dim = len(self.data_cols)
        print self.data_dim

        self.df = pd.DataFrame(columns=self.data_cols, data=np.load(os.path.join(data_dir, 'features.npy')))

        data_cols = [
            'order_id',
            'product_id',
            'label'
        ]

        for col in data_cols:
            self.df[col] = np.load(os.path.join(data_dir, '{}.npy'.format(col)))

        self.training_df = self.df.loc[self.df['label'] != -1]
        self.test_df = self.df.loc[self.df['label'] == -1]

        print len(self.df)
        print len(self.training_df)
        print len(self.test_df)
        print 'finishing getting data'
        type(self.training_df['label'][0])
        # self.training_df['label'] = self.training_df['label'].apply(np.float)


    def build_model(self):
        print "start for modeling building"
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_dim=self.data_dim))
        self.model.add(Dense(units=1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.model.fit(self.training_df[self.data_cols].values, self.training_df['label'].values, epochs=1, batch_size=4096,  validation_split=0.1)

        # print "start for evalution"
        # scores = self.model.evaluate(self.df[self.data_cols].values, self.df['label'].values)
        # print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

        print 'finishing building model'

    def predict(self):

        classes = self.model.predict(self.test_df[self.data_cols].values, batch_size=4096)
        self.test_df['prediction_nn'] = classes

        np.save(os.path.join(dirname, 'order_ids.npy'), self.test_df['order_id'])
        np.save(os.path.join(dirname, 'product_ids.npy'), self.test_df['product_id'])
        np.save(os.path.join(dirname, 'predictions.npy'), self.test_df['prediction_nn'])
        np.save(os.path.join(dirname, 'labels.npy'), self.test_df['label'])

if __name__ == '__main__':
    base_dir = './'
    data_dir=os.path.join(base_dir, 'data')

    dirname = 'predictions_nn'
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    nn = nn_model()
    nn.build_model()
    nn.predict()



# # Create first network with Keras
# from keras.models import Sequential
# from keras.layers import Dense
# import numpy
# # fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)
# # load pima indians dataset
# dataset = numpy.loadtxt("../../pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # create model
# model = Sequential()
# model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
# model.add(Dense(8, init='uniform', activation='relu'))
# model.add(Dense(1, init='uniform', activation='sigmoid'))
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
# model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# # calculate predictions
# predictions = model.predict(X)
#
#
# print predictions
#
#
# # # round predictions
# # rounded = [round(x[0]) for x in predictions]
# # print len(rounded)
# # print(rounded)
