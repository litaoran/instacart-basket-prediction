from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import pandas as pd
import os

class nn_model():
    def __init__(self, data_dir):

        self.get_data(data_dir)


        # self.data_dim = self.df['features'].shape[1]
        # self.NNmodel = self.build_model()

    def get_data(self, data_dir):
        print "start getting data"
        data_cols = [
            'order_id',
            'product_id',
            'features'
            'label'
        ]
        data 
        data = data.concate np.load(os.path.join(data_dir, '{}.npy'.format(i))) for i in data_cols]

        data_dict = {}
        for col in data_cols:
            print col
            data_dict['a'] = np.load(os.path.join(data_dir, '{}.npy'.format(col)))
            # data_dict[col] = data_dict[col].sample(3)
            print "finish"
            print data_dict[col].shape

        df = pd.DataFrame(data=data_dict)
        df = pd.DataFrame(columns=data_cols, data=data)

        # print df
        #
        # self.training_df = df.loc[df['label'] != -1]
        # self.test_df = df.loc[df['label'] == -1]
        # print "finishing getting data"


    def build_model(self):
        print "start for modeling building"
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_dim=self.data_dim))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print "start for model training"
        model.fit(self.df['features'].values, self.df['label'].values, epochs=1, batch_size=4096, steps_per_epoch=150, validation_split=0.1)


        print "start for evalution"
        scores = model.evaluate(self.df['features'].values, self.df['label'].values)

        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        return model


    def predict(self):
        # classes = model.predict(x_test, batch_size=128)
        return


if __name__ == '__main__':

    base_dir = './'
    data_dir=os.path.join(base_dir, 'data')
    nn = nn_model(data_dir)

    # prediction_dir=os.path.join(base_dir, 'predictions_nn')
    #
    # model = nn.build_model();




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
