
from keras import Model
from keras.layers import Layer
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import GRU
from keras.layers import Dense, Dropout, add

from autoregression import AR

import numpy as np


# override keras.Model
class nonR4ML(Model):

    def __init__(self, n_rnn_state, n_var):
        super(nonR4ML, self).__init__(name='UR4ML')
        self.n_var = n_var

        # data_format = "channels_last",  # (batch, steps, channels)
        self.conv_layer0 = Conv1D(filters=1, kernel_size=1, name='Conv1D-0')
        self.conv_layer1 = Conv1D(filters=1, kernel_size=3, name='Conv1D-1')
        self.conv_layer2 = Conv1D(filters=1, kernel_size=5, name='Conv1D-2')

        self.rnn_layer0 = GRU(units=n_rnn_state, name='GRU-0')
        self.rnn_layer1 = GRU(units=n_rnn_state, name='GRU-1')
        self.rnn_layer2 = GRU(units=n_rnn_state, name='GRU-2')

        self.ouput_layer = Dense(
            units=self.n_var,
            activation=None,
            kernel_regularizer=None,
            name='ouput_layer'
        )

    # override Model.call for implementing forward process
    def call(self, inputs, mask=None):
        """
        :param inputs: with shape of (batch_size, window, n_var)
        do convolution on n_var-dimension as Conv1D
        """

        self.conv_output0 = self.conv_layer0(inputs)
        self.conv_output1 = self.conv_layer1(inputs)
        self.conv_output2 = self.conv_layer2(inputs)

        self.rnn_output0 = self.rnn_layer0(self.conv_output0)
        self.rnn_output1 = self.rnn_layer1(self.conv_output1)
        self.rnn_output2 = self.rnn_layer2(self.conv_output2)
        self.rnn_outputs = [self.rnn_output0, self.rnn_output1, self.rnn_output2]

        self.merge_outputs = add(self.rnn_outputs)
        outputs = self.ouput_layer(self.merge_outputs)
        return outputs


def nonR4ML_test():
    model = nonR4ML(
        n_rnn_state=16,
        n_var=4
    )
    model.compile(optimizer='sgd', loss='mse')
    X = np.ones(shape=(8, 16, 4))  # (num_instance, windows, n_var)
    y = np.ones(shape=(8, 4))  # (num_instance, n_var)
    model.fit(X, y, epochs=4)
    model.summary()
    y_pred = model.predict(X)
    print(y_pred)


if __name__ == '__main__':
    nonR4ML_test()
