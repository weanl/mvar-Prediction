
from keras import Model
from keras.layers import Layer
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import GRU
from keras.layers import Dense, Dropout, add

from autoregression import AR

import numpy as np


# override keras.Model
class UR4ML(Model):

    def __init__(self, n_rnn_state, n_var, p_list, ar_weights_files):
        if n_var != len(p_list) or n_var != len(ar_weights_files):
            raise ValueError('n_var, length of p_list and '
                             'ar_weights_files must be equal.')
        super(UR4ML, self).__init__(name='UR4ML')
        self.n_var = n_var
        self.p_list = p_list

        # self.ar_models = [
        #     AR(p)
        #     for p in self.p_list
        # ]
        # [
        #     self.ar_models[idx].load_weights(ar_weights_files[idx])
        #     for idx in range(self.n_var)
        # ]  # initialize list of ar-models and load corresponding weights

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

    # override Model.fit for residual learning
    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):
        """
        ur4ml model is built to learn the residual of y
        y = ar_pred + residual (means: residual = y - ar_pred)
        """
        # ar_models_pred = [
        #     self.ar_models[idx].predict(
        #         x[:, -self.p_list[idx]:, idx]
        #     )
        #     for idx in range(self.n_var)
        # ]  # make auto-regression prediction with shape of (n_var, batch_size)
        # ar_models_pred = np.array(ar_models_pred).T  # (batch_size, n_var)
        # y_res = y - ar_models_pred
        y_res = y

        return super(UR4ML, self).fit(
            x=x,
            y=y_res,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
            # kwargs=kwargs
        )


def UR4ML_test():
    model = UR4ML(
        n_rnn_state=16,
        n_var=4,
        p_list=[32, 32, 32, 32],
        ar_weights_files=['', '', '', '']
    )
    model.compile(optimizer='sgd', loss='mse')
    X = np.ones(shape=(8, 16, 4))  # (num_instance, windows, n_var)
    y = np.ones(shape=(8, 4))  # (num_instance, n_var)
    model.fit(X, y, epochs=400)
    model.summary()
    y_pred = model.predict(X)
    print(y_pred)


if __name__ == '__main__':
    UR4ML_test()

