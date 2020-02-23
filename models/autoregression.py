
from keras.layers import Layer
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import InputLayer

import numpy as np


# override keras.layers.Layer
class ARLayer(Layer):
    """
    only support for one-step forecast model
    """
    def __init__(self, p, units=1, **kwargs):
        if units != 1:
            raise ValueError('Auto-Regression only '
                             'support one-step forecast model')
        self.p = p
        self.units = units
        super(ARLayer, self).__init__(**kwargs)
        self.build()
        
    def build(self, input_shape=None):  # build shape depends on initialization
        self.horizon_weights = self.add_weight(
            name='horizon-weights',
            shape=(self.p, self.units),
            initializer='uniform',
            trainable=True
        )
        self.horizon_bias = self.add_weight(
            name='horizon-bias',
            shape=(self.units,),
            initializer='uniform',
            trainable=True
        )
        super(ARLayer, self).build(input_shape)

    def call(self, input):
        return tf.matmul(
         name='product',
         a=input,
         b=self.horizon_weights
        ) + self.horizon_bias


# override keras.Model
class AR(Model):
    """

    """
    def __init__(self, p):
        super(AR, self).__init__(name='AR-model')
        self.ar_layer = ARLayer(
            name='ar-layer',
            p=p,
            units=1
        )

    # define forward process
    def call(self, inputs, mask=None):
        outputs = self.ar_layer(inputs)
        return outputs

    def predict(self, x,
                batch_size=None,
                verbose=0,
                steps=None):
        """
        do some preprocess like noise-smoothing
        """
        return super(AR, self).predict(x)


def AR_test():
    model = AR(p=32)
    model.compile(optimizer='sgd', loss='mse')
    X = np.ones(shape=(4,32))  # (num_instance, window)
    y = np.ones(shape=(4,))
    model.fit(X, y, epochs=100)
    model.summary()
    y_pred = model.predict(X)
    print(y_pred)


if __name__ == '__main__':
    # model = Sequential()
    # model.add(ARLayer(units=1, input_shape=(32,)))
    # model.summary()
    AR_test()
    pass
