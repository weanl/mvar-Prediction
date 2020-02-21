
from keras.layers import Layer
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import InputLayer


# override keras.layers.Layer
class ARLayer(Layer):
    """
    only support for one-step forecast model
    """
    def __init__(self, units=1, **kwargs):
        if units != 1:
            raise ValueError('Auto-Regression only '
                             'support one-step forecast model')
        self.units = units
        super(ARLayer, self).__init__(**kwargs)
        
    def build(self, input_dim):
        self.horizon_weights = self.add_weight(
            name='horizon-weights',
            shape=(input_dim[-1], self.units),
            initializer='uniform',
            trainable=True
        )
        self.horizon_bias = self.add_weight(
            name='horizon-bias',
            shape=(self.units,),
            initializer='uniform',
            trainable=True
        )
        super(ARLayer, self).build(input_dim)

    def call(self, input):
        return tf.matmul(
         name='product',
         a=input,
         b=self.horizon_weights
        ) + self.horizon_bias


# override keras.layers.Layer
class AR(Model):
    """

    """
    def __init__(self, input_shape):
        super(AR, self).__init__(name='AR-model')
        self.ar_layer = ARLayer(
            name='ar-layer',
            units=1,
            input_shape=input_shape
        )

    def call(self, inputs, mask=None):
        outputs = self.ar_layer(inputs)
        return outputs


if __name__ == '__main__':
    # model = Sequential()
    # model.add(ARLayer(units=1, input_shape=(32,)))
    # model.summary()
    model = AR(input_shape=(32,))
    pass
