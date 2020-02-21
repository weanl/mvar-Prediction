
from keras import Model
from keras.layers import Layer

from .autoregression import AR


# override keras.layers.Layer
class CRLayer(Layer):

    def __init__(self, num_channel=3, **kwargs):
        super(CRLayer, self).__init__(**kwargs)


# override keras.Model
class UR4ML(Model):

    def __init__(self, input_shape, ur_model_path):
        super(UR4ML, self).__init__(name='UR4ML')
        self.ur_model = AR(input_shape)
        self.ur_model.load_weights(filepath=ur_model_path)

    def call(self, inputs, mask=None):
        """

        :param inputs:
        :param mask:
        :return:
        """
        return


if __name__ == '__main__':
    pass
