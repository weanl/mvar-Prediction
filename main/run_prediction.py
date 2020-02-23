
from models.autoregression import AR
from keras import Model

MODE = 'train'


def main(method):
    if method == 'Auto-Regression':
        model = AR(p=32)
    elif method == ' ':
        model = Model()
    else:
        raise NameError('improper method')

    model.load_weights()
    if MODE == 'train':
        model.fit()  # save training log
        model.save()
    elif MODE == 'test':
        model.predict()  # save testing output and log
    elif MODE == 'visual':
        pass

    return


if __name__ == '__main__':
    pass
