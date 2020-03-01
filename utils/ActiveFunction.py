import tensorflow as tf


def Swish(x, beta=1):
    """
    Swish 函数
    :param x:
    :param beta:
    :return:
    """
    return x * tf.nn.sigmoid(x * beta)
