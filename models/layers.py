import tensorflow as tf


def weight_variable(shape, stddev=0.01):
    """
    Initialize a weight variable with given shape,
    by sampling randomly from Normal(0.0, stddev^2).
    :param shape: list(int).
    :param stddev: float, standard deviation of Normal distribution for weights.
    :return weights: tf.Variable.
    """
    weights = tf.get_variable('weights', shape, tf.float32,
                              tf.random_normal_initializer(mean=0.0, stddev=stddev))
    return weights


def bias_variable(shape, value=1.0):
    """
    Initialize a bias variable with given shape,
    with given constant value.
    :param shape: list(int).
    :param value: float, initial value for biases.
    :return biases: tf.Variable.
    """
    biases = tf.get_variable('biases', shape, tf.float32,
                             tf.constant_initializer(value=value))
    return biases


def conv2d(x, W, stride, padding='SAME'):
    """
    Compute a 2D convolution from given input and filter weights.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param W: tf.Tensor, shape: (fh, fw, ic, oc).
    :param stride: int, the stride of the sliding window for each dimension.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :return: tf.Tensor.
    """
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, side_l, stride, padding='SAME'):
    """
    Performs max pooling on given input.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, the side length of the pooling window for each dimension.
    :param stride: int, the stride of the sliding window for each dimension.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :return: tf.Tensor.
    """
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def conv_layer_no_bias(x, side_l, stride, out_depth, padding='SAME', **kwargs):
    """
    Add a new convolutional layer.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, the side length of the filters for each dimension.
    :param stride: int, the stride of the filters for each dimension.
    :param out_depth: int, the total number of filters to be applied.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :param kwargs: dict, extra arguments, including weights/biases initialization hyperparameters.
        - weight_stddev: float, standard deviation of Normal distribution for weights.
        - biases_value: float, initial value for biases.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)

    in_depth = int(x.get_shape()[-1])

    filters = weight_variable([side_l, side_l, in_depth, out_depth], stddev=weights_stddev)
  
    return conv2d(x, filters, stride, padding=padding)


def conv_layer(x, side_l, stride, out_depth, padding='SAME', **kwargs):
    """
    Add a new convolutional layer.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, the side length of the filters for each dimension.
    :param stride: int, the stride of the filters for each dimension.
    :param out_depth: int, the total number of filters to be applied.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :param kwargs: dict, extra arguments, including weights/biases initialization hyperparameters.
        - weight_stddev: float, standard deviation of Normal distribution for weights.
        - biases_value: float, initial value for biases.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)

    in_depth = int(x.get_shape()[-1])

    filters = weight_variable([side_l, side_l, in_depth, out_depth], stddev=weights_stddev)
    biases = bias_variable([out_depth], value=biases_value)
  
    return conv2d(x, filters, stride, padding=padding) + biases


def fc_layer(x, out_dim, **kwargs):
    """
    Add a new fully-connected layer.
    :param x: tf.Tensor, shape: (N, D).
    :param out_dim: int, the dimension of output vector.
    :param kwargs: dict, extra arguments, including weights/biases initialization hyperparameters.
        - weight_stddev: float, standard deviation of Normal distribution for weights.
        - biases_value: float, initial value for biases.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_dim = int(x.get_shape()[-1])

    weights = weight_variable([in_dim, out_dim], stddev=weights_stddev)
    biases = bias_variable([out_dim], value=biases_value)
    return tf.matmul(x, weights) + biases

#def batch_norm(x, is_training, momentum=0.99, epsilon=0.001):
def batch_norm(x, is_training, momentum=0.9, epsilon=0.00001):
    """
    Add a new batch-normalization layer.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param is_training: bool, train mode : True, test mode : False
    :return: tf.Tensor.
    """
    x = tf.layers.batch_normalization(x, momentum=momentum, epsilon=epsilon, training=is_training)
    return x
    
