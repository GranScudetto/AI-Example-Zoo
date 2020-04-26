"""
Additional Activation functions not yet present in tensorflow

Creation Date: April 2020
Creator: GranScudetto
"""

import tensorflow as tf


def mish_activation(x):
    """
    Mish activation function
    
    as described in:
    "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    https://arxiv.org/abs/1908.08681

    formula: mish(x) = x * tanh(ln(1 + exp(x)))
                     = x * tanh(softplus(x))

    """
    return (x * tf.math.tanh(tf.math.softplus(x)))


def swish_activation(x):
    """
    Swish activation function (currently only in tf-nightly)

    as described in:
    "Searching for Activation Functions"
    https://arxiv.org/abs/1710.05941

    formula: swish(x) = x* sigmoid(x)

    """

    return(x * tf.math.sigmoid(x))


tf.keras.utils.get_custom_objects().update(
    {'custom_activation': (tf.keras.layers.Activation(mish_activation),
                           tf.keras.layers.Activation(swish_activation))
     }
)
