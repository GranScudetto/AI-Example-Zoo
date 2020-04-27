import numpy as np


def one_hot_encoding(y, nb_classes) -> np.ndarray:
    """
    Converts decimal encoded class information into one hot encoding
    hence f.e. if we have 4 classes the conversion will look as follows
    cls 0 -> 1 0 0 0
    cls 1 -> 0 1 0 0
    ...
    cls 3 -> 0 0 0 1

    :param y: the decimal encoded class information
    :param nb_classes: the number of classes
    :return: encoded class information
    """
    one_hot_enc = np.zeros((len(y), nb_classes))
    for idx, val in enumerate(y):
        one_hot_enc[idx][val] = 1
    # sanity check
    assert int(np.sum(one_hot_enc)) == len(y), 'Sanity Check Failed ' + \
        str(np.sum(one_hot_enc)) + ' does not match ' + str(len(y))
    return one_hot_enc
