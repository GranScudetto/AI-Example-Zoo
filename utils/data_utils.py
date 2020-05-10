import numpy as np
import tensorflow as tf
import random


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


class Normalization(object):
    """
    This class contains various normalization methods:
    - mean / standard deviation normalization
    - 0 to 255 normalization
    """

    def __init__(self):
        raise NotImplemented('Object creation not providec!')

    @staticmethod
    def normalize_mean_std_all(x: np.ndarray) -> np.ndarray:
        for i, proc_x in enumerate(x):
            x[i] = Normalization.normalize_mean_std(proc_x)

        return x.astype(np.float32)

    @staticmethod
    def normalize_0_1_all(x: np.ndarray) -> np.ndarray:
        for i, proc_x in enumerate(x):
            x[i] = Normalization.normalize_0_1(proc_x)

        return x

    @staticmethod
    def normalize_0_255_all(x: np.ndarray) -> np.ndarray:
        for i, proc_x in enumerate(x):
            x[i] = Normalization.normalize_0_255(proc_x)

        return x

    @staticmethod
    def normalize_percentile_all(x: np.ndarray, percentile: tuple) -> np.ndarray:
        for i, proc_x in enumerate(x):
            x[i] = Normalization.normalize_percentile(proc_x, percentile)

        return x

    @staticmethod
    def normalize_mean_std(x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std = x.std()

        proc_x = x - mean
        proc_x = proc_x / std

        return proc_x

    @staticmethod
    def normalize_0_1(x: np.ndarray) -> np.ndarray:
        minimum = x.min()
        proc_x = x - minimum
        proc_x = proc_x / proc_x.max()

        return proc_x

    @staticmethod
    def normalize_0_255(x: np.ndarray) -> np.ndarray:
        proc_x = Normalization.normalize_0_1(x)
        proc_x *= 255.
        return proc_x

    @staticmethod
    def normalize_percentile(x: np.ndarray, percentile: tuple) -> np.ndarray:
        minimum, maximum = np.percentile(x, percentile)
        proc_x = x - minimum
        proc_x = proc_x / (maximum - minimum)

        return proc_x


class DataAugmentation(object):
    @staticmethod
    def flip_vertical_all(images: np.ndarray, prob: float) -> np.ndarray:
        for i, image in enumerate(images):
            if random.random() < prob:
                images[i] = DataAugmentation.flip_vertical(image)

        return images

    @staticmethod
    def adjust_brightness_all(images: np.ndarray, brightness: float, prob: float) -> np.ndarray:
        for i, image in enumerate(images):
            if random.random() < prob:
                images[i] = DataAugmentation.adjust_brightness(image, brightness)

        return images

    @staticmethod
    def adjust_saturation_all(images: np.ndarray, saturation: float, prob: float) -> np.ndarray:
        for i, image in enumerate(images):
            if random.random() < prob:
                images[i] = DataAugmentation.adjust_saturation(image, saturation)

        return images

    @staticmethod
    def flip_vertical(image):
        return tf.image.flip_left_right(image)

    @staticmethod
    def adjust_brightness(image, brightness):
        return tf.image.adjust_brightness(image, brightness)

    @staticmethod
    def adjust_saturation(image, saturation):
        return tf.image.adjust_saturation(image, saturation)
