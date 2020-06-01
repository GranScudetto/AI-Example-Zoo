import numpy as np
import tensorflow as tf
import random
import math



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
    def noise_gaussian_all(images: np.ndarray, prob: float) -> np.ndarray:
        for i, image in enumerate(images):
            if random.random() < prob:
                images[i] = DataAugmentation.noise_gaussian(image)

        return images

    @staticmethod
    def noise_salt_peper_all(images: np.ndarray, prob: float) -> np.ndarray:
        for i, image in enumerate(images):
            if random.random() < prob:
                images[i] = DataAugmentation.noise_salt_peper(image)

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

    @staticmethod
    def noise_gaussian(image):
        h, w, c = image.shape
        mean = 0
        var = 0.1**0.5
        gauss = np.random.normal(mean, var, (h, w, c))
        gauss = gauss.reshape(h, w, c)
        return np.clip(image + gauss, 0, 255)

    @staticmethod
    def noise_salt_peper(image):
        h, w, c = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv


class Generator(tf.keras.utils.Sequence):
    def __init__(self, images: np.ndarray, targets: np.ndarray, batch_size: int):#, transform: list):
        targets = one_hot_encoding(targets, 10)

        self.images, self.targets = images, targets
        self.batch_size = batch_size

        #self.transform = transform

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]

        # ToDo: Make configurable
        batch_images = DataAugmentation.flip_vertical_all(batch_images, 0.6)
        batch_images = DataAugmentation.adjust_brightness_all(batch_images, 0.1, 0.6)
        batch_images = DataAugmentation.adjust_saturation_all(batch_images, 0.1, 0.6)
        batch_images = DataAugmentation.noise_gaussian_all(batch_images, 0.6)
        batch_images = Normalization.normalize_mean_std_all(batch_images)

        return batch_images, batch_targets
