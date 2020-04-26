"""
Small Example of a simple TF2 CIFAR10 Classifier
Creation Date: April 2020
Creator: GranScudetto
"""
# import packages
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tqdm
# import self-implemented stuff
from utils.clf_vis_confusion_matrix import ConfusionMatrix

print('Using Tensorflow:', tf.version.VERSION)


tf.keras.backend.clear_session()  # reset previous states
# category names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']


def load_cifar_data(limit = None) -> np.ndarray:
    """
    
    """
    # cifar10 data (integrated in tensorflow, downloaded on first use)
    cifar10_data = tf.keras.datasets.cifar10
    # split into training and test data
    train_data, test_data = cifar10_data.load_data()
    # split dta into image and label (not yet desired format c.f. PreProc)
    x_train, label_train = train_data
    x_test, label_test = test_data

    if limit is not None:  # optional limit to develop/test faster
        x_train = x_train[:limit, :, :, :]
        label_train = label_train[:limit]
        x_test = x_test[:limit, :, :, :]
        label_test = label_test[:limit]
    
    # provide some basic information about data
    print('Number of images in training set', len(x_train))
    print('Number of images in testing set', len(x_test))
    print('Input image size', x_train.shape[1], 'x',
      x_train.shape[2], 'in', x_train.shape[-1], 'channels')

    return x_train, label_train, x_test, label_test


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


def visualize_input_examples(x, y) -> None:
    interpolation_method = 'spline16'  # spline16 seems to work best
    if len(x) >= 12:
        random_choice = np.random.randint(0, len(x) - 12)
        _, axs = plt.subplots(nrows=3, ncols=4, figsize=(9, 6),
                              subplot_kw={'xticks': [], 'yticks': []})

        for ax, index in zip(axs.flat, range(12)):
            ax.imshow(x_train[random_choice + index, :, :, :],
                      interpolation=interpolation_method)
            ax.set_title(
                str(label_names[int(label_train[random_choice + index])]))

    plt.show()


def preprocess_data(x, y, nb_classes) -> np.ndarray:
    proc_x = x /255.0
    proc_y = one_hot_encoding(y, nb_classes)
    return proc_x, proc_y


class Cifar10Classifier():

    def __init__(self, input_shape, nb_classes, class_names):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.classes = class_names
        self.model = self.create_model()  # initialize model
        self.model.summary() # print an overview of the architecture

    def create_model(self):
        layers = tf.keras.layers  # abbreviation to save space
        # 32 x 32
        inp = layers.Input(shape=self.input_shape)
        conv_1 = layers.Conv2D(filters=16, kernel_size=(3, 3),
                        activation='relu', padding='same')(inp)

        pool_1 = layers.MaxPool2D(pool_size=(2, 2))(conv_1)
        # 16 x 16
        conv_2 = layers.Conv2D(filters=32, kernel_size=(3, 3),
                            activation='relu', padding='same')(pool_1)
        conv_3 = layers.Conv2D(filters=64, kernel_size=(3, 3),
                            activation='relu', padding='same')(conv_2)
        pool_2 = layers.MaxPool2D(pool_size=(2, 2))(conv_3)
        # 8 x 8
        conv_4 = layers.Conv2D(filters=64, kernel_size=(3, 3),
                            activation='relu', padding='same')(pool_2)
        conv_5 = layers.Conv2D(filters=128, kernel_size=(3, 3),
                            activation='relu', padding='same')(conv_4)
        pool_3 = layers.MaxPool2D(pool_size=(2, 2))(conv_5)
        # 4 x 4
        conv_6 = layers.Conv2D(filters=128, kernel_size=(3, 3),
                            activation='relu', padding='same')(pool_3)
        conv_7 = layers.Conv2D(filters=256, kernel_size=(3, 3),
                            activation='relu', padding='same')(conv_6)
        flatten = layers.Flatten()(conv_7)
        dense_1 = layers.Dense(units=512, activation='relu')(flatten)
        out = layers.Dense(units=self.nb_classes, activation='softmax')(dense_1)

        return tf.keras.Model(inputs=inp, outputs=out)

    def train(self, optimizer, loss, metrics, x, y, nb_epochs, batch_size,
              callbacks, validation_split=0.05, multiprocessing=True):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.fit(x, y, epochs=nb_epochs, batch_size=batch_size,
                       validation_split=validation_split,
                       use_multiprocessing=multiprocessing,
                       callbacks=callbacks)
        self.save_classifier()

    def save_classifier(self):
        json_architecture = self.model.to_json()
        with open('saved_model/model_config.json', 'w') as json_file:
            json_file.write(json_architecture)

        self.model.save('./saved_model/tf_model', save_format='tf')
        self.model.save('./saved_model/model.h5')
        self.model.save_weights('./saved_model/model_weights.h5')

    def load_trained_weights(self):
        self.model.load_weights()  # todo implement

    def evaluate(self, x, label):        
        confusion_matrix = ConfusionMatrix(
            nb_classes=self.nb_classes, labels=self.classes)

        for test_image in tqdm.tqdm(range(len(x))):
            img_to_classify = x[test_image, :, :, :].reshape(1, 32, 32, 3)
            gt = label[test_image][0]

            prediction = self.model.predict(img_to_classify)
            predicted_cls = int(np.argmax(prediction))

            confusion_matrix.update_matrix(gt, predicted_cls)

        y = one_hot_encoding(label, nb_classes=self.nb_classes)
        _, acc = self.model.evaluate(x, y, verbose=1)
        my_acc, misscls = confusion_matrix.get_accuracy()
        print('Tensorflow Evaluation:', acc)
        print('Results:\nConfusion Matrix:\n', confusion_matrix.get_matrix)
        print('Accuracy', my_acc, 'MissClassifcation', misscls)
        confusion_matrix.plot_confusion_matrix()


if __name__ == '__main__':
    # training paramters
    limit = None
    nb_classes = 10
    batch_size, nb_epochs = 64, 10

    # load data
    x_train, label_train, x_test, label_test = load_cifar_data(limit=limit)

    if True:
        visualize_input_examples(x_train, label_train)

    # preprocess data (normalization/ one-hot encoding)
    x_train, y_train = preprocess_data(x_train, label_train, nb_classes)
    x_test, y_test = preprocess_data(x_test, label_test, nb_classes)

    # callbacks
    cb_checkpnt = tf.keras.callbacks.ModelCheckpoint(
        filepath='./saved_model/checkpnts', monitor='val_loss', save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch'
    )
    cb_tb = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    list_of_callbacks = [cb_checkpnt, cb_tb]

    # define input shape (h x w x ch)
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    CNN = Cifar10Classifier(input_shape=input_shape, nb_classes=nb_classes,
                            class_names=label_names)
    CNN.train(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'], x=x_train, y=y_train, nb_epochs=nb_epochs,
              batch_size=batch_size, callbacks=list_of_callbacks)

    CNN.evaluate(x_test, label_test)
