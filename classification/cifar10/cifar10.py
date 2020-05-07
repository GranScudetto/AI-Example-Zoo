"""
Small Example of a simple TF2 CIFAR10 Classifier

Creation Date: April 2020
Creator: GranScudetto & Mafuba09
"""
# import packages
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import tqdm
import sys

# Append .\AI-Example-Zoo to sys path.
sys.path.append(os.path.join(os.path.split(__file__)[0], '..', '..'))
# import self-implemented stuff
from utils.visualization import ConfusionMatrix, TinyClassificationViewer
from utils.data_utils import one_hot_encoding, Normalization
from utils.fileoperations import get_experiment_dir, get_latest_experiment_dir

print('Using Tensorflow:', tf.version.VERSION)

tf.keras.backend.clear_session()  # reset previous states


def load_cifar_data(limit=None) -> np.ndarray:
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


def visualize_input_examples(x, label) -> None:
    interpolation_method = 'spline16'  # spline16 seems to work best
    if len(x) >= 12:
        random_choice = np.random.randint(0, len(x) - 12)
        _, axs = plt.subplots(nrows=3, ncols=4, figsize=(9, 6),
                              subplot_kw={'xticks': [], 'yticks': []})

        for ax, index in zip(axs.flat, range(12)):
            ax.imshow(x_train[random_choice + index, :, :, :],
                      interpolation=interpolation_method)
            ax.set_title(
                str(label_names[int(label[random_choice + index])]))

    plt.show()


def preprocess_data(x, y, nb_classes) -> np.ndarray:
    proc_x = Normalization.normalize_mean_std(x)
    proc_y = one_hot_encoding(y, nb_classes)
    return proc_x, proc_y


class Cifar10Classifier:

    def __init__(self, input_shape, nb_classes, class_names):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.classes = class_names
        self.model = self.create_model()  # initialize model
        self.model.summary()  # print an overview of the architecture

    def create_model(self):
        layers = tf.keras.layers  # abbreviation to save space
        # 32 x 32
        inp = layers.Input(shape=self.input_shape)
        conv_1 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inp)

        pool_1 = layers.MaxPool2D(pool_size=(2, 2))(conv_1)
        # 16 x 16
        conv_2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(pool_1)
        conv_3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv_2)
        pool_2 = layers.MaxPool2D(pool_size=(2, 2))(conv_3)
        # 8 x 8
        conv_4 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool_2)
        conv_5 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv_4)
        pool_3 = layers.MaxPool2D(pool_size=(2, 2))(conv_5)
        # 4 x 4
        conv_6 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool_3)
        conv_7 = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv_6)
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
        with open(os.path.join(output_dir, 'saved_model', 'model_config.json'), 'w') as json_file:
            json_file.write(json_architecture)

        self.model.save(os.path.join(output_dir, 'saved_model', 'tf_model'), save_format='tf')
        self.model.save(os.path.join(output_dir, 'saved_model', 'model.h5'))
        self.model.save_weights(os.path.join(output_dir, 'saved_model', 'model_weights.h5'))

    def load_trained_weights(self):
        filepath = os.path.join(output_dir, 'saved_model', 'model.h5')
        if os.path.isfile(filepath):
            print('Load trained weights from {}'.format(filepath))
            self.model.load_weights(filepath)
        else:
            print('No trained weights! Build model from scratch!')

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
        my_acc, missed = confusion_matrix.get_accuracy()
        cls_acc, cls_prec = confusion_matrix.get_cls_accuracies(), confusion_matrix.get_cls_precision()
        results = 'Results:\n' + 'Accuracy:' + str(my_acc) + '\n' + 'Missclassification:' + str(missed) + '\n\n'\
                  + 'Classwise information:\n' + 'Clsw. Accuracies:\n' + str(cls_acc) +\
                  '\nClsw. Precision:\n' + str(cls_prec) +\
                  '\nConfusion_matrix:\n' + str(confusion_matrix.get_matrix()[0]) +\
                  '\nnormalized:\n' + np.array2string((confusion_matrix.get_matrix()[1]), precision=2)

        print(results)
        confusion_matrix.plot_confusion_matrix()

        # store results
        # create evaluation folder
        if not os.path.exists(os.path.join(output_dir, 'evaluation')):
            os.makedirs(os.path.join(output_dir, 'evaluation'))
        # store metrics into file
        with open(os.path.join(output_dir, 'evaluation', 'metrics.txt'), 'w') as metrics_file:
            metrics_file.write(results)

    def get_x_predictions(self, nb_predictions, data):
        prediction_out = np.zeros(shape=(nb_predictions, 1))
        for i in range(nb_predictions):
            prediction_out[i] = int(np.argmax(self.model.predict(data[i, :, :, :].reshape(1, 32, 32, 3))))
        return prediction_out


if __name__ == '__main__':
    # Create output directory
    output_dir = get_latest_experiment_dir()
    if len(output_dir) <= 0:
        output_dir = get_experiment_dir(__file__)

    # training parameters
    limit = 200
    nb_classes = 10
    batch_size, nb_epochs = 64, 1
    # category names
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # load data
    x_train, label_train, x_test, label_test = load_cifar_data(limit=limit)
    x_test_raw = x_test  # for visualization no preprocessing

    if True:
        visualize_input_examples(x_train, label_train)

    # pre-process data (normalization/ one-hot encoding)
    x_train, y_train = preprocess_data(x_train, label_train, nb_classes)
    x_test, y_test = preprocess_data(x_test, label_test, nb_classes)

    # callbacks
    cb_checkpnt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, 'saved_model', 'checkpnts'), monitor='val_loss', save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch'
    )
    cb_tb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'logs'))
    list_of_callbacks = [cb_checkpnt, cb_tb]

    # define input shape (h x w x ch)
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    CNN = Cifar10Classifier(input_shape=input_shape, nb_classes=nb_classes,
                            class_names=label_names)
    CNN.load_trained_weights()
    CNN.train(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'], x=x_train, y=y_train, nb_epochs=nb_epochs,
              batch_size=batch_size, callbacks=list_of_callbacks)

    CNN.evaluate(x_test, label_test)

    vis_choice = input('Do you want to visualize the predictions? (yes/no)')
    if vis_choice.lower() in ['yes', 'y', 'ja', 'j']:
        nb_vis_samples = int(input('How many samples do you want to view? (enter an integer number)'))
        if nb_vis_samples > len(x_train): nb_vis_samples = x_train

        predictions = CNN.get_x_predictions(nb_predictions=nb_vis_samples, data=x_test)

        tool = TinyClassificationViewer(data=x_test_raw, nb_samples=nb_vis_samples, ground_truth=label_test,
                                        predictions=predictions, label_names=label_names)
        tool.tiny_viewer()
