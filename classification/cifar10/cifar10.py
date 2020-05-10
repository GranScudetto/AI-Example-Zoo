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
from utils.visualization import ConfusionMatrix, TinyClassificationViewer, visualize_input_examples
from utils.data_utils import one_hot_encoding, Normalization, DataAugmentation
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


def preprocess_data(x, y, nb_classes) -> np.ndarray:
    proc_x = Normalization.normalize_mean_std_all(x)
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

        # 32, 16, 8, 4, 2
        inp = layers.Input(shape=self.input_shape)  # 32 x 32

        conv_3x3_1 = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same')(inp)
        conv_3x3_1 = layers.BatchNormalization()(conv_3x3_1)
        conv_3x3_1 = layers.Activation(activation='relu')(conv_3x3_1)

        conv_5x5_1 = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same')(inp)
        conv_5x5_1 = layers.BatchNormalization()(conv_5x5_1)
        conv_5x5_1 = layers.Activation(activation='relu')(conv_5x5_1)

        network_layer_1 = layers.Concatenate()([conv_3x3_1, conv_5x5_1])
        network_layer_1_pooled = layers.MaxPool2D(pool_size=(2, 2))(network_layer_1)  # 16x16

        conv_3x3_2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(network_layer_1_pooled)
        conv_3x3_2 = layers.BatchNormalization()(conv_3x3_2)
        conv_3x3_2 = layers.Activation(activation='relu')(conv_3x3_2)

        conv_5x5_2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(network_layer_1_pooled)
        conv_5x5_2 = layers.BatchNormalization()(conv_5x5_2)
        conv_5x5_2 = layers.Activation(activation='relu')(conv_5x5_2)

        scaled_input = layers.MaxPool2D(pool_size=(2, 2))(inp)
        conv_3x3_1_3 = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same')(scaled_input)
        conv_3x3_1_3 = layers.BatchNormalization()(conv_3x3_1_3)
        conv_3x3_1_3 = layers.Activation(activation='relu')(conv_3x3_1_3)
        conv_3x3_2_3 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(conv_3x3_1_3)
        conv_3x3_2_3 = layers.BatchNormalization()(conv_3x3_2_3)
        conv_3x3_2_3 = layers.Activation(activation='relu')(conv_3x3_2_3)

        network_layer_2 = layers.Concatenate()([conv_3x3_2, conv_5x5_2, conv_3x3_2_3])
        network_layer_2_pooled = layers.MaxPool2D(pool_size=(2, 2))(network_layer_2)  # 8x8

        conv_3x3_3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(network_layer_2_pooled)
        conv_3x3_3 = layers.BatchNormalization()(conv_3x3_3)
        conv_3x3_3 = layers.Activation(activation='relu')(conv_3x3_3)

        conv_3x3_3_3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv_3x3_2_3)
        conv_3x3_3_3 = layers.BatchNormalization()(conv_3x3_3_3)
        conv_3x3_3_3 = layers.Activation(activation='relu')(conv_3x3_3_3)

        conv_3x3_3_3 = layers.MaxPool2D(pool_size=(2, 2))(conv_3x3_3_3)
        network_layer_3 = layers.Concatenate()([conv_3x3_3, conv_3x3_3_3])
        network_layer_3_pooled = layers.MaxPool2D(pool_size=(2, 2))(network_layer_3)

        conv_3x3_4 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(network_layer_3_pooled)
        conv_3x3_4 = layers.BatchNormalization()(conv_3x3_4)
        conv_3x3_4 = layers.Activation(activation='relu')(conv_3x3_4)

        conv_3x3_5 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv_3x3_4)
        conv_3x3_5 = layers.BatchNormalization()(conv_3x3_5)
        conv_3x3_5 = layers.Activation(activation='relu')(conv_3x3_5)

        conv_3x3_6 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(conv_3x3_5)
        conv_3x3_6 = layers.BatchNormalization()(conv_3x3_6)
        conv_3x3_6 = layers.Activation(activation='relu')(conv_3x3_6)

        flattened = layers.Flatten()(conv_3x3_6)
        flattened = layers.Dense(units=128, activation='relu')(flattened)
        dense_pre_out = layers.Dense(units=self.nb_classes, activation='relu')(flattened)

        out = layers.Dense(units=self.nb_classes, activation='softmax')(dense_pre_out)

        return tf.keras.Model(inputs=inp, outputs=out)

    def train(self, optimizer, loss, metrics, x, y, nb_epochs, batch_size, callbacks, validation_split=0.05,
              multiprocessing=True):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.fit(x, y, epochs=nb_epochs, batch_size=batch_size, shuffle=True,
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
        confusion_matrix = ConfusionMatrix(labels=self.classes)

        for test_image in tqdm.tqdm(range(len(x))):
            img_to_classify = x[test_image, :, :, :].reshape(1, 32, 32, 3)
            gt = label[test_image][0]

            prediction = self.model.predict(img_to_classify)
            predicted_cls = int(np.argmax(prediction))

            confusion_matrix.update_matrix(gt, predicted_cls)

        y = one_hot_encoding(label, nb_classes=self.nb_classes)
        test_loss, tf_acc, tf_precision, tf_t3_acc = self.model.evaluate(x, y, verbose=1)
        results = 'Results:\n' + 'TF_Loss:\t' + '%.3f' % test_loss +\
                  '\tTF_Accuracy:\t' + '%.3f' % tf_acc +\
                  '\tTF_Precision\t' + '%.3f' % tf_precision +\
                  '\tTF_Top3 Accuracy:\t' + '%.3f' % tf_t3_acc

        results += confusion_matrix.get_complete_result_string()
        print('\n\n' + results)
        # store results
        # create evaluation folder
        if not os.path.exists(os.path.join(output_dir, 'evaluation')):
            os.makedirs(os.path.join(output_dir, 'evaluation'))
        # store metrics into file
        with open(os.path.join(output_dir, 'evaluation', 'metrics.txt'), 'w') as metrics_file:
            metrics_file.write(results)

        confusion_matrix.plot_confusion_matrix()

    def get_x_predictions(self, nb_predictions, data):
        prediction_out = np.zeros(shape=(nb_predictions, 1))
        for i in range(nb_predictions):
            prediction_out[i] = int(np.argmax(self.model.predict(data[i, :, :, :].reshape(1, 32, 32, 3))))
        return prediction_out


if __name__ == '__main__':
    # training parameters
    limit = 100
    nb_classes = 10
    batch_size, nb_epochs = 64, 1
    reload_weights = False
    experiment_name = ''

    # category names
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # load data
    x_train, label_train, x_test, label_test = load_cifar_data(limit=limit)
    x_test_raw = x_test  # for visualization no pre-processing

    if True:
        visualize_input_examples(x_train, label_train, label_names)

    # pre-process data (normalization/ one-hot encoding)
    x_train, y_train = preprocess_data(x_train, label_train, nb_classes)
    x_test, y_test = preprocess_data(x_test, label_test, nb_classes)

    # Create output directory
    if reload_weights:
        output_dir = get_latest_experiment_dir()
        if len(output_dir) <= 0:
            output_dir = get_experiment_dir(__file__)
    else:
        output_dir = get_experiment_dir(__file__, sub_name=experiment_name)

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
              metrics=['categorical_accuracy', tf.keras.metrics.Precision(),
                       tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_categorical_accuracy')],
              x=x_train, y=y_train, nb_epochs=nb_epochs,
              batch_size=batch_size, callbacks=list_of_callbacks)

    CNN.evaluate(x_test, label_test)

    vis_choice = input('\nDo you want to visualize the predictions? (yes/no)')

    if vis_choice.lower() in ['yes', 'y', 'ja', 'j']:
        nb_vis_samples = int(input('How many samples do you want to view? (enter an integer number)'))

        if nb_vis_samples > len(x_train):
            nb_vis_samples = x_train

        predictions = CNN.get_x_predictions(nb_predictions=nb_vis_samples, data=x_test)

        tool = TinyClassificationViewer(data=x_test_raw, nb_samples=nb_vis_samples, ground_truth=label_test,
                                        predictions=predictions, label_names=label_names)
        tool.tiny_viewer()
