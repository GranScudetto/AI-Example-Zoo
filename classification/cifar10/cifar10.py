"""
Small Example of a simple TF2 CIFAR10 Classifier

Creation Date: April 2020
Creator: GranScudetto & Mafuba09
"""
# import packages
import tensorflow as tf
import numpy as np
import copy
import os
import tqdm
import sys

# append .\AI-Example-Zoo to sys path.
sys.path.append(os.path.join(os.path.split(__file__)[0], '..', '..'))
# import self-implemented stuff
import classification.cifar10.cifar_models as cifar_models
from utils.visualization import ConfusionMatrix, TinyClassificationViewer, visualize_input_examples
from utils.data_processing import one_hot_encoding, Normalization, DataAugmentation
from utils.file_operations import get_experiment_dir, get_latest_experiment_dir

print('Using TensorFlow:', tf.version.VERSION)  # will print TensorFlow version
print(tf.config.experimental_list_devices())  # will display the device the code is running on

tf.keras.backend.clear_session()  # reset previous states


def load_cifar_data(limit=None) -> np.ndarray:
    """
    :param limit:
    :return:
    """
    # cifar10 data (integrated in TensorFlow, downloaded on first use)
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


def save_classifier(classifier: tf.keras.Model, out_dir: str) -> None:
    """
    This function will create a folder called "saved_model" within the out_dir. This folder will contain the passed
    classifier saved in various formats, these include a json architecture file, a tf model structure,
    a h5 model file and a h5 weight only file. These can later easily be used to restore the model.

    :param classifier: Model to store
    :param out_dir: path where to store
    """
    json_architecture = classifier.to_json()  # will serialize the model structure into json format
    with open(os.path.join(out_dir, 'saved_model', 'model_config.json'), 'w') as json_file:
        json_file.write(json_architecture)  # writes model architecture to disk

    # storing the model and its weights in different formats
    classifier.save(os.path.join(out_dir, 'saved_model', 'tf_model'), save_format='tf')  # TensorFlow format
    classifier.save(os.path.join(out_dir, 'saved_model', 'model.h5'))  # weights + model in h5 format
    classifier.save_weights(os.path.join(out_dir, 'saved_model', 'model_weights.h5'))  # weights only


def get_x_predictions(classifier: tf.keras.Model, nb_predictions: int, data: np.ndarray) -> np.ndarray:
    """

    :param classifier: The classifier to get the predictions from
    :param nb_predictions: the number of desired predictions
    :param data: the data to apply the classifier on
    :return prediction_out: a numpy array containing the predictions
    """
    prediction_out = np.zeros(shape=(nb_predictions, 1))  # placeholder for the later predictions
    for i in range(nb_predictions):  # loop samples
        prediction_out[i] = int(np.argmax(classifier.predict(data[i, :, :, :].reshape(1, 32, 32, 3))))  # prediction
    return prediction_out  # result


def load_trained_weights(experiment_dir: str, model: tf.keras.Model) ->tf.keras.Model:
    """
    :param experiment_dir:
    :param model:
    """
    file_path = os.path.join(experiment_dir, 'saved_model', 'model.h5')
    if os.path.isfile(file_path):
        print('Load trained weights from {}'.format(file_path))
        model.load_weights(file_path)
        return model
    else:
        print('No trained weights! Build model from scratch!')
        return model


def evaluate(classifier, x, label, categories, ev_dir):
    confusion_matrix = ConfusionMatrix(labels=categories)

    for test_image in tqdm.tqdm(range(len(x))):
        img_to_classify = x[test_image, :, :, :].reshape(1, 32, 32, 3)
        gt = label[test_image][0]

        prediction = classifier.predict(img_to_classify)
        predicted_cls = int(np.argmax(prediction))

        confusion_matrix.update_matrix(gt, predicted_cls)

    y = one_hot_encoding(label, nb_classes=len(categories))
    test_loss, tf_acc, tf_precision, tf_t3_acc = classifier.evaluate(x, y, verbose=1)
    results = 'Results:\n' + 'TF_Loss:\t' + '%.3f' % test_loss +\
              '\tTF_Accuracy:\t' + '%.3f' % tf_acc +\
              '\tTF_Precision\t' + '%.3f' % tf_precision +\
              '\tTF_Top3 Accuracy:\t' + '%.3f' % tf_t3_acc

    results += confusion_matrix.get_complete_result_string()
    print('\n\n' + results)
    # store results
    # create evaluation folder
    if not os.path.exists(os.path.join(ev_dir, 'evaluation')):
        os.makedirs(os.path.join(ev_dir, 'evaluation'))
    # store metrics into file
    with open(os.path.join(ev_dir, 'evaluation', 'metrics.txt'), 'w') as metrics_file:
        metrics_file.write(results)

    confusion_matrix.plot_confusion_matrix()


if __name__ == '__main__':
    # categories c.f. CIFAR10
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    nb_classes = len(label_names)  # 10

    # configure training parameters
    limit, val_split = None, 0.05
    batch_size, nb_epochs = 64, 20
    optimizer, training_loss = 'adam', 'categorical_crossentropy'
    training_metrics = ['categorical_accuracy', tf.keras.metrics.Precision(),
                        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_categorical_accuracy')]
    reload_weights = False
    experiment_name = ''

    # load data
    x_train, label_train, x_test, label_test = load_cifar_data(limit=limit)
    x_test_raw = copy.deepcopy(x_test)  # for visualization no pre-processing

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
    cb_create_checks = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, 'saved_model', 'checkpoints'), monitor='val_loss', save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch'
    )
    cb_tb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'logs'))
    list_of_callbacks = [cb_create_checks, cb_tb]

    # define input shape (h x w x ch)
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    # load and compile model
    classifier = cifar_models.model_2(input_shape, nb_classes)  # c.f. cifar_models.py to see architecture definition
    classifier.summary()  # displays an overview of the architecture
    classifier = load_trained_weights(output_dir, classifier)
    classifier.compile(optimizer=optimizer, loss=training_loss, metrics=training_metrics)
    # training
    classifier.fit(x=x_train, y=y_train, epochs=nb_epochs, batch_size=batch_size, shuffle=True,
                   validation_split=val_split, callbacks=list_of_callbacks, use_multiprocessing=True)
    # save the classifier to disk
    save_classifier(classifier, output_dir)

    evaluate(classifier, x_test, label_test, label_names, output_dir)

    vis_choice = input('\nDo you want to visualize the predictions? (yes/no)')

    if vis_choice.lower() in ['yes', 'y', 'ja', 'j']:
        nb_vis_samples = int(input('How many samples do you want to view? (enter an integer number)'))

        if nb_vis_samples > len(x_train):
            nb_vis_samples = x_train

        predictions = get_x_predictions(classifier=classifier, nb_predictions=nb_vis_samples, data=x_test)

        tool = TinyClassificationViewer(data=x_test_raw, nb_samples=nb_vis_samples, ground_truth=label_test,
                                        predictions=predictions, label_names=label_names)
        tool.tiny_viewer()
