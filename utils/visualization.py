"""
Utils related to visualization of inputs, results or visualization in general

Currently includes:
- interactive prediction viewer for tiny images classification
- simple confusion matrix

Creation Date: April 2020
Creator: GranScudetto
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import ImageTk, Image


def pretty_print_matrix(matrix: np.ndarray, class_labels=None) -> str:
    """
    method will create a "pretty printed" string based on matrix data, if provided each row respectively column will be
    labeled with the corresponding label (to save space only in rows the full string is displayed, columns will stick to
    the category index.

    example:
    some data   [[ 0.43  0.34  0.23]                class a [00]    0.43    0.34    0.23
                 [ 0.21  0.41  0.38]     --->       class b [01]    0.21    0.41    0.38
                 [ 0.36  0.25  0.39]]               class c [02]    0.36    0.25    0.39
                                                                    [00]    [01]    [02]
    :param matrix: matrix to convert
    :param class_labels: optional class labels
    :return: string containing the formatted text
    """
    pre_pri_conf = ''  # string buffer
    for row in range(matrix.shape[0]):  # iterate rows
        if class_labels is not None:   # check if class label is available
            pre_pri_conf += '%15s' % class_labels[row]  # class label
            pre_pri_conf += '[%02d]' % row + '\t\t'  # add index '[X] to class label
        else:
            pre_pri_conf += 'class [%02d]' % row + '\t\t'  # if no class label stick to index
        for column in range(matrix.shape[1]):  # iterate column
            pre_pri_conf += '%3.2f' % matrix[row][column] + '\t'  # matrix content
        pre_pri_conf += '\n'  # next line, new row
    # column index
    pre_pri_conf += ' ' * (15 + 4) + '\t\t' if class_labels is not None else ' ' * 4 + '\t\t'  # blanks of equal size
    for prediction_idx in range(len(class_labels)):
        pre_pri_conf += ' [%02d]' % prediction_idx + '\t'  # index

    return pre_pri_conf  # return formatted text


def normalize_str_length(categories):
    """
    method that matches the length of each label to the longest one. The difference in length will simply be filled
    with blanks. This eases the later formatting inside the GUI texts.

    F.e. considering the classes: ['airplane', 'automobile', 'bird']
    the classes 'bird' (4) and 'airplane' (8) will be both filled up to the length of 'automobile' (10)
    => 'bird______', 'airplane__' where '_' represents a blank space

    :param categories: list of categories of unequal string length
    :return: categories with "normalized" length
    """
    maximum_length = len(max(categories, key=len))  # determine the longest category name and store the size

    for category_index in range(len(categories)):  # iterate all classes
        len_difference = maximum_length - len(categories[category_index])  # calculate the diff to the max one
        if len_difference != 0:  # if length is not equal
            categories[category_index] = categories[category_index] + ' ' * len_difference  # add blanks
    return categories


class ConfusionMatrix:
    """
    Simple Numpy implementation of a Confusion Matrix. The Confusion Matrix basically relates predicted categories to a
    label. In this implementation the rows represent the different labels and the column the predictions. Hence, an
    ideal classification system would result in a diagonal matrix (of ones if normed).

    Interpretation:
    The diagonal element divided by the sum of the corresponding row, can basically be interpreted as ACCURACY, how many
    objects of that class were detected correct.
    The diagonal element divided by the sum of the corresponding column, can basically be interpreted as PRECISION, how
    many of the predicted objects were actually of that class.
    """
    def __init__(self, labels: list):
        self.class_labels = labels  # categories
        self.nb_classes = len(self.class_labels)  # number of classes
        self.matrix = None  # placeholder for the matrix
        self.normed_matrix = None  # placeholder for a normed variant of the matrix
        self.intialize()  # will initialize the matrix will zeros

    def intialize(self):  # will initialize the matrix with zeros
        self.matrix = np.zeros(shape=(len(self.class_labels), len(self.class_labels)))

    def update_matrix(self, x, y):
        self.matrix[x][y] += 1.0

    def get_accuracy(self):  # calculates the accuracy
        acc = np.trace(self.matrix) / float(np.sum(self.matrix))
        miss_class = 1 - acc
        return acc, miss_class

    def get_cls_accuracies(self):  # calculates the class-wise accuracy
        cls_row_sum = self.matrix.sum(axis=1)
        cls_acc = np.diag(self.matrix) / cls_row_sum
        return cls_acc

    def get_cls_precision(self):  # calculates the class-wise precision
        cls_col_sum = self.matrix.sum(axis=0)
        cls_prec = np.diag(self.matrix) / cls_col_sum
        return cls_prec

    def set_matrix(self, set_values):  # only for development purpose...
        self.matrix = set_values

    def norm_matrix(self, matrix):  # normalizes the matrix
        normed_matrix = np.zeros(shape=(matrix.shape[0], matrix.shape[0]))
        for (k, l), v in np.ndenumerate(self.matrix):
            normed_matrix[k][l] = (v / np.sum(matrix[k][:])) * 100.0
        return normed_matrix

    def get_matrix(self):  # returns matrix and normed matrix
        self.normed_matrix = self.norm_matrix(self.matrix)
        return self.matrix, self.normed_matrix

    def plot_confusion_matrix(self, title: str = 'Confusion Matrix'):
        self.normed_matrix = self.norm_matrix(self.matrix)

        _, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(self.normed_matrix, cmap=plt.get_cmap('YlOrRd'))

        for (i, j), z in np.ndenumerate(self.matrix):
            ax.text(j, i, '{:0.0f}'.format(z) +
                    '\n{:0.2f} %'.format((z / np.sum(self.matrix[i][:])) * 100.0),
                    ha='center', va='center', fontsize=7)

        # formatting and legends
        plt.title(title)
        ticks = np.arange(len(self.class_labels))
        plt.xticks(ticks, self.class_labels, rotation=45)
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('Prediction')
        plt.yticks(ticks, self.class_labels)
        plt.ylabel('Label')
        plt.show()

    def get_complete_result_string(self):
        acc, miss_class = self.get_accuracy()
        acc_cl = self.get_cls_accuracies()
        precision_cl = self.get_cls_precision()
        cm, cm_n = self.get_matrix()
        str_cm = pretty_print_matrix(cm, self.class_labels)
        str_cm_n = pretty_print_matrix(cm_n, self.class_labels)

        for i, cl in enumerate(self.class_labels):
            str_classes = '%s' % (cl + '[' + str(i) + ']') if i == 0 else\
                str_classes + ', ' + '%s' % (cl + '[' + str(i) + ']')

        for k, accuracy in enumerate(acc_cl):
            str_accuracies = '%2.3f' % accuracy if k == 0 else str_accuracies + ', ' + '%2.3f' % accuracy

        for m, precision in enumerate(precision_cl):
            str_precisions = '%2.3f' % precision if m == 0 else str_precisions + ', ' + '%2.3f' % precision

        str_statistic = 'Results:\n\nAccuracy:\t' + str(acc) + '\nMissClassification-Rate:\t' + str(miss_class)
        str_statistic += '\n\nClass-wise Metrics:\nClasses:\t' + str_classes
        str_statistic += '\nAccuracy\t' + str_accuracies + '\nPrecision\t' + str_precisions
        str_statistic += '\n\nConfusion Matrix (Absolute):\n' + str_cm + '\nNormalized Matrix:\n' + str_cm_n

        return str_statistic


class TinyClassificationViewer:
    """

    """

    def __init__(self, data, label_names, nb_samples: int, ground_truth, predictions, scaling_stages=[1, 2, 4]):
        self.data = data
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.categories = label_names
        self.total_images = nb_samples
        self.scaling_res = scaling_stages
        self.initialize()

    def initialize(self):
        """

        """
        self.normalize_category_length()

    def normalize_category_length(self):
        """
        method that matches the length of each label to the longest one. The difference in length will simply be filled
        with blanks. This eases the later formatting inside the GUI texts.

        F.e. considering the classes: ['airplane', 'automobile', 'bird']
        the classes 'bird' (4) and 'airplane' (8) will be both filled up to the length of 'automobile' (10)
        => 'bird______', 'airplane__' where '_' represents a blank space

        """
        maximum_length = len(max(self.categories, key=len))  # determine the longest category name and store the size

        for category_index in range(len(self.categories)):  # iterate all classes
            len_difference = maximum_length - len(self.categories[category_index])  # calculate the diff to the max one
            if len_difference != 0:  # if length is not equal
                self.categories[category_index] = self.categories[category_index] + ' ' * len_difference  # add blanks

    def tiny_viewer(self):
        """

        """
        TkinterTinyClassificationFrame(title='Simple Tiny Classification Visualizer v.0.1',
                                       scaling_factors=self.scaling_res,
                                       tk_images=self.data[:self.total_images, :, :, :],
                                       class_categories=self.categories, ground_truth=self.ground_truth,
                                       predictions=self.predictions)


class TkinterTinyClassificationFrame:
    """

    """

    def __init__(self, title: str, scaling_factors, class_categories,
                 tk_images, ground_truth, predictions, orig_img_height=32, orig_img_width=32, ):
        self.master = tk.Tk()
        self.title = title
        self.img_width = orig_img_width
        self.img_height = orig_img_height
        self.resolution_factor = scaling_factors
        self.tk_images = tk_images
        self.classes = class_categories
        self.ground_truth = ground_truth
        self.predictions = predictions

        self.ground_truth_label, self.prediction_label = None, None
        self.button_back, self.button_forward = None, None
        self.resolution_1, self.resolution_2, self.resolution_3 = None, None, None
        self.initialize()

    def initialize(self):
        self.master.title = self.title
        self.generate_tk_image_list()
        resolution_1_txt = tk.Label(master=self.master,
                                    text=str(int(self.img_width * self.resolution_factor[0])) + 'x' + str(
                                        int(self.img_height * self.resolution_factor[0])) + 'px')  # 32 x 32
        resolution_2_txt = tk.Label(master=self.master,
                                    text=str(int(self.img_width * self.resolution_factor[1])) + 'x' + str(
                                        int(self.img_height * self.resolution_factor[1])) + 'px')  # 64 x 64
        resolution_3_txt = tk.Label(master=self.master,
                                    text=str(int(self.img_width * self.resolution_factor[2])) + 'x' + str(
                                        int(self.img_height * self.resolution_factor[2])) + 'px')  # 128x128

        resolution_1_txt.grid(row=0, column=0, columnspan=self.resolution_factor[0])
        resolution_2_txt.grid(row=0, column=self.resolution_factor[0], columnspan=self.resolution_factor[1])
        resolution_3_txt.grid(row=0, column=sum(self.resolution_factor[:2]), columnspan=self.resolution_factor[2])

        self.resolution_1 = tk.Label(image=self.tk_images[0][0])
        self.resolution_2 = tk.Label(image=self.tk_images[0][1])
        self.resolution_3 = tk.Label(image=self.tk_images[0][2])

        self.resolution_1.grid(row=1, column=0, columnspan=self.resolution_factor[0])
        self.resolution_2.grid(row=1, column=self.resolution_factor[0], columnspan=self.resolution_factor[1])
        self.resolution_3.grid(row=1, column=sum(self.resolution_factor[:2]), columnspan=self.resolution_factor[2])

        self.ground_truth_label = tk.Label(master=self.master, font=('bold'),
                                           text="Groundtruth: \t" + self.classes[int(self.ground_truth[0])])

        self.prediction_label = tk.Label(master=self.master, font=('bold'),
                                         text=" Prediction: \t" + self.classes[int(self.predictions[0])],
                                         fg='green' if self.ground_truth[0] == self.predictions[0] else 'red')

        self.ground_truth_label.grid(row=2, column=0, columnspan=sum(self.resolution_factor[:]))
        self.prediction_label.grid(row=3, column=0, columnspan=sum(self.resolution_factor[:]))

        self.button_back = tk.Button(self.master, text='<<', state=tk.DISABLED)
        self.button_forward = tk.Button(self.master, text='>>', command=lambda: self.change_image_button(image_number=2))
        button_exit = tk.Button(self.master, text='exit', command=self.master.quit)

        self.button_back.grid(row=4, column=0)
        self.button_forward.grid(row=4, column=1)
        button_exit.grid(row=4, column=2)
        self.master.mainloop()

    def change_image_button(self, image_number: int):
        self.resolution_1.grid_forget()
        self.resolution_2.grid_forget()
        self.resolution_3.grid_forget()

        self.ground_truth_label.grid_forget()
        self.prediction_label.grid_forget()

        self.resolution_1 = tk.Label(image=self.tk_images[image_number - 1][0])
        self.resolution_2 = tk.Label(image=self.tk_images[image_number - 1][1])
        self.resolution_3 = tk.Label(image=self.tk_images[image_number - 1][2])

        self.resolution_1.grid(row=1, column=0, columnspan=self.resolution_factor[0])
        self.resolution_2.grid(row=1, column=self.resolution_factor[0], columnspan=self.resolution_factor[1])
        self.resolution_3.grid(row=1, column=sum(self.resolution_factor[:2]), columnspan=self.resolution_factor[2])

        self.ground_truth_label = tk.Label(master=self.master, font=('bold'),
                                           text="Groundtruth: \t" + self.classes[
                                               int(self.ground_truth[image_number - 1])])

        self.prediction_label = tk.Label(master=self.master, font=('bold'),
                                         text=" Prediction: \t" + self.classes[int(self.predictions[image_number - 1])],
                                         fg='green' if self.ground_truth[image_number - 1] == self.predictions[
                                             image_number - 1] else 'red')

        self.ground_truth_label.grid(row=2, column=0, columnspan=sum(self.resolution_factor[:]))
        self.prediction_label.grid(row=3, column=0, columnspan=sum(self.resolution_factor[:]))

        self.button_back = tk.Button(self.master, text='<<', command=lambda: self.change_image_button(image_number - 1))
        self.button_forward = tk.Button(self.master, text='>>',
                                        command=lambda: self.change_image_button(image_number + 1))

        if image_number == len(self.tk_images):
            self.button_forward = tk.Button(self.master, text='>>', state=tk.DISABLED)
        elif image_number <= 1:
            self.button_back = tk.Button(self.master, text='<<', state=tk.DISABLED)

        self.button_back.grid(row=4, column=0)
        self.button_forward.grid(row=4, column=1)

    def get_scaled_series(self, pil_image, method=Image.LANCZOS):
        """

        """
        img_scaled_1 = pil_image.resize(size=(pil_image.width * self.resolution_factor[0],
                                              pil_image.height * self.resolution_factor[0]), resample=method)

        img_scaled_2 = pil_image.resize(size=(pil_image.width * self.resolution_factor[1],
                                              pil_image.height * self.resolution_factor[1]), resample=method)

        img_scaled_3 = pil_image.resize(size=(pil_image.width * self.resolution_factor[2],
                                              pil_image.height * self.resolution_factor[2]), resample=method)

        return [img_scaled_1, img_scaled_2, img_scaled_3]

    def generate_tk_image_list(self):
        """

        """

        tk_image_list = list()  # create a list for the later use of Tkinter

        for image in range(len(self.tk_images)):
            image_series = self.get_scaled_series(pil_image=Image.fromarray(self.tk_images[image]))
            tk_image_list.append([ImageTk.PhotoImage(image=elem) for elem in image_series])

        self.tk_images = tk_image_list


if __name__ == '__main__':
    test_interactive_prediction_viewer = False
    test_confusion_matrix = True

    if test_interactive_prediction_viewer:
        train_data, test_data = tf.keras.datasets.cifar10.load_data()  # use cifar10 as sample data
        x_train, label_train = train_data  # split data into images and labels

        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        vwr = TinyClassificationViewer(data=x_train, label_names=label_names, nb_samples=10, ground_truth=label_train,
                                       predictions=label_train)
        vwr.tiny_viewer()

    elif test_confusion_matrix:
        test_matrix = ConfusionMatrix(['airplane', 'automobile', 'bird', 'cat'])
        test_matrix.set_matrix(np.random.random((4, 4)) * 100)

        print(pretty_print_matrix(test_matrix.matrix, test_matrix.class_labels))
        # print(test_matrix.get_matrix())
        #test_matrix.plot_confusion_matrix()
        # print(test_matrix.get_accuracy())
        # print(test_matrix.get_cls_accuracies())
        # print(test_matrix.get_cls_precision())
        print(test_matrix.get_complete_result_string())