"""
Simple confusion matrix implementation

Creation Date: April 2020
Creator: GranScudetto
"""
import matplotlib.pyplot as plt
import numpy as np


class ConfusionMatrix():
    def __init__(self, nb_classes:int, labels:list):
        self.nb_classes = nb_classes
        self.class_labels = labels
        self.intialize()

    def intialize(self):
        self.matrix = np.zeros(shape=(len(self.class_labels),
                                      len(self.class_labels)))

    def update_matrix(self, x, y):
        self.matrix[x][y] += 1.0
    
    def get_accuracy(self):
        acc = np.trace(self.matrix) / float(np.sum(self.matrix))
        miss_class = 1 - acc
        return acc, miss_class

    def get_cls_accuracies(self):
        cls_row_sum = self.matrix.sum(axis=1)
        cls_acc = np.diag(self.matrix)/cls_row_sum
        return cls_acc

    def get_cls_precision(self):
        cls_col_sum = self.matrix.sum(axis=0)
        cls_prec = np.diag(self.matrix) / cls_col_sum
        return cls_prec

    def set_matrix(self, set_values):  # only for development purpose...
        self.matrix = set_values

    def get_matrix(self, normalized=False):
        return self.matrix

    def plot_confusion_matrix(self, title: str = 'Confusion Matrix'):
        self.normed_matrix = np.zeros(shape=(len(self.class_labels),
                                      len(self.class_labels)))

        for (k, l), v in np.ndenumerate(self.matrix):
            self.normed_matrix[k][l] = (v / np.sum(self.matrix[k][:]))*100.0

        _, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(self.normed_matrix, cmap=plt.get_cmap('YlOrRd'))

        for (i, j), z in np.ndenumerate(self.matrix):
            ax.text(j, i, '{:0.0f}'.format(z) +
             '\n{:0.2f} %'.format((z / np.sum(self.matrix[i][:]))*100.0),
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


if __name__ == '__main__':
    test_matrix = ConfusionMatrix(4, ['a', 'b', 'c', 'd'])
    test_matrix.set_matrix(np.random.random((4, 4))*100)

    test_matrix.get_cls_accuracies()
    test_matrix.get_cls_precision()

    # print(test_matrix.get_matrix())
    # test_matrix.plot_confusion_matrix()
    # print(test_matrix.get_accuracy())
