import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(predictions, labels, classes, title='Confusion Matrix', cmap="YlGnBu",
                          fontsize=16, symmetric=False):
    N = len(classes)
    confusion_matrix = np.zeros((N, N), dtype=np.int32)

    classed_idxs = np.arange(N)
    for i in range(len(predictions)):
        row = np.where(classed_idxs == labels[i])[0][0]
        col = np.where(classed_idxs == predictions[i])[0][0]
        confusion_matrix[row][col] += 1

    if symmetric:
        confusion_matrix = (confusion_matrix + confusion_matrix.T) / 2

    sns.set(font_scale=1.4)
    sns.heatmap(confusion_matrix, annot=True, annot_kws={"size": fontsize}, cmap=cmap,
                fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.xticks(np.arange(N) + 0.5, classes, rotation=90)
    plt.yticks(np.arange(N) + 0.5, classes, rotation=0)
    plt.show()
    return confusion_matrix


def _plot_confusion_matrix(confusion_matrix, classes, title='Confusion Matrix', cmap="YlGnBu",
                           fontsize=16, symmetric=False):
    N = len(confusion_matrix)
    if symmetric:
        confusion_matrix = (confusion_matrix + confusion_matrix.T)/2

    sns.set(font_scale=1.4)
    sns.heatmap(confusion_matrix, annot=True, annot_kws={"size": fontsize}, cmap=cmap,
                fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.xticks(np.arange(N) + 0.5, classes, rotation=90)
    plt.yticks(np.arange(N) + 0.5, classes, rotation=0)
    plt.show()
    return confusion_matrix


