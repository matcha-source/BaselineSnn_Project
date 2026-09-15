from __future__ import annotations
import matplotlib.pyplot as plt

def plot_confusion_matrix(matrix) -> None:
    """ Plot confusion matrix """
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("LIF SNN Confusion matrix")
    plt.colorbar()
    plt.show()