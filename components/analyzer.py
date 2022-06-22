import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_plot(path, history, acc_or_loss="accuracy"):
    plt.figure()
    plt.plot(history[acc_or_loss])
    plt.plot(history['val_'+acc_or_loss])
    plt.title('Model_'+acc_or_loss)
    plt.ylabel(acc_or_loss)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, acc_or_loss+".png"))
    
def save_confusion_matrix(original_data, maxed_predictions,path):
    confusion = confusion_matrix(original_data, maxed_predictions)
    plt.figure()
    heatmap = sns.heatmap(confusion, annot=True)
    heatmap.figure.savefig(path)