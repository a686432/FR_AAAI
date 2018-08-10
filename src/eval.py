import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def calROC(pred,y):

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

if __name__ == "__main__":
    y = np.array([0, 0, 1, 1])
    pred = np.array([0.1, 0.4, 0.35, 0.8])
    calROC(pred,y)