
import numpy as np
from sklearn.metrics import confusion_matrix


def get_scores(cmx, metrics=['acc', 'recall']):
    """
    Gets a sklearn confusion matrix and a list of metrics 
    and prints the values of the given metrics

    :param cmx: confusion matrix
    :type cmx: np.array
    :param metrics: a list of metrics
    :type metrics: list()
    :return: (FP,FN,TP,TN)
    :rtype: tuple
    """

    FP = cmx.sum(axis=0) - np.diag(cmx)
    FN = cmx.sum(axis=1) - np.diag(cmx)
    TP = np.diag(cmx)
    TN = cmx.sum() - (FP + FN + TP)

    FP = FP[~np.isnan(FP)]
    FN = FN[~np.isnan(FN)]
    TP = TP[~np.isnan(TP)]
    TN = TN[~np.isnan(TN)]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    # Overall informedness
    INF = TPR + TNR - 1

    # F1 score
    F1 = 2*TP/(2*TP+FP+FN)
    top = ((TN + FP) * (TN + FN) + (FN + TP) * (FP + TP))
    down = (TP+FP+FN+TN)*(TP+FP+FN+TN)
    RACC = top/down
    KAPPA = (ACC-RACC)/(1-RACC)

    return {"ACC": ACC, "F1": F1, "REC": TPR, "PRE": PPV, "INF": INF, "KAPPA": KAPPA}


def get_prediction_values(cmx):
    """Gets a sklearn confusion matrix and returns the prediction values
    (FP,FN,TP,TN).

    Arguments:
    cmx {numpy.ndarray} -- [a sklearn confusion matrix]
    """

    FP = cmx.sum(axis=0) - np.diag(cmx)
    FN = cmx.sum(axis=1) - np.diag(cmx)
    TP = np.diag(cmx)
    TN = cmx.sum() - (FP + FN + TP)

    FP = FP[~np.isnan(FP)]
    FN = FN[~np.isnan(FN)]
    TP = TP[~np.isnan(TP)]
    TN = TN[~np.isnan(TN)]
    return (FP, FN, TP, TN)


def accuracy_score(y_true, y_pred):
    cmx = confusion_matrix(y_true, y_pred)
    FP, FN, TP, TN = get_prediction_values(cmx)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return ACC.mean()


def recall_score(y_true, y_pred):
    cmx = confusion_matrix(y_true, y_pred)
    FP, FN, TP, TN = get_prediction_values(cmx)
    TPR = TP/(TP+FN)
    return TPR.mean()


def f1_score(y_true, y_pred):
    cmx = confusion_matrix(y_true, y_pred)
    FP, FN, TP, TN = get_prediction_values(cmx)
    F1 = 2*TP/(2*TP+FP+FN)
    return F1.mean()


def precision_score(y_true, y_pred):
    cmx = confusion_matrix(y_true, y_pred)
    FP, FN, TP, TN = get_prediction_values(cmx)
    PPV = TP/(TP+FP)
    return PPV.mean()


def informedness_score(y_true, y_pred):
    cmx = confusion_matrix(y_true, y_pred)
    FP, FN, TP, TN = get_prediction_values(cmx)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    INF = TPR + TNR - 1
    return INF.mean()


def accuracy_scorer(estimator, X_train, y_train):
    y_pred = estimator.predict(X_train)
    cmx = confusion_matrix(y_train, y_pred)
    FP, FN, TP, TN = get_prediction_values(cmx)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return ACC.mean()
