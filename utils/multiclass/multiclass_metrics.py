import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error


class MulticlassMetrics:
    """MulticlassMetrics

    Keyword arguments:
        estimator (sklearn.base.BaseEstimator): The trained classifier.
        X_test (pandas.DataFrame): The test-set dataframe.
        y_test (Series): The test-set's target.

    """

    def __init__(self, estimator, X_test, y_test):
        """MulticlassMetrics

        Keyword arguments:
            estimator (sklearn.base.BaseEstimator): The trained classifier.
            X_test (pandas.DataFrame): The test-set dataframe.
            y_test (Series): The test-set's target.

        """

        self.estimator = estimator
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = np.round(estimator.predict(X_test))
        self.cmx = confusion_matrix(self.y_test, self.y_pred)
        self.get_prediction_values()

    def get_prediction_values(self):
        """Returns the prediction values
        (self.FP,self.FN,self.TP,self.TN).
        """

        self.FP = self.cmx.sum(axis=0) - np.diag(self.cmx)
        self.FN = self.cmx.sum(axis=1) - np.diag(self.cmx)
        self.TP = np.diag(self.cmx)
        self.TN = self.cmx.sum() - (self.FP + self.FN + self.TP)

        self.FP = self.FP[~np.isnan(self.FP)]
        self.FN = self.FN[~np.isnan(self.FN)]
        self.TP = self.TP[~np.isnan(self.TP)]
        self.TN = self.TN[~np.isnan(self.TN)]

        return (self.FP, self.FN, self.TP, self.TN)

    def accuracy_score(self):
        ACC = (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)
        ACC = np.nan_to_num(ACC)
        return ACC.mean()

    def recall_score(self):
        self.TPR = self.TP / (self.TP + self.FN)
        return np.nan_to_num(self.TPR).mean()

    def f1_score(self):
        F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
        return np.nan_to_num(F1).mean()

    def f2_score(self):
        F2 = 5 * (self.precision_score() * self.recall_score()) / ((4 * self.precision_score()) + self.recall_score())
        return np.nan_to_num(F2).mean()

    def precision_score(self):
        PPV = self.TP / (self.TP + self.FP)
        PPV = np.nan_to_num(PPV)
        return PPV.mean()

    def informedness_score(self):
        # Sensitivity, hit rate, recall, or true positive rate
        self.TPR = self.TP / (self.TP + self.FN)
        # Specificity or true negative rate
        self.TNR = self.TN / (self.TN + self.FP)
        INF = self.TPR + self.TNR - 1
        return np.nan_to_num(INF).mean()

    def rmse(self):
        return np.sqrt(mean_squared_error(self.y_test, self.y_pred))

    def to_string(self):
        print('Accuracy       ' + str(self.accuracy_score()))
        print('F1-Score       ' + str(self.f1_score()))
        print('F2-Score       ' + str(self.f2_score()))
        print('Recall         ' + str(self.recall_score()))
        print('Precision      ' + str(self.precision_score()))
        print('Informedness   ' + str(self.informedness_score()))
        print('RMSE           ' + str(self.rmse()))

    def print(self):
        print("Accuracy F1-Score F2-Score Recall Precision Informedness RMSE")
        print(str(self.accuracy_score()) + ' ' + str(self.f1_score()) + ' ' + str(self.f2_score()) + ' ' + str(self.recall_score()) + ' ' + str(self.precision_score()) + ' ' + str(
                self.informedness_score()) + ' ' + str(self.rmse()))
