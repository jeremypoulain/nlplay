"""
Multiclass Naive Bayes SVM (NB-SVM)
https://github.com/lrei/nbsvm
Luis Rei <luis.rei@ijs.si> @lmrei - http://luisrei.com
Learns a multiclass (OneVsRest) classifier based on word ngrams.

Notes:
NBSVM code adapted from Luis Rei nbsvm Github repo,
The idea is to use SGDClassifier as the base classifier,instead of LinearSVC
This allows to easily switch between different loss functions/algorithms :
ie Logistic regression,SVM, or Modified huber... which also speeds up the training phase
"""
import numpy as np
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize, LabelBinarizer
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot


class NBSVM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        loss: str = "log",
        alpha: float = 1.0,
        c: float = 0.001,
        max_iter=10000,
        n_jobs: int = -1,
    ):
        self.alpha = alpha
        self.C = c  # regularization term
        self.loss = loss
        self.max_iter = max_iter
        self.n_jobs = n_jobs

        self.classifiers = []
        self.classes_ = None
        self.class_count_ = None
        self.ratios_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, "csr")
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)
        Y = Y.astype(np.float64)

        # Count raw events from data
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.ratios_ = np.full(
            (n_effective_classes, n_features), self.alpha, dtype=np.float64
        )
        self._compute_ratios(X, Y)

        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            classifier = SGDClassifier(
                loss=self.loss, alpha=self.C, max_iter=self.max_iter, n_jobs=self.n_jobs
            )
            Y_i = Y[:, i]
            classifier.fit(X_i, Y_i)
            self.classifiers.append(classifier)

        return self

    def predict(self, X):
        n_effective_classes = self.class_count_.shape[0]
        n_examples = X.shape[0]

        D = np.zeros((n_effective_classes, n_examples))

        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            D[i] = self.classifiers[i].decision_function(X_i)

        return self.classes_[np.argmax(D, axis=0)]

    def _compute_ratios(self, X, Y):
        """Count feature occurrences and compute ratios."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")

        self.ratios_ += safe_sparse_dot(Y.T, X)
        normalize(self.ratios_, norm="l1", axis=1, copy=False)
        row_calc = lambda r: np.log(np.divide(r, (1 - r)))
        self.ratios_ = np.apply_along_axis(row_calc, axis=1, arr=self.ratios_)
        check_array(self.ratios_)
        self.ratios_ = sparse.csr_matrix(self.ratios_)
