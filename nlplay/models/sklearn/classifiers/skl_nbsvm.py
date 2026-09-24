"""
Title    : Baselines and Bigrams: Simple, Good Sentiment and Topic Classification - 2012
Authors  : Sida Wang and Christopher D. Manning
Papers   : https://www.aclweb.org/anthology/P12-2018.pdf
Source   : https://github.com/lrei/nbsvm (Luis Rei)
Note     : Multiclass (One-vs-Rest) NB-SVM using SGDClassifier as the base classifier instead of
           LinearSVC, which allows to switch between losses (logistic regression, SVM, modified huber)
           and speeds up training. Each feature is scaled by its Naive Bayes log-count ratio
           r = log((p / |p|_1) / (q / |q|_1)), with p and q the smoothed feature counts of the
           class and of all the other classes.
"""
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


def _fit_binary(estimator, X, y):
    return estimator.fit(X, y)


class NBSVM(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        loss: str = "log_loss",
        alpha: float = 1.0,
        sgd_alpha: float = 0.001,
        binarize: bool = True,
        max_iter: int = 10000,
        tol: float | None = 1e-3,
        n_jobs: int | None = None,
        random_state: int | None = None,
    ):
        """
        :param loss: SGDClassifier loss, e.g. "log_loss", "hinge" or "modified_huber".
        :param alpha: additive smoothing of the Naive Bayes feature counts.
        :param sgd_alpha: SGDClassifier regularization strength, higher → stronger regularization.
        :param binarize: use 1{x > 0} instead of the raw feature values, as in the paper.
        :param max_iter: maximum number of SGD epochs.
        :param tol: SGD stopping criterion.
        :param n_jobs: number of One-vs-Rest classifiers trained in parallel.
        :param random_state: seed of the SGD shuffling, for reproducible results.
        """
        self.loss = loss
        self.alpha = alpha
        self.sgd_alpha = sgd_alpha
        self.binarize = binarize
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.input_tags.positive_only = True
        return tags

    def _preprocess(self, X):
        if (X.data if issparse(X) else X).min(initial=0) < 0:
            raise ValueError("Negative values in data passed to NBSVM, X must be non-negative")
        if self.binarize:
            X = (X > 0).astype(np.float64)
        return X

    @staticmethod
    def _scale(X, r):
        return X.multiply(r).tocsr() if issparse(X) else X * r

    def fit(self, X, y):
        """
        :param X: non-negative feature matrix of shape (n_samples, n_features), e.g. ngram counts.
        :param y: class labels of shape (n_samples,).
        :returns: the fitted estimator.
        """
        X, y = validate_data(self, X, y, accept_sparse="csr", dtype=np.float64)
        check_classification_targets(y)
        X = self._preprocess(X)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(f"NBSVM requires at least 2 classes, got {n_classes} class")

        # Feature counts per class (n_classes, n_features), then class vs rest log-count ratios
        one_hot = np.eye(n_classes)[y_idx]
        counts = np.asarray(X.T @ one_hot).T
        p = self.alpha + counts
        q = self.alpha + counts.sum(axis=0) - counts
        ratios = np.log(p / p.sum(axis=1, keepdims=True)) - np.log(q / q.sum(axis=1, keepdims=True))

        # Binary → a single classifier for the positive class, as in the paper
        positives = [1] if n_classes == 2 else range(n_classes)
        self.ratios_ = ratios[list(positives)]
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_binary)(
                SGDClassifier(
                    loss=self.loss,
                    alpha=self.sgd_alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                ),
                self._scale(X, r),
                (y_idx == k).astype(int),
            )
            for k, r in zip(positives, self.ratios_)
        )
        self.n_iter_ = max(est.n_iter_ for est in self.estimators_)
        return self

    def decision_function(self, X):
        """
        :param X: non-negative feature matrix of shape (n_samples, n_features).
        :returns: scores of shape (n_samples,) for binary problems, (n_samples, n_classes) otherwise.
        """
        check_is_fitted(self)
        X = self._preprocess(validate_data(self, X, accept_sparse="csr", dtype=np.float64, reset=False))
        scores = np.column_stack(
            [est.decision_function(self._scale(X, r)) for est, r in zip(self.estimators_, self.ratios_)]
        )
        return scores.ravel() if len(self.classes_) == 2 else scores

    def predict(self, X):
        """
        :param X: non-negative feature matrix of shape (n_samples, n_features).
        :returns: predicted class labels of shape (n_samples,).
        """
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[scores.argmax(axis=1)]

    def _has_proba(self):
        return self.loss in ("log_loss", "modified_huber")

    @available_if(_has_proba)
    def predict_proba(self, X):
        """
        Probability estimates, only for loss="log_loss" or "modified_huber".
        Multiclass → One-vs-Rest probabilities normalized to sum to 1, as in SGDClassifier.
        :param X: non-negative feature matrix of shape (n_samples, n_features).
        :returns: probabilities of shape (n_samples, n_classes).
        """
        check_is_fitted(self)
        X = self._preprocess(validate_data(self, X, accept_sparse="csr", dtype=np.float64, reset=False))
        prob = np.column_stack(
            [est.predict_proba(self._scale(X, r))[:, 1] for est, r in zip(self.estimators_, self.ratios_)]
        )
        if len(self.classes_) == 2:
            return np.column_stack([1.0 - prob[:, 0], prob[:, 0]])
        norm = prob.sum(axis=1, keepdims=True)
        # Rows where every OvR probability is 0 get a uniform distribution
        uniform = np.full_like(prob, 1.0 / prob.shape[1])
        return np.where(norm > 0, prob / np.where(norm > 0, norm, 1.0), uniform)

    @available_if(_has_proba)
    def predict_log_proba(self, X):
        """
        :param X: non-negative feature matrix of shape (n_samples, n_features).
        :returns: log probabilities of shape (n_samples, n_classes).
        """
        return np.log(self.predict_proba(X))
