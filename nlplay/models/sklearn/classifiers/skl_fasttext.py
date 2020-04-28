"""
Title   : Bag of Tricks for Efficient Text Classification
Author  : Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas
Papers  : https://arxiv.org/abs/1607.01759

A scikit-learn wrapper for Facebook FastText python interface
Inspired by work from :
Author: Evgenii Nikitin <e.nikitin@nyu.edu>
Github : https://github.com/crazyfrogspb/RedditScore
Copyright (c) 2018 Evgenii Nikitin. All rights reserved.
This work is licensed under the terms of the MIT license.
"""
import gzip
import multiprocessing
import shutil
import tempfile
from pathlib import Path
import fasttext
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class FastTextClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        lr=0.1,
        dim=100,
        ws=5,
        epoch=5,
        minCount=1,
        minCountLabel=0,
        minn=0,
        maxn=0,
        neg=5,
        wordNgrams=2,
        loss="softmax",
        bucket=2000000,
        thread=multiprocessing.cpu_count() - 1,
        lrUpdateRate=100,
        t=1e-4,
        label="__label__",
        verbose=2,
        ft_pretrained_vec_path=None,
        ft_tmp_model_name="ft_tmp_model",
    ):
        self.lr = lr
        self.dim = dim
        self.ws = ws
        self.epoch = epoch
        self.minCount = minCount
        self.minCountLabel = minCountLabel
        self.minn = minn
        self.maxn = maxn
        self.neg = neg
        self.wordNgrams = wordNgrams
        self.loss = loss
        self.bucket = bucket
        self.thread = thread
        self.lrUpdateRate = lrUpdateRate
        self.t = t
        self.label = label
        self.verbose = verbose

        self.ft_pretrained_vec_path = (
            None if ft_pretrained_vec_path is None else str(ft_pretrained_vec_path)
        )
        self.ft_tmp_model_name = ft_tmp_model_name

        self._model = None

    @property
    def label_dict(self):
        return {lbl: ix for ix, lbl in enumerate(self._model.labels)}

    def fit(self, X=None, y=None, tmp_file_path=None):  # pylint: disable=invalid-name
        """Fit, as per SKLearn"""

        # Fit model
        if tmp_file_path is None:
            assert not (X is None) and (
                y is None
            ), "Must provide X and y or a ft_formatted file."
            tmp_file_path = self._data_to_temp(X, y)

        self._model = fasttext.train_supervised(
            tmp_file_path,
            lr=self.lr,
            dim=self.dim,
            ws=self.ws,
            epoch=self.epoch,
            minCount=self.minCount,
            minCountLabel=self.minCountLabel,
            minn=self.minn,
            maxn=self.maxn,
            neg=self.neg,
            wordNgrams=self.wordNgrams,
            loss=self.loss,
            bucket=self.bucket,
            thread=self.thread,
            lrUpdateRate=self.lrUpdateRate,
            t=self.t,
            label=self.label,
            verbose=self.verbose,
            pretrainedVectors=self.ft_pretrained_vec_path,
        )
        return self

    def predict(self, X):
        # Return predicted labels
        if isinstance(X[0], list):
            docs = [" ".join(doc) for doc in X]
        elif isinstance(X[0], str):
            docs = list(X)
        else:
            raise ValueError("X has to contain sequence of tokens or strings")
        predictions = self._model.predict(docs, k=1)[0]
        return np.array([pred[0][len(self.label) :] for pred in predictions])

    def predict_proba(
        self,
        X,
        k_best: int = 1,
        threshold: float = 0.0,
        order_by_confidence: bool = True,
    ):
        """Predict and return confidence scores"""
        # Return predicted probabilities
        if isinstance(X[0], list):
            docs = [" ".join(doc) for doc in X]
        elif isinstance(X[0], str):
            docs = list(X)
        else:
            raise ValueError("X has to contain sequence of tokens or strings")

        # Launch fasttext inference process
        labels, probs = self._model.predict(docs, k_best, threshold)

    def save_model(
        self, model_path: str, quantize: bool = False, compress: bool = True
    ):
        """Save the FastText model with optional quantization and gzip compression"""
        if quantize:
            self._model.quantize()

        model_path = Path(model_path)

        if compress:
            if model_path.suffix == ".gz":
                compressed_path = model_path
                model_path = (
                    str(model_path.parents[0])
                    + model_path.stem
                    + model_path.suffixes[0]
                )
            else:
                compressed_path = model_path.with_suffixes(model_path.suffix + ".gz")

        # Save the uncompressed FT model
        self._model.save_model(str(model_path))

        if compress:
            # Compress the model
            with open(model_path, "rb") as uncompressed_model_f:
                with gzip.open(compressed_path, "wb") as compressed_model_f:
                    shutil.copyfileobj(uncompressed_model_f, compressed_model_f)

            # Remove the original file
            Path(model_path).unlink()

    def load_model(self, model_path: str):
        """Load a previously saved FastText model"""
        model_path = Path(model_path)
        suffixes = model_path.suffixes
        if suffixes[-1] == ".gz":
            with gzip.open(model_path, "rb") as compressed_model_f:
                with tempfile.NamedTemporaryFile(mode="wb") as uncompressed_model_f:
                    shutil.copyfileobj(compressed_model_f, uncompressed_model_f)
                    self._model = fasttext.load_model(uncompressed_model_f.name)
        else:
            self._model = fasttext.load_model(str(model_path))

    def _data_to_temp(self, X, y=None):
        """Dumps the given X and y matrices to a fasttext-compatible csv file.
        Parameters
        ----------
        X : array-like, shape = [n_samples]
            The input samples. An array of strings.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        """
        _, path = tempfile.mkstemp()
        with open(path, "w+") as wfile:
            for text, cls in zip(X, y):
                text = text.encode("utf8", "replace")
                wfile.write("{}{} {}\n".format(self.label, cls, text))
        return path
