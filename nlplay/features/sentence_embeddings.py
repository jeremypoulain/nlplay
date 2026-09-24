"""
Title    : A Simple but Tough-to-Beat Baseline for Sentence Embeddings (SIF) - 2017
Authors  : Sanjeev Arora, Yingyu Liang, Tengyu Ma
Papers   : https://openreview.net/forum?id=SyK00v5xx
Source   : https://github.com/PrincetonML/SIF

Title    : Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline (uSIF) - 2018
Authors  : Kawin Ethayarajh
Papers   : https://www.aclweb.org/anthology/W18-3012.pdf
Source   : https://github.com/kawine/usif

Note     : Sentence embeddings as a weighted average of word vectors followed by the removal of
           the common components, computed on the fit corpus and applied to any new sentence.
           SIF : weight(w) = a / (a + p(w)), projections on the first principal components removed.
           uSIF: weight(w) = a / (a / 2 + p(w)) with a derived from the vocabulary size and the
                 expected sentence length n, projections on the first m principal components
                 removed with weights lambda_i proportional to the squared singular values.
"""
from collections import Counter
from collections.abc import Callable, Iterable, Mapping

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted


def word_probabilities(word_counts: Mapping[str, float]) -> dict[str, float]:
    """
    :param word_counts: word → count (or frequency) mapping, e.g. from a large reference corpus.
    :returns: word → unigram probability mapping.
    """
    total = float(sum(word_counts.values()))
    if total <= 0:
        raise ValueError("word_counts must contain positive counts")
    return {w: c / total for w, c in word_counts.items()}


def sif_weights(probs: np.ndarray, a: float = 1e-3) -> np.ndarray:
    """
    SIF word weights a / (a + p(w)), words without a probability (p = 0) get a weight of 1.
    :param probs: unigram probabilities.
    :param a: smoothing parameter, 1e-3 to 1e-4 in the paper.
    :returns: the word weights.
    """
    return a / (a + np.asarray(probs, dtype=np.float64))


def usif_a(probs: np.ndarray, n: int = 11) -> float:
    """
    uSIF smoothing parameter a = (1 - alpha) / (alpha * Z), with Z = |V| / 2 and alpha the share of
    words whose probability exceeds the threshold 1 - (1 - 1 / |V|)^n.
    :param probs: unigram probabilities of the whole vocabulary.
    :param n: expected random walk length, i.e. the average sentence length (about 11 for STS).
    :returns: the smoothing parameter a.
    """
    if not (isinstance(n, (int, np.integer)) and n > 0):
        raise ValueError("n must be a positive integer")
    probs = np.asarray(probs, dtype=np.float64)
    vocab_size = float(len(probs))
    threshold = 1 - (1 - 1 / vocab_size) ** n
    alpha = np.count_nonzero(probs > threshold) / vocab_size
    if alpha == 0:
        raise ValueError("No word probability exceeds the uSIF threshold, increase n or use a larger vocabulary")
    return (1 - alpha) / (alpha * 0.5 * vocab_size)


def usif_weights(probs: np.ndarray, a: float) -> np.ndarray:
    """
    :param probs: unigram probabilities.
    :param a: smoothing parameter from usif_a.
    :returns: the uSIF word weights a / (a / 2 + p(w)).
    """
    return a / (0.5 * a + np.asarray(probs, dtype=np.float64))


def common_components(
    embeddings: np.ndarray, n_components: int, n_iter: int = 5, random_state=0, algorithm: str = "randomized"
):
    """
    Top right singular vectors of the (non centered) sentence embeddings matrix.
    :param embeddings: sentence embeddings of shape (n_sentences, dim).
    :param n_components: number of components.
    :param n_iter: TruncatedSVD iterations.
    :param random_state: TruncatedSVD seed.
    :param algorithm: "randomized" (reference code) or "arpack" (exact, useful when singular values are close).
    :returns: (components of shape (n_components, dim), singular values of shape (n_components,)).
    """
    svd = TruncatedSVD(
        n_components=n_components, algorithm=algorithm, n_iter=n_iter, random_state=random_state
    ).fit(embeddings)
    return svd.components_, svd.singular_values_


def remove_components(embeddings: np.ndarray, components: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """
    :param embeddings: sentence embeddings of shape (n_sentences, dim).
    :param components: orthonormal components of shape (n_components, dim).
    :param lambdas: weight of each removed projection, 1 → full removal.
    :returns: v - sum_i lambda_i (v . u_i) u_i for every sentence embedding v.
    """
    return embeddings - (embeddings @ components.T * lambdas) @ components


class _WeightedAverageVectorizer(TransformerMixin, BaseEstimator):
    """Shared weighted average, fit and transform logic of SIF and uSIF."""

    def _tokenize(self, X: Iterable) -> list[list[str]]:
        tokenizer = self.tokenizer or str.split
        return [tokenizer(s) if isinstance(s, str) else list(s) for s in X]

    def _probabilities(self, sentences: list[list[str]]) -> dict[str, float]:
        if self.word_counts is not None:
            return word_probabilities(self.word_counts)
        # No reference counts → probabilities estimated on the fit corpus
        return word_probabilities(Counter(t for s in sentences for t in s))

    def _word_weight(self, word: str) -> float:
        raise NotImplementedError

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        return vectors

    def _average(self, sentences: list[list[str]]) -> np.ndarray:
        out = np.zeros((len(sentences), self.dim_))
        for i, tokens in enumerate(sentences):
            tokens = [t for t in tokens if t in self.word_vectors]
            # Sentences without any known word → zero vector
            if tokens:
                vectors = np.stack([np.asarray(self.word_vectors[t], dtype=np.float64) for t in tokens])
                vectors = self._normalize(vectors)
                weights = np.array([self.word_weights_.get(t) or self._word_weight(t) for t in tokens])
                out[i] = weights @ vectors / len(tokens)
        return out

    def _fit_components(self, embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def fit(self, X, y=None):
        """
        Compute the word probabilities (unless word_counts is given) and the common components.
        :param X: sentences, as strings (split with tokenizer) or lists of tokens.
        :param y: ignored.
        :returns: the fitted transformer.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """
        :param X: sentences, as strings (split with tokenizer) or lists of tokens.
        :param y: ignored.
        :returns: sentence embeddings of shape (n_sentences, dim), identical to fit(X).transform(X).
        """
        sentences = self._tokenize(X)
        if not sentences:
            raise ValueError("X must contain at least one sentence")
        self.dim_ = len(next(iter(self.word_vectors.values())))
        self.word_probs_ = self._probabilities(sentences)
        self._fit_weights()
        embeddings = self._average(sentences)
        self._fit_components(embeddings)
        return remove_components(embeddings, self.components_, self.lambdas_)

    def transform(self, X):
        """
        :param X: sentences, as strings (split with tokenizer) or lists of tokens.
        :returns: sentence embeddings of shape (n_sentences, dim).
        """
        check_is_fitted(self, "components_")
        return remove_components(self._average(self._tokenize(X)), self.components_, self.lambdas_)


class SIFVectorizer(_WeightedAverageVectorizer):
    def __init__(
        self,
        word_vectors: Mapping[str, np.ndarray],
        word_counts: Mapping[str, float] | None = None,
        a: float = 1e-3,
        n_components: int = 1,
        tokenizer: Callable[[str], list[str]] | None = None,
        random_state: int | None = 0,
        svd_algorithm: str = "randomized",
    ):
        """
        SIF sentence embeddings, Arora et al. (2017).
        :param word_vectors: word → vector mapping, e.g. a dict or gensim KeyedVectors.
        :param word_counts: word → count mapping from a large corpus, estimated on the fit corpus if None.
            Words without a count get a weight of 1, as in the reference code.
        :param a: smoothing parameter of the weights a / (a + p(w)).
        :param n_components: number of principal components removed, 0 → weighted average only.
        :param tokenizer: callable splitting a string sentence, str.split if None.
        :param random_state: seed of the TruncatedSVD.
        :param svd_algorithm: "randomized" as in the reference code, or "arpack" for exact components.
        """
        self.word_vectors = word_vectors
        self.word_counts = word_counts
        self.a = a
        self.n_components = n_components
        self.tokenizer = tokenizer
        self.random_state = random_state
        self.svd_algorithm = svd_algorithm

    def _fit_weights(self) -> None:
        words = list(self.word_probs_)
        self.word_weights_ = dict(zip(words, sif_weights([self.word_probs_[w] for w in words], self.a)))

    def _word_weight(self, word: str) -> float:
        return 1.0

    def _fit_components(self, embeddings: np.ndarray) -> None:
        if self.n_components > 0:
            self.components_, _ = common_components(
                embeddings, self.n_components, 7, self.random_state, self.svd_algorithm
            )
        else:
            self.components_ = np.zeros((0, self.dim_))
        self.lambdas_ = np.ones(len(self.components_))


class USIFVectorizer(_WeightedAverageVectorizer):
    def __init__(
        self,
        word_vectors: Mapping[str, np.ndarray],
        word_counts: Mapping[str, float] | None = None,
        n: int = 11,
        m: int = 5,
        normalize: str = "reference",
        tokenizer: Callable[[str], list[str]] | None = None,
        random_state: int | None = 0,
        svd_algorithm: str = "randomized",
    ):
        """
        uSIF sentence embeddings, Ethayarajh (2018).
        :param word_vectors: word → vector mapping, e.g. a dict or gensim KeyedVectors.
        :param word_counts: word → count mapping from a large corpus, estimated on the fit corpus if None.
            Words without a count get the smallest probability, as in the reference code.
        :param n: expected random walk length, i.e. the average sentence length (about 11 for STS).
        :param m: number of common discourse vectors removed, 0 → weighted average only.
        :param normalize: word vectors normalization before averaging.
            "reference" → each dimension divided by its L2 norm over the sentence words, as in the
            reference code that produced the paper results, "l2" → unit length word vectors, "none".
        :param tokenizer: callable splitting a string sentence, str.split if None.
        :param random_state: seed of the TruncatedSVD.
        :param svd_algorithm: "randomized" as in the reference code, or "arpack" for exact components.
        """
        self.word_vectors = word_vectors
        self.word_counts = word_counts
        self.n = n
        self.m = m
        self.normalize = normalize
        self.tokenizer = tokenizer
        self.random_state = random_state
        self.svd_algorithm = svd_algorithm

    def _fit_weights(self) -> None:
        if self.normalize not in ("reference", "l2", "none"):
            raise ValueError(f"Unknown normalize: {self.normalize}")
        words = list(self.word_probs_)
        probs = np.array([self.word_probs_[w] for w in words])
        self.a_ = usif_a(probs, self.n)
        self.min_prob_ = probs.min()
        self.word_weights_ = dict(zip(words, usif_weights(probs, self.a_)))

    def _word_weight(self, word: str) -> float:
        return float(usif_weights(self.min_prob_, self.a_))

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if self.normalize == "reference":
            norms = np.linalg.norm(vectors, axis=0)
        elif self.normalize == "l2":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        else:
            return vectors
        return vectors / np.where(norms > 0, norms, 1.0)

    def _fit_components(self, embeddings: np.ndarray) -> None:
        if self.m > 0:
            self.components_, singular_values = common_components(
                embeddings, self.m, 5, self.random_state, self.svd_algorithm
            )
            self.lambdas_ = singular_values ** 2 / (singular_values ** 2).sum()
        else:
            self.components_, self.lambdas_ = np.zeros((0, self.dim_)), np.zeros(0)
