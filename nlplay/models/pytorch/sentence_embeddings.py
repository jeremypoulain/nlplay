"""
Title    : A Simple but Tough-to-Beat Baseline for Sentence Embeddings (SIF) - 2017
Authors  : Sanjeev Arora, Yingyu Liang, Tengyu Ma
Papers   : https://openreview.net/forum?id=SyK00v5xx
Source   : https://github.com/PrincetonML/SIF

Title    : Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline (uSIF) - 2018
Authors  : Kawin Ethayarajh
Papers   : https://www.aclweb.org/anthology/W18-3012.pdf
Source   : https://github.com/kawine/usif

Note     : PyTorch versions working on padded batches of token ids, same weights and component
           removal as nlplay.features.sentence_embeddings. The common components are computed by
           fit_components from the Gram matrix accumulated over batches, so the corpus never has to
           fit in memory. They are fixed afterwards, call fit_components again if the embeddings change.
"""
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from nlplay.features.sentence_embeddings import sif_weights, usif_a, usif_weights
from nlplay.models.pytorch.utils import padding_mask


class _WeightedAverageEmbedding(nn.Module):
    def __init__(self, embedding_matrix, word_weights: np.ndarray, padding_idx: int | None, update_embedding: bool):
        super().__init__()
        weights = torch.as_tensor(np.asarray(embedding_matrix), dtype=torch.float32)
        if weights.dim() != 2 or len(word_weights) != weights.size(0):
            raise ValueError("embedding_matrix must have shape (vocabulary_size, dim) and match word_counts")
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=not update_embedding, padding_idx=padding_idx)
        self.register_buffer("word_weights", torch.as_tensor(word_weights, dtype=torch.float32))
        dim = weights.size(1)
        self.register_buffer("components", torch.zeros(0, dim))
        self.register_buffer("lambdas", torch.zeros(0))

    def _normalize(self, vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return vectors

    def _lambdas(self, singular_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def weighted_average(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: token ids of shape (batch, seq_len).
        :returns: weighted average of the word vectors of shape (batch, dim), padding excluded.
        """
        mask = padding_mask(x, self.embedding.padding_idx)
        vectors = self._normalize(self.embedding(x), mask)
        weights = self.word_weights[x] * mask
        return (weights.unsqueeze(2) * vectors).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    @torch.no_grad()
    def fit_components(self, batches: torch.Tensor | Iterable[torch.Tensor], n_components: int | None = None):
        """
        Compute the common components of a corpus, the top right singular vectors of its (non centered)
        weighted averages matrix, from the eigen decomposition of the accumulated Gram matrix.
        :param batches: token ids of shape (n_sentences, seq_len), or an iterable of such batches.
        :param n_components: number of components, the value given at construction if None.
        :returns: self.
        """
        n_components = self.n_components if n_components is None else n_components
        dim = self.embedding.embedding_dim
        if not 0 <= n_components < dim:
            raise ValueError(f"n_components must be in [0, {dim}), got {n_components}")
        if isinstance(batches, torch.Tensor):
            batches = [batches]
        gram = torch.zeros(dim, dim, dtype=torch.float64, device=self.word_weights.device)
        for x in batches:
            v = self.weighted_average(x.to(self.word_weights.device)).double()
            gram += v.T @ v
        # Right singular vectors of V are the eigenvectors of V^T V, singular values = sqrt(eigenvalues)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        top = eigenvalues.argsort(descending=True)[:n_components]
        self.components = eigenvectors[:, top].T.float().contiguous()
        self.lambdas = self._lambdas(eigenvalues[top].clamp(min=0).sqrt()).float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: token ids of shape (batch, seq_len).
        :returns: sentence embeddings of shape (batch, dim), with the fitted common components removed.
        """
        v = self.weighted_average(x)
        return v - (v @ self.components.T * self.lambdas) @ self.components


class SIFEmbedding(_WeightedAverageEmbedding):
    def __init__(
        self,
        embedding_matrix,
        word_counts,
        a: float = 1e-3,
        n_components: int = 1,
        padding_idx: int | None = 0,
        update_embedding: bool = False,
    ):
        """
        SIF sentence embeddings, Arora et al. (2017), call fit_components on a corpus before use.
        :param embedding_matrix: word vectors of shape (vocabulary_size, dim), numpy array or tensor.
        :param word_counts: count of each vocabulary id in a large corpus, shape (vocabulary_size,).
            Ids with a zero count get a weight of 1, as in the reference code.
        :param a: smoothing parameter of the weights a / (a + p(w)).
        :param n_components: number of principal components removed by fit_components.
        :param padding_idx: padding token id, excluded from the average.
        :param update_embedding: train (True) or freeze (False) the word vectors.
        """
        counts = np.asarray(word_counts, dtype=np.float64)
        super().__init__(embedding_matrix, sif_weights(counts / counts.sum(), a), padding_idx, update_embedding)
        self.a = a
        self.n_components = n_components

    def _lambdas(self, singular_values: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(singular_values)


class USIFEmbedding(_WeightedAverageEmbedding):
    def __init__(
        self,
        embedding_matrix,
        word_counts,
        n: int = 11,
        m: int = 5,
        normalize: str = "reference",
        padding_idx: int | None = 0,
        update_embedding: bool = False,
    ):
        """
        uSIF sentence embeddings, Ethayarajh (2018), call fit_components on a corpus before use.
        :param embedding_matrix: word vectors of shape (vocabulary_size, dim), numpy array or tensor.
        :param word_counts: count of each vocabulary id in a large corpus, shape (vocabulary_size,).
            Ids with a zero count get the smallest probability, as in the reference code.
        :param n: expected random walk length, i.e. the average sentence length (about 11 for STS).
        :param m: number of common discourse vectors removed by fit_components.
        :param normalize: "reference" (each dimension divided by its L2 norm over the sentence words,
            as in the reference code), "l2" (unit length word vectors) or "none".
        :param padding_idx: padding token id, excluded from the average.
        :param update_embedding: train (True) or freeze (False) the word vectors.
        """
        if normalize not in ("reference", "l2", "none"):
            raise ValueError(f"Unknown normalize: {normalize}")
        counts = np.asarray(word_counts, dtype=np.float64)
        known = counts > 0
        if padding_idx is not None:
            known[padding_idx] = False
        probs = counts[known] / counts[known].sum()
        a = usif_a(probs, n)
        # Words without a count get the smallest probability
        all_probs = np.full(len(counts), probs.min())
        all_probs[known] = probs
        super().__init__(embedding_matrix, usif_weights(all_probs, a), padding_idx, update_embedding)
        self.a = a
        self.n = n
        self.n_components = m
        self.normalize = normalize

    def _normalize(self, vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vectors = vectors * mask.unsqueeze(2)
        if self.normalize == "reference":
            norms = vectors.norm(dim=1, keepdim=True)
        elif self.normalize == "l2":
            norms = vectors.norm(dim=2, keepdim=True)
        else:
            return vectors
        return vectors / torch.where(norms > 0, norms, torch.ones_like(norms))

    def _lambdas(self, singular_values: torch.Tensor) -> torch.Tensor:
        return singular_values ** 2 / (singular_values ** 2).sum()
