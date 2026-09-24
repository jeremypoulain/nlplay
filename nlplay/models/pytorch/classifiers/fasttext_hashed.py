"""
Title    : Bag of Tricks for Efficient Text Classification - 2017
Authors  : Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov
Papers   : https://aclanthology.org/E17-2068.pdf
Source   : https://github.com/facebookresearch/fastText (C++ reference implementation)
           https://github.com/InseeFrLab/torchTextClassifiers (PyTorch lineage)

Note     : fastText as in the C++ supervised mode rather than a bag of words: the input is the flat bag
           of word ids, hashed word n-grams and hashed character n-grams built by FastTextFeaturizer,
           averaged by a single EmbeddingBag of nwords + bucket rows, followed by a linear layer without
           bias, initialized like fastText (input uniform in +-1 / dim, output zero). Train it with
           CrossEntropyLoss and SGD with a linear learning rate decay (fasttext_sgd) as fastText does,
           sparse=True keeps the updates of the large embedding table cheap.
"""
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class HashedFastText(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embedding_size: int = 100,
        drop_out: float = 0.0,
        bias: bool = False,
        sparse: bool = False,
        fasttext_init: bool = True,
    ):
        """
        :param num_features: size of the feature space, FastTextFeaturizer.num_features (nwords + bucket).
        :param num_classes: number of classes.
        :param embedding_size: size of the embeddings (fastText -dim).
        :param drop_out: dropout on the averaged features, 0 as in fastText.
        :param bias: output layer bias, fastText has none.
        :param sparse: sparse gradients for the embedding table, use SGD or SparseAdam for it.
        :param fasttext_init: input uniform in +-1 / embedding_size and zero output as in fastText,
            PyTorch default init otherwise.
        """
        super().__init__()
        if not 0.0 <= drop_out < 1.0:
            raise ValueError("drop_out must be in [0, 1)")
        self.num_features = num_features
        # One extra row used as padding for batches of padded feature ids, never averaged
        self.embedding = nn.EmbeddingBag(
            num_features + 1, embedding_size, mode="mean", sparse=sparse, padding_idx=num_features
        )
        self.dropout = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(embedding_size, num_classes, bias=bias)
        if fasttext_init:
            with torch.no_grad():
                nn.init.uniform_(self.embedding.weight, -1.0 / embedding_size, 1.0 / embedding_size)
                self.embedding.weight[num_features].zero_()
                nn.init.zeros_(self.fc1.weight)
                if bias:
                    nn.init.zeros_(self.fc1.bias)

    @property
    def padding_idx(self) -> int:
        return self.num_features

    def forward(self, x: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param x: padded feature ids of shape (batch, n_features) padded with padding_idx, or flat
            feature ids of shape (total_features,) with offsets.
        :param offsets: start of each text in x when x is flat.
        :returns: raw scores of shape (batch, num_classes), texts without features score as zero features.
        """
        hidden = self.embedding(x, offsets)
        return self.fc1(self.dropout(hidden))

    @classmethod
    def from_fasttext(cls, native_model, **kwargs) -> "HashedFastText":
        """
        Copy the weights of a native fastText supervised model (fasttext python package).
        :param native_model: loaded fasttext model, with its input matrix of nwords + bucket rows.
        :param kwargs: extra HashedFastText parameters.
        :returns: a model producing the same scores as the native model for the same feature ids.
        """
        input_matrix = torch.as_tensor(np.asarray(native_model.get_input_matrix()), dtype=torch.float32)
        output_matrix = torch.as_tensor(np.asarray(native_model.get_output_matrix()), dtype=torch.float32)
        model = cls(input_matrix.size(0), output_matrix.size(0), input_matrix.size(1), fasttext_init=False, **kwargs)
        with torch.no_grad():
            model.embedding.weight[:-1].copy_(input_matrix)
            model.embedding.weight[-1].zero_()
            model.fc1.weight.copy_(output_matrix)
            if model.fc1.bias is not None:
                model.fc1.bias.zero_()
        return model


def encode_batch(features: Sequence[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    :param features: feature ids of each text, from FastTextFeaturizer.transform.
    :returns: (flat feature ids, offsets) for HashedFastText(x, offsets), no padding involved.
    """
    lengths = torch.tensor([len(f) for f in features], dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])
    flat = np.concatenate(features) if len(features) else np.zeros(0, dtype=np.int64)
    return torch.as_tensor(flat, dtype=torch.long), offsets


class FastTextDataset(Dataset):
    """
    Variable length feature ids and labels, padded per batch by collate_fn, which the
    PytorchModelTrainer uses automatically.
    """

    def __init__(self, features: Sequence[np.ndarray], labels, padding_idx: int):
        """
        :param features: feature ids of each text, from FastTextFeaturizer.transform.
        :param labels: class index of each text.
        :param padding_idx: HashedFastText.padding_idx, i.e. FastTextFeaturizer.num_features.
        """
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        self.features = features
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.padding_idx = padding_idx

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def collate_fn(self, batch):
        """
        :param batch: list of (feature ids, label).
        :returns: (feature ids padded to the longest text of the batch, labels).
        """
        features, labels = zip(*batch)
        x = torch.full((len(features), max(1, max(len(f) for f in features))), self.padding_idx, dtype=torch.long)
        for i, f in enumerate(features):
            x[i, : len(f)] = torch.as_tensor(f, dtype=torch.long)
        return x, torch.stack(labels)


def fasttext_sgd(model: nn.Module, lr: float, total_steps: int):
    """
    Plain SGD with the linear learning rate decay of fastText (lr * (1 - progress)), stepped per batch.
    Works with sparse embedding gradients.
    :param model: model to optimize.
    :param lr: initial learning rate, 0.1 by default in fastText supervised mode, often higher with
        minibatches, e.g. 0.5 to 1.
    :param total_steps: total number of optimizer steps, i.e. epochs * batches per epoch.
    :returns: (optimizer, scheduler), the scheduler to step after every optimizer step.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / total_steps))
    return optimizer, scheduler
