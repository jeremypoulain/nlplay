"""
Title    : Bag of Tricks for Efficient Text Classification - 2017
Authors  : Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov
Papers   : https://aclanthology.org/E17-2068.pdf
Source   : https://github.com/facebookresearch/fastText (C++ reference implementation)
           https://github.com/InseeFrLab/torchTextClassifiers (PyTorch lineage)

Note     : fastText as in the C++ supervised mode rather than a bag of words: the input is the flat bag
           of word ids, hashed word n-grams and hashed character n-grams built by FastTextFeaturizer,
           averaged by a single EmbeddingBag of nwords + bucket rows, followed by a linear layer without
           bias (loss="softmax") or a Huffman tree hierarchical softmax (loss="hs"), initialized like
           fastText (input uniform in +-1 / dim, output zero). Train it with fasttext_optimizer, SGD with
           the linear learning rate decay of fastText by default, sparse=True keeps the updates of the
           large embedding table cheap.
"""
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from nlplay.models.pytorch.optimizer.prodigy import Prodigy
from nlplay.models.pytorch.optimizer.schedulefree import AdamWScheduleFree, SGDScheduleFree


class HierarchicalSoftmax(nn.Module):
    """
    Hierarchical softmax over a Huffman tree of the classes, as fastText -loss hs: each class is a leaf and
    p(class) is the product of the binary logistic decisions along its path, one vector per inner node.
    The tree is built with the fastText algorithm, so the node weights of a native model can be reused.
    Training cost per sample is O(dim * log2(num_classes)) instead of O(dim * num_classes).
    """

    def __init__(self, embedding_size: int, class_counts: Sequence[int]):
        """
        :param embedding_size: size of the hidden vectors.
        :param class_counts: number of training samples of each class, frequent classes get short paths.
        """
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.int64)
        n_classes = len(counts)
        if n_classes < 2:
            raise ValueError("Hierarchical softmax needs at least 2 classes")
        # The fastText tree building expects the leaves sorted by decreasing count
        order = np.argsort(-counts, kind="stable")
        paths, codes = self._build_tree(counts[order])
        depth = max(len(p) for p in paths)
        nodes = torch.zeros(n_classes, depth, dtype=torch.long)
        binary = torch.zeros(n_classes, depth, dtype=torch.bool)
        mask = torch.zeros(n_classes, depth, dtype=torch.bool)
        for leaf, cls in enumerate(order):
            nodes[cls, : len(paths[leaf])] = torch.tensor(paths[leaf], dtype=torch.long)
            binary[cls, : len(codes[leaf])] = torch.tensor(codes[leaf], dtype=torch.bool)
            mask[cls, : len(paths[leaf])] = True
        self.register_buffer("nodes", nodes)
        self.register_buffer("codes", binary)
        self.register_buffer("mask", mask)
        # One vector per inner node, zero init as the fastText output matrix
        self.weight = nn.Parameter(torch.zeros(n_classes - 1, embedding_size))

    @staticmethod
    def _build_tree(counts: np.ndarray) -> tuple[list[list[int]], list[list[bool]]]:
        # Port of HierarchicalSoftmaxLoss::buildTree, leaves 0 .. n - 1, inner nodes n .. 2n - 2
        n = len(counts)
        count = [int(c) for c in counts] + [int(1e15)] * (n - 1)
        parent, is_right = [-1] * (2 * n - 1), [False] * (2 * n - 1)
        leaf, node = n - 1, n
        for i in range(n, 2 * n - 1):
            mini = [0, 0]
            for j in range(2):
                if leaf >= 0 and count[leaf] < count[node]:
                    mini[j], leaf = leaf, leaf - 1
                else:
                    mini[j], node = node, node + 1
            count[i] = count[mini[0]] + count[mini[1]]
            parent[mini[0]] = parent[mini[1]] = i
            is_right[mini[1]] = True
        paths, codes = [], []
        for i in range(n):
            path, code, j = [], [], i
            while parent[j] != -1:
                path.append(parent[j] - n)
                code.append(is_right[j])
                j = parent[j]
            paths.append(path)
            codes.append(code)
        return paths, codes

    def _path_log_prob(self, scores: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # log sigmoid(s) for a right turn, log(1 - sigmoid(s)) = log sigmoid(-s) for a left one
        return (F.logsigmoid(torch.where(codes, scores, -scores)) * mask).sum(dim=-1)

    def log_prob(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        :param hidden: hidden vectors of shape (batch, embedding_size).
        :returns: log probabilities of every class, shape (batch, num_classes), rows sum to 1 in probability.
        """
        scores = hidden @ self.weight.T
        return self._path_log_prob(scores[:, self.nodes], self.codes, self.mask)

    def loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Negative log likelihood computed along the target paths only.
        :param hidden: hidden vectors of shape (batch, embedding_size).
        :param targets: class indices of shape (batch,).
        :returns: mean negative log probability of the targets.
        """
        nodes = self.nodes[targets]
        scores = (hidden.unsqueeze(1) * self.weight[nodes]).sum(dim=-1)
        return -self._path_log_prob(scores, self.codes[targets], self.mask[targets]).mean()


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
        loss: str = "softmax",
        class_counts: Sequence[int] | None = None,
    ):
        """
        :param num_features: size of the feature space, FastTextFeaturizer.num_features (nwords + bucket).
        :param num_classes: number of classes.
        :param embedding_size: size of the embeddings (fastText -dim).
        :param drop_out: dropout on the averaged features, 0 as in fastText.
        :param bias: output layer bias, fastText has none, softmax only.
        :param sparse: sparse gradients for the embedding table, see fasttext_optimizer.
        :param fasttext_init: input uniform in +-1 / embedding_size and zero output as in fastText,
            PyTorch default init otherwise.
        :param loss: "softmax" → forward returns raw scores for CrossEntropyLoss, "hs" → hierarchical
            softmax, forward returns log probabilities (NLLLoss) and training_loss only computes the target
            paths, which the PytorchModelTrainer uses automatically.
        :param class_counts: number of training samples of each class, required by loss="hs".
        """
        super().__init__()
        if not 0.0 <= drop_out < 1.0:
            raise ValueError("drop_out must be in [0, 1)")
        if loss not in ("softmax", "hs"):
            raise ValueError(f"Unknown loss: {loss}")
        if loss == "hs" and (class_counts is None or len(class_counts) != num_classes):
            raise ValueError("loss='hs' requires class_counts with one count per class")
        self.num_features = num_features
        self.loss_name = loss
        # One extra row used as padding for batches of padded feature ids, never averaged
        self.embedding = nn.EmbeddingBag(
            num_features + 1, embedding_size, mode="mean", sparse=sparse, padding_idx=num_features
        )
        self.dropout = nn.Dropout(drop_out)
        if loss == "hs":
            self.hs = HierarchicalSoftmax(embedding_size, class_counts)
        else:
            self.fc1 = nn.Linear(embedding_size, num_classes, bias=bias)
        if fasttext_init:
            with torch.no_grad():
                nn.init.uniform_(self.embedding.weight, -1.0 / embedding_size, 1.0 / embedding_size)
                self.embedding.weight[num_features].zero_()
                if loss == "softmax":
                    nn.init.zeros_(self.fc1.weight)
                    if bias:
                        nn.init.zeros_(self.fc1.bias)

    @property
    def padding_idx(self) -> int:
        return self.num_features

    def hidden(self, x: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param x: padded feature ids of shape (batch, n_features) padded with padding_idx, or flat
            feature ids of shape (total_features,) with offsets.
        :param offsets: start of each text in x when x is flat.
        :returns: averaged feature embeddings of shape (batch, embedding_size), after dropout.
        """
        return self.dropout(self.embedding(x, offsets))

    def forward(self, x: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param x: padded feature ids of shape (batch, n_features) padded with padding_idx, or flat
            feature ids of shape (total_features,) with offsets.
        :param offsets: start of each text in x when x is flat.
        :returns: raw scores (softmax) or log probabilities (hs) of shape (batch, num_classes).
        """
        hidden = self.hidden(x, offsets)
        return self.hs.log_prob(hidden) if self.loss_name == "hs" else self.fc1(hidden)

    def training_loss(self, x: torch.Tensor, targets: torch.Tensor, offsets: torch.Tensor | None = None):
        """
        :param x: feature ids, as in forward.
        :param targets: class indices of shape (batch,).
        :param offsets: start of each text in x when x is flat.
        :returns: the hierarchical softmax loss over the target paths only, None for loss="softmax"
            (the PytorchModelTrainer then uses its criterion).
        """
        if self.loss_name != "hs":
            return None
        return self.hs.loss(self.hidden(x, offsets), targets)

    @torch.no_grad()
    def load_word_vectors(self, featurizer, words: Sequence[str], vectors: np.ndarray) -> int:
        """
        Copy pretrained word vectors into the word rows, n-gram rows keep their init, as fastText
        -pretrainedVectors. Fit the featurizer with pretrained_words=words to keep all of them.
        :param featurizer: fitted FastTextFeaturizer used to build the features.
        :param words: pretrained words, e.g. from read_word_vectors.
        :param vectors: pretrained vectors of shape (len(words), embedding_size).
        :returns: number of vocabulary words initialized from the pretrained vectors.
        """
        vectors = torch.as_tensor(np.asarray(vectors), dtype=self.embedding.weight.dtype)
        if vectors.shape != (len(words), self.embedding.embedding_dim):
            raise ValueError(
                f"vectors must have shape ({len(words)}, {self.embedding.embedding_dim}), got {tuple(vectors.shape)}"
            )
        rows = [(featurizer.word2id_[w], i) for i, w in enumerate(words) if w in featurizer.word2id_]
        if rows:
            ids, src = map(list, zip(*rows))
            self.embedding.weight[ids] = vectors[src].to(self.embedding.weight.device)
        return len(rows)

    @classmethod
    def from_fasttext(cls, native_model, **kwargs) -> "HashedFastText":
        """
        Copy the weights of a native fastText supervised model (fasttext python package), softmax or hs.
        :param native_model: loaded fasttext model, with its input matrix of nwords + bucket rows.
        :param kwargs: extra HashedFastText parameters.
        :returns: a model producing the same predictions as the native model for the same feature ids,
            classes in the native_model.labels order.
        """
        import fasttext_pybind

        input_matrix = torch.as_tensor(np.asarray(native_model.get_input_matrix()), dtype=torch.float32)
        output_matrix = torch.as_tensor(np.asarray(native_model.get_output_matrix()), dtype=torch.float32)
        num_classes = len(native_model.labels)
        is_hs = native_model.f.getArgs().loss == fasttext_pybind.loss_name.hs
        if is_hs:
            kwargs.update(loss="hs", class_counts=native_model.get_labels(include_freq=True)[1])
        model = cls(input_matrix.size(0), num_classes, input_matrix.size(1), fasttext_init=False, **kwargs)
        with torch.no_grad():
            model.embedding.weight[:-1].copy_(input_matrix)
            model.embedding.weight[-1].zero_()
            if is_hs:
                # The native output matrix has one row per class, the first num_classes - 1 are the nodes
                model.hs.weight.copy_(output_matrix[: num_classes - 1])
            else:
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


class CombinedOptimizer:
    """Several optimizers stepped together, e.g. SparseAdam for sparse embeddings and AdamW for the rest."""

    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self) -> list[dict]:
        return [group for opt in self.optimizers for group in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step()

    def train(self):
        """Train mode of the optimizers defining one, e.g. schedule-free ones."""
        for opt in self.optimizers:
            if hasattr(opt, "train"):
                opt.train()

    def eval(self):
        """Eval mode of the optimizers defining one, e.g. schedule-free ones."""
        for opt in self.optimizers:
            if hasattr(opt, "eval"):
                opt.eval()

    def state_dict(self) -> dict:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict):
        for opt, state in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(state)


class CombinedScheduler:
    """Learning rate schedulers of a CombinedOptimizer stepped together."""

    def __init__(self, schedulers: Sequence[torch.optim.lr_scheduler.LRScheduler]):
        self.schedulers = list(schedulers)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_last_lr(self) -> list[float]:
        return [lr for scheduler in self.schedulers for lr in scheduler.get_last_lr()]

    def state_dict(self) -> dict:
        return {"schedulers": [scheduler.state_dict() for scheduler in self.schedulers]}

    def load_state_dict(self, state_dict: dict):
        for scheduler, state in zip(self.schedulers, state_dict["schedulers"]):
            scheduler.load_state_dict(state)


_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop,
    "adamw_schedulefree": AdamWScheduleFree,
    "sgd_schedulefree": SGDScheduleFree,
    "prodigy": Prodigy,
}
# Optimizers without learning rate schedule, their warmup is built in
_SCHEDULE_FREE = (AdamWScheduleFree, SGDScheduleFree)
# Optimizers accepting sparse gradients, the others get SparseAdam for the sparse parameters
_SPARSE_OK = (torch.optim.SGD, torch.optim.Adagrad, torch.optim.SparseAdam)


def linear_decay(total_steps: int, warmup_steps: int = 0) -> Callable[[int], float]:
    """
    :param total_steps: total number of optimizer steps.
    :param warmup_steps: steps of linear warmup from 0, 0 as in fastText.
    :returns: LambdaLR multiplier, linear warmup then linear decay to 0 (fastText lr * (1 - progress)).
    """
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    return factor


def fasttext_optimizer(
    model: nn.Module,
    lr: float,
    total_steps: int | None,
    optimizer: str | type[torch.optim.Optimizer] = "sgd",
    warmup_steps: int = 0,
    sparse_lr: float | None = None,
    **optimizer_kwargs,
):
    """
    Optimizer and per step linear learning rate decay for HashedFastText (or any model), sparse aware:
    parameters receiving sparse gradients (EmbeddingBag(sparse=True)) go to the same optimizer when it
    supports them (SGD, Adagrad), to a SparseAdam otherwise.
    Native fastText behavior: optimizer="sgd", warmup_steps=0, no weight_decay, with a model built with
    sparse=True and drop_out=0. fastText updates after every sample (batch_size=1) with lr=0.1 and decays
    the rate with the share of processed tokens, here the decay follows the optimizer steps, so with
    minibatches raise lr (0.5 to 1 is a usual start) rather than keeping 0.1.
    :param model: model to optimize.
    Schedule-free optimizers ("adamw_schedulefree", "sgd_schedulefree") get no scheduler, their warmup is built
    in and total_steps is not needed (only for the SparseAdam of sparse embeddings), they need
    optimizer.train() / optimizer.eval() calls, which the PytorchModelTrainer does. "prodigy" estimates the
    learning rate itself, keep lr=1.0, its safeguard_warmup is enabled with warmup_steps. For fully learning
    rate free training, use them with dense embeddings (sparse=False), the SparseAdam lr is not estimated.
    Performance: dense embedding gradients update the whole nwords + bucket table at every step, sparse ones only
    the rows of the batch. Measured on CPU (4 threads, dim 50, batch 64, AG News): sparse SGD 2 to 3 ms per step
    whatever the bucket, dense AdamW / schedule-free / Prodigy 73 to 97 ms per step with 200k buckets and 317 to
    522 ms with 1M buckets. With large hash tables, prefer sparse SGD (or sparse Adam) and tune the learning rate
    rather than paying the dense update of the learning rate free optimizers.
    :param lr: peak learning rate, fastText uses 0.1 with SGD on single samples, minibatch SGD usually
        needs 0.5 to 1, Adam / AdamW around 1e-3 to 1e-2, AdamW schedule-free 1 to 10 times AdamW, Prodigy 1.0.
    :param total_steps: total number of optimizer steps, i.e. epochs * batches per epoch, may be None for
        schedule-free optimizers with dense embeddings.
    :param optimizer: "sgd", "adam", "adamw", "adagrad", "rmsprop", "adamw_schedulefree", "sgd_schedulefree",
        "prodigy" or a torch.optim.Optimizer class.
    :param warmup_steps: steps of linear warmup before the decay.
    :param sparse_lr: learning rate of the SparseAdam used for sparse parameters, lr if None.
    :param optimizer_kwargs: extra arguments of the optimizer, e.g. weight_decay (dense parameters only).
    :returns: (optimizer, scheduler), step the scheduler after every optimizer step, as the
        PytorchModelTrainer does, scheduler is None for schedule-free optimizers of dense parameters only.
    """
    optimizer_cls = _OPTIMIZERS[optimizer.lower()] if isinstance(optimizer, str) else optimizer
    schedule_free = issubclass(optimizer_cls, _SCHEDULE_FREE)
    if schedule_free:
        optimizer_kwargs.setdefault("warmup_steps", warmup_steps)
    elif issubclass(optimizer_cls, Prodigy) and warmup_steps > 0:
        optimizer_kwargs.setdefault("safeguard_warmup", True)
    sparse_params = [
        m.weight for m in model.modules()
        if isinstance(m, (nn.Embedding, nn.EmbeddingBag)) and m.sparse and m.weight.requires_grad
    ]
    sparse_ids = {id(p) for p in sparse_params}
    dense_params = [p for p in model.parameters() if p.requires_grad and id(p) not in sparse_ids]

    if not sparse_params or issubclass(optimizer_cls, _SPARSE_OK):
        optimizers = [optimizer_cls(sparse_params + dense_params, lr=lr, **optimizer_kwargs)]
    else:
        optimizers = [torch.optim.SparseAdam(sparse_params, lr=sparse_lr or lr)]
        if dense_params:
            optimizers.append(optimizer_cls(dense_params, lr=lr, **optimizer_kwargs))
    decayed = [opt for opt in optimizers if not isinstance(opt, _SCHEDULE_FREE)]
    if decayed and total_steps is None:
        raise ValueError("total_steps is required for the learning rate decay")
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, linear_decay(total_steps, warmup_steps)) for opt in decayed]
    optimizer = optimizers[0] if len(optimizers) == 1 else CombinedOptimizer(optimizers)
    scheduler = None if not schedulers else schedulers[0] if len(schedulers) == 1 else CombinedScheduler(schedulers)
    return optimizer, scheduler


def fasttext_sgd(model: nn.Module, lr: float, total_steps: int):
    """
    Plain SGD with the linear learning rate decay of fastText, kept for backward compatibility.
    :param model: model to optimize.
    :param lr: initial learning rate.
    :param total_steps: total number of optimizer steps.
    :returns: (optimizer, scheduler).
    """
    return fasttext_optimizer(model, lr, total_steps, optimizer="sgd")
