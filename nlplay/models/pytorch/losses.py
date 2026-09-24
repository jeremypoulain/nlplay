import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def _check_reduction(reduction: str) -> None:
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Unknown reduction: {reduction}")


class FocalLoss(nn.Module):
    """
    Title    : Focal Loss for Dense Object Detection - 2017
    Authors  : Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar
    Papers   : https://arxiv.org/pdf/1708.02002.pdf
    Source   : https://github.com/mbsariyildiz/focal-loss.pytorch
    Note     : loss = -weight[y] * (1 - p_y)^gamma * log(p_y), computed from raw logits.
               The alpha balanced binary version of the paper is weight=[1 - alpha, alpha].
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        """
        :param gamma: focusing parameter, gamma = 0 gives the cross entropy.
        :param weight: optional per class weight of shape (C,).
        :param reduction: "mean", "sum" or "none".
        """
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        _check_reduction(reduction)
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param input: logits of shape (N, C) or (N, C, d1, d2, ...).
        :param target: class indices of shape (N,) or (N, d1, d2, ...).
        :returns: the reduced loss, or the per element loss if reduction is "none".
        """
        if input.dim() > 2:
            # N, C, d1, ... → N * d1 * ..., C
            input = input.movedim(1, -1).reshape(-1, input.size(1))
        target = target.reshape(-1).long()

        logpt = F.log_softmax(input, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        # The focal factor must stay in the graph, detaching it gives a plain weighted CE gradient
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.to(loss)[target]
        return _reduce(loss, self.reduction)


class LabelSmoothingLoss(nn.Module):
    """
    Title    : Rethinking the Inception Architecture for Computer Vision - 2015
    Authors  : Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
    Papers   : https://arxiv.org/pdf/1512.00567.pdf
    Source   : https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/loss.py
    Note     : Cross entropy against a smoothed target, computed from raw logits: the true class
               gets 1 - label_smoothing, the rest is spread evenly over the other classes.
               If ignore_index is a real class (e.g. a PAD token), that class is also excluded
               from the smoothing, as in OpenNMT. Targets equal to ignore_index are skipped.
    """

    def __init__(
        self,
        label_smoothing: float,
        tgt_vocab_size: int,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        """
        :param label_smoothing: smoothing factor in (0, 1].
        :param tgt_vocab_size: number of classes C.
        :param ignore_index: target value to skip, also excluded from smoothing if in [0, C).
        :param reduction: "mean" (over non ignored targets), "sum" or "none".
        """
        super().__init__()
        if not 0.0 < label_smoothing <= 1.0:
            raise ValueError("label_smoothing must be in (0, 1]")
        _check_reduction(reduction)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.confidence = 1.0 - label_smoothing

        pad_in_vocab = 0 <= ignore_index < tgt_vocab_size
        n_smoothed = tgt_vocab_size - 1 - int(pad_in_vocab)
        if n_smoothed < 1:
            raise ValueError("tgt_vocab_size is too small for label smoothing")
        smoothing = torch.full((tgt_vocab_size,), label_smoothing / n_smoothed)
        if pad_in_vocab:
            smoothing[ignore_index] = 0.0
        self.register_buffer("smoothing", smoothing.unsqueeze(0))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param input: logits of shape (N, C).
        :param target: class indices of shape (N,).
        :returns: the reduced loss, or the per sample loss of shape (N,) if reduction is "none".
        """
        target = target.reshape(-1).long()
        keep = target != self.ignore_index
        # Placeholder class for ignored rows so that scatter never sees ignore_index
        safe_target = torch.where(keep, target, torch.zeros_like(target))

        log_probs = F.log_softmax(input, dim=1)
        true_dist = self.smoothing.to(log_probs).expand_as(log_probs).clone()
        true_dist.scatter_(1, safe_target.unsqueeze(1), self.confidence)

        loss = -(true_dist * log_probs).sum(dim=1) * keep
        if self.reduction == "mean":
            return loss.sum() / keep.sum().clamp(min=1)
        return _reduce(loss, self.reduction)


class MultiClassHingeLoss(nn.Module):
    """
    Weston and Watkins multiclass hinge loss @ https://en.wikipedia.org/wiki/Hinge_loss
    For a sample with scores s and label y:
        loss = weight[y] * sum_{j != y} max(0, margin - s_y + s_j)^p
    Note     : hinge loss is not differentiable at the margin, p = 2 (squared hinge) smooths it.
    """

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        """
        :param p: exponent applied to each hinge term, integer >= 1.
        :param margin: margin between the true class score and the other scores.
        :param weight: optional per class weight of shape (C,), each sample is weighted
            by the weight of its true class.
        :param reduction: "mean", "sum" or "none".
        """
        super().__init__()
        _check_reduction(reduction)
        if not isinstance(p, int) or p < 1:
            raise ValueError("p must be an integer >= 1")
        self.p = p
        self.margin = margin
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param input: raw scores of shape (N, C).
        :param target: class indices of shape (N,).
        :returns: the reduced loss, or the per sample loss of shape (N,) if reduction is "none".
        """
        target = target.reshape(-1).long()
        true_score = input.gather(1, target.unsqueeze(1))
        loss = torch.clamp(self.margin - true_score + input, min=0.0)
        if self.p != 1:
            loss = loss**self.p
        true_mask = F.one_hot(target, num_classes=input.size(1)).bool()
        loss = loss.masked_fill(true_mask, 0.0).sum(dim=1)
        if self.weight is not None:
            loss = loss * self.weight.to(loss)[target]
        return _reduce(loss, self.reduction)


class SupConLoss(nn.Module):
    """
    Title    : Supervised Contrastive Learning - 2020
    Authors  : Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian,
               Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan
    Papers   : https://arxiv.org/pdf/2004.11362.pdf
    Source   : https://github.com/HobbitLong/SupContrast [Yonglong Tian (yonglong@mit.edu)]
    Note     : It also supports the unsupervised contrastive loss in SimCLR.
               Anchors without any positive contrast contribute 0 instead of NaN.
    """

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Compute loss for model. If both labels and mask are None, it degenerates to the
        SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf
        :param features: hidden vectors of shape (bsz, n_views, ...).
        :param labels: ground truth of shape (bsz,).
        :param mask: contrastive mask of shape (bsz, bsz), mask_{i,j} = 1 if sample j has
            the same class as sample i. Can be asymmetric.
        :returns: a loss scalar.
        """
        device, dtype = features.device, features.dtype

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=dtype, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).to(device=device, dtype=dtype)
        else:
            mask = mask.to(device=device, dtype=dtype)

        contrast_count = features.shape[1]
        if batch_size * contrast_count <= 1:
            raise ValueError("SupConLoss requires at least one non-self contrast")
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        logits = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        self_mask = torch.zeros_like(mask, dtype=torch.bool)
        self_index = torch.arange(batch_size * anchor_count, device=device).view(-1, 1)
        self_mask.scatter_(1, self_index, True)
        mask = mask.masked_fill(self_mask, 0.0)

        # log_prob with logsumexp over non self contrasts, stable even in fp16 or at low temperature
        masked_logits = logits.masked_fill(self_mask, float("-inf"))
        log_norm = torch.logsumexp(masked_logits, dim=1, keepdim=True)
        log_prob = logits - log_norm

        # compute mean of log-likelihood over positive, anchors without positive → 0
        pos_count = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count.clamp(min=1.0)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ModifiedHuberLoss(nn.Module):
    """
    Modified Huber loss, PyTorch equivalent of the sklearn SGDClassifier loss="modified_huber".
    For a raw score z = y * f(x) with y in {-1, +1}:
        loss = max(0, 1 - z)^2   if z >= -1
        loss = -4 * z            otherwise
    Binary    : input of shape (N,) or (N, 1), target in {0, 1} or {-1, +1}.
                Labels are not validated: any target > 0 is the positive class.
    Multiclass: input of shape (N, C) with C >= 2, target of class indices in [0, C - 1].
                As in sklearn, a One-vs-Rest scheme is used: column k is scored against
                +1 for class k and -1 for the others, and the C binary losses are summed.
    Source   : https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_sgd_fast.pyx.tp
    """

    def __init__(self, weight: torch.Tensor | None = None, reduction: str = "mean"):
        """
        :param weight: optional per class weight, equivalent of sklearn class_weight.
            Binary → shape (2,), weights of the negative and positive samples.
            Multiclass → shape (C,), as in sklearn the OvR classifier of class k weights
            its positive samples by weight[k] and its negative samples by 1.
        :param reduction: "mean" (average over samples), "sum" or "none" (per sample loss).
        """
        super().__init__()
        _check_reduction(reduction)
        if weight is not None and weight.dim() != 1:
            raise ValueError("weight must be a 1-D tensor")
        self.reduction = reduction
        self.register_buffer("weight", weight)

    @staticmethod
    def _is_binary(input: torch.Tensor) -> bool:
        if input.dim() not in (1, 2):
            raise ValueError("input must have shape (N,), (N, 1) or (N, C)")
        return input.dim() == 1 or input.size(1) == 1

    @staticmethod
    def _modified_huber(z: torch.Tensor) -> torch.Tensor:
        return torch.where(z >= -1.0, torch.clamp(1.0 - z, min=0.0).square(), -4.0 * z)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param input: raw scores (no sigmoid / softmax), shape (N,), (N, 1) or (N, C).
        :param target: class labels of shape (N,) or (N, 1), on the same device as input.
        :returns: the reduced loss, or the per sample loss of shape (N,) if reduction is "none".
        """
        binary = self._is_binary(input)
        # Flatten both sides so that (N, 1) inputs or targets never broadcast to (N, N)
        target = target.reshape(-1)
        if target.size(0) != input.size(0):
            raise ValueError("input and target batch sizes differ")
        weight = None if self.weight is None else self.weight.to(input)

        if binary:
            # Map {0, 1} labels to {-1, +1}, labels already in {-1, +1} are kept as is
            input = input.reshape(-1)
            pos = target > 0
            loss = self._modified_huber(torch.where(pos, input, -input))
            if weight is not None:
                if weight.numel() != 2:
                    raise ValueError("Binary weight must have shape (2,)")
                loss = loss * weight[pos.long()]
        else:
            # One-vs-Rest targets in {-1, +1} of shape (N, C)
            n_classes = input.size(1)
            one_hot = F.one_hot(target.long(), num_classes=n_classes).bool()
            loss = self._modified_huber(torch.where(one_hot, input, -input))
            if weight is not None:
                if weight.numel() != n_classes:
                    raise ValueError(f"weight must have shape ({n_classes},)")
                loss = loss * torch.where(one_hot, weight, torch.ones_like(weight))
            loss = loss.sum(dim=1)

        return _reduce(loss, self.reduction)

    @staticmethod
    @torch.no_grad()
    def predict_proba(input: torch.Tensor) -> torch.Tensor:
        """
        Probability estimates from raw scores, same formula as sklearn SGDClassifier.predict_proba.
        :param input: raw scores of shape (N,), (N, 1) or (N, C).
        :returns: probabilities of shape (N, 2) for binary inputs, (N, C) otherwise.
        """
        if ModifiedHuberLoss._is_binary(input):
            prob = (torch.clamp(input.reshape(-1), -1.0, 1.0) + 1.0) / 2.0
            return torch.stack([1.0 - prob, prob], dim=1)

        prob = (torch.clamp(input, -1.0, 1.0) + 1.0) / 2.0
        norm = prob.sum(dim=1, keepdim=True)
        # Rows where every OvR score is <= -1 get a uniform distribution
        uniform = torch.full_like(prob, 1.0 / prob.size(1))
        safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        return torch.where(norm > 0, prob / safe_norm, uniform)
