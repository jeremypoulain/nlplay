import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Title    : Focal Loss for Dense Object Detection - 2017
    Authors  : Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
    Papers   : https://www.aclweb.org/anthology/P12-2018.pdf
    Source   : https://github.com/mbsariyildiz/focal-loss.pytorch
    """

    def __init__(self, gamma: int = 0, alpha: float = 0.5, size_average: bool = True):
        super(FocalLoss, self).__init__()

        # default parameters
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

        assert (
            self.alpha <= 1 and self.alpha >= 0
        ), "The parameter alpha in Focal Loss must be in range [0, 1]."

        if self.alpha is not None:
            self.alpha = torch.Tensor([self.alpha, 1 - self.alpha])

    def forward(self, input, target):
        """ Get focal loss
        Args:
            input (Variable):  the prediction with shape [batch_size, number of classes]
            target (Variable): the answer with shape [batch_size, number of classes]
        Returns:
            Variable (float): loss
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LabelSmoothingLoss(nn.Module):
    """
    Title    : Rethinking the Inception Architecture for Computer Vision - 2015
    Authors  : Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
    Papers   : https://arxiv.org/pdf/1512.00567.pdf
    Source   : https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/loss.py
    Note     : With label smoothing,KL-divergence between q_{smoothed ground truth prob.}(w)
               and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction="sum")


class MultiClassHingeLoss(nn.Module):
    """SVM loss
    Weston and Watkins version multiclass hinge loss @ https://en.wikipedia.org/wiki/Hinge_loss
    for each sample, given output (a vector of n_class values) and label y (an int \in [0,n_class-1])
    loss = sum_i(max(0, (margin - output[y] + output[i]))^p) where i=0 to n_class-1 and i!=y
    Note: hinge loss is not differentiable
          Let's denote hinge loss as h(x)=max(0,1-x). h'(x) does not exist when x=1,
          because the left and right limits do not converge to the same number, i.e.,
          h'(1-delta)=-1 but h'(1+delta)=0.
          To overcome this obstacle, people proposed squared hinge loss h2(x)=max(0,1-x)^2. In this case,
          h2'(1-delta)=h2'(1+delta)=0
    """

    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(MultiClassHingeLoss, self).__init__()
        self.p = p
        self.margin = margin
        self.weight = weight
        self.size_average = size_average

    def forward(self, output, y):
        output_y = output[torch.arange(0, y.size()[0]).long(), y.data].view(
            -1, 1
        )  # view for transpose
        # margin - output[y] + output[i]
        loss = output - output_y + self.margin  # contains i=y
        # remove i=y items
        loss[torch.arange(0, y.size()[0]).long(), y.data] = 0
        # max(0,_)
        loss[loss < 0] = 0
        # ^p
        if self.p != 1:
            loss = torch.pow(loss, self.p)
        # add weight
        if self.weight is not None:
            loss = loss * self.weight
        # sum up
        loss = torch.sum(loss)
        if self.size_average:
            loss /= output.size()[0]
        return loss


class SupConLoss(nn.Module):
    """
    Title    : Supervised Contrastive Learning - 2020
    Authors  : Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian,
               Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan
    Papers   : https://arxiv.org/pdf/2004.11362.pdf
    Source   : https://github.com/HobbitLong/SupContrast [Yonglong Tian (yonglong@mit.edu)]
    Note     : It also supports the unsupervised contrastive loss in SimCLR
    """

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model.
        If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

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
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
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
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

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
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction: {reduction}")
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
        :param target: class labels of shape (N,) or (N, 1).
        :returns: the reduced loss, or the per sample loss of shape (N,) if reduction is "none".
        """
        binary = self._is_binary(input)
        # Flatten both sides so that (N, 1) inputs or targets never broadcast to (N, N)
        target = target.reshape(-1).to(input.device)
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

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

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
        return torch.where(norm > 0, prob / norm.clamp(min=1e-12), uniform)
