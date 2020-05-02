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
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
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
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


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
        output_y = output[torch.arange(0, y.size()[0]).long(), y.data].view(-1, 1)  # view for transpose
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