"""
Title    : The Road Less Scheduled - 2024
Authors  : Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky
Papers   : https://arxiv.org/abs/2405.15682
Source   : https://github.com/facebookresearch/schedule_free (Apache License 2.0)
Note     : Schedule-Free AdamW and SGD: no learning rate schedule nor total number of steps, the optimizer keeps
           an average of the iterates instead. The parameters hold the training point in train mode and the
           averaged point in eval mode, so call optimizer.train() before training and optimizer.eval() before
           evaluating or saving a checkpoint (the PytorchModelTrainer does it). Dense gradients only.
           Performance: every step updates every parameter (z and the second moment for AdamW), about the cost of
           AdamW. For large sparse embedding tables (e.g. HashedFastText) this is far slower than sparse SGD, see
           fasttext_optimizer for measured timings.
"""
from collections.abc import Callable

import torch


class _ScheduleFree(torch.optim.Optimizer):
    """Shared train / eval switching and iterate averaging weights."""

    def _momentum(self, group) -> float:
        raise NotImplementedError

    @torch.no_grad()
    def eval(self):
        """Switch the parameters to the averaged point x, for evaluation and checkpoints."""
        for group in self.param_groups:
            if group["train_mode"]:
                momentum = self._momentum(group)
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"].to(p.device), weight=1 - 1 / momentum)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        """Switch the parameters to the training point y."""
        for group in self.param_groups:
            if not group["train_mode"]:
                momentum = self._momentum(group)
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"].to(p.device), weight=1 - momentum)
                group["train_mode"] = True

    @staticmethod
    def _schedule(group) -> tuple[float, float]:
        # Warmup, then the weight of the new iterate in the average (ckp1)
        k = group["k"]
        lr = group["lr"] * ((k + 1) / group["warmup_steps"] if k < group["warmup_steps"] else 1.0)
        group["scheduled_lr"] = lr
        lr_max = group["lr_max"] = max(lr, group["lr_max"])
        weight = ((k + 1) ** group["r"]) * (lr_max ** group["weight_lr_power"])
        group["weight_sum"] = group["weight_sum"] + weight
        ckp1 = weight / group["weight_sum"] if group["weight_sum"] else 0.0
        return lr, ckp1

    def _check_train_mode(self):
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "The optimizer is not in train mode, call optimizer.train() before training and "
                "optimizer.eval() before evaluating"
            )


class AdamWScheduleFree(_ScheduleFree):
    def __init__(
        self,
        params,
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
    ):
        """
        :param params: parameters or parameter groups.
        :param lr: learning rate, usually 1 to 10 times the one of AdamW.
        :param betas: interpolation / averaging momentum and second moment decay.
        :param eps: denominator term.
        :param weight_decay: weight decay computed at the training point.
        :param warmup_steps: linear warmup steps, the only schedule needed.
        :param r: polynomial weighting power of the average.
        :param weight_lr_power: power of the learning rate in the averaging weights.
        """
        if lr < 0.0 or eps < 0.0 or weight_decay < 0.0:
            raise ValueError("lr, eps and weight_decay must be >= 0")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, r=r, k=0, warmup_steps=warmup_steps, train_mode=False, weight_sum=0.0,
            lr_max=-1.0, scheduled_lr=0.0, weight_lr_power=weight_lr_power, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _momentum(self, group) -> float:
        return group["betas"][0]

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        self._check_train_mode()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            bias_correction2 = 1 - beta2 ** (group["k"] + 1)
            lr, ckp1 = self._schedule(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamWScheduleFree does not support sparse gradients")
                state = self.state[p]
                if "z" not in state:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                y, z, exp_avg_sq = p, state["z"], state["exp_avg_sq"]
                grad = p.grad
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                grad_normalized = grad / exp_avg_sq.div(bias_correction2).sqrt_().add_(group["eps"])
                if group["weight_decay"] != 0:
                    grad_normalized.add_(y, alpha=group["weight_decay"])
                # y moves towards z, then takes the interpolated gradient step, z takes the plain step
                y.lerp_(end=z, weight=ckp1)
                y.add_(grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))
                z.sub_(grad_normalized, alpha=lr)
            group["k"] += 1
        return loss


class SGDScheduleFree(_ScheduleFree):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
    ):
        """
        :param params: parameters or parameter groups.
        :param lr: learning rate, usually 1 to 10 times the one of SGD with momentum.
        :param momentum: interpolation / averaging momentum, in ]0, 1[.
        :param weight_decay: weight decay computed at the training point.
        :param warmup_steps: linear warmup steps, the only schedule needed.
        :param r: polynomial weighting power of the average.
        :param weight_lr_power: power of the learning rate in the averaging weights.
        """
        if lr < 0.0 or weight_decay < 0.0:
            raise ValueError("lr and weight_decay must be >= 0")
        if not 0.0 < momentum < 1.0:
            raise ValueError("momentum must be in ]0, 1[")
        defaults = dict(
            lr=lr, momentum=momentum, r=r, k=0, warmup_steps=warmup_steps, train_mode=False, weight_sum=0.0,
            lr_max=-1.0, scheduled_lr=0.0, weight_lr_power=weight_lr_power, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _momentum(self, group) -> float:
        return group["momentum"]

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        self._check_train_mode()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            momentum = group["momentum"]
            lr, ckp1 = self._schedule(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SGDScheduleFree does not support sparse gradients")
                state = self.state[p]
                if "z" not in state:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)
                y, z = p, state["z"]
                grad = p.grad
                if group["weight_decay"] != 0:
                    grad = grad.add(y, alpha=group["weight_decay"])
                y.lerp_(end=z, weight=ckp1)
                y.add_(grad, alpha=lr * (momentum * (1 - ckp1) - 1))
                z.sub_(grad, alpha=lr)
            group["k"] += 1
        return loss
