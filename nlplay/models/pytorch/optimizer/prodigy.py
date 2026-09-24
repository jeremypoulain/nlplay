"""
Title    : Prodigy: An Expeditiously Adaptive Parameter-Free Learner - 2024
Authors  : Konstantin Mishchenko, Aaron Defazio
Papers   : https://arxiv.org/abs/2306.06101
Source   : https://github.com/konstmish/prodigy (MIT License)
Note     : Adam with an automatically estimated learning rate d: keep lr=1.0, d grows from d0 towards the scale
           of the distance to the solution. A decaying schedule (e.g. linear or cosine) on lr still helps.
           Dense gradients only, all parameter groups must share the same lr (or use 0 to freeze a group).
           Performance: about 1.3 to 1.6 times the cost of an AdamW step (extra statistics per parameter and one
           device to host synchronization per parameter tensor for the estimate), far slower than sparse SGD for
           large sparse embedding tables, see fasttext_optimizer for measured timings.
"""
import math
from collections.abc import Callable

import torch


class Prodigy(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        beta3: float | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decouple: bool = True,
        use_bias_correction: bool = False,
        safeguard_warmup: bool = False,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = float("inf"),
        slice_p: int = 1,
    ):
        """
        :param params: parameters or parameter groups.
        :param lr: multiplier of the estimated learning rate, keep 1.0 and change d_coef instead if needed.
        :param betas: first and second moment decays.
        :param beta3: decay of the learning rate estimate statistics, sqrt(beta2) if None.
        :param eps: denominator term.
        :param weight_decay: weight decay.
        :param decouple: AdamW style decoupled weight decay.
        :param use_bias_correction: Adam bias correction, usually not needed.
        :param safeguard_warmup: remove the lr from the estimate statistics, safer with a warmup schedule.
        :param d0: initial learning rate estimate.
        :param d_coef: scale of the estimate, e.g. 0.5 for a smaller or 2 for a larger learning rate.
        :param growth_rate: max multiplicative growth of the estimate per step.
        :param slice_p: use one value out of slice_p to compute the estimate, saves memory for large models.
        """
        if not 0.0 < d0 or not 0.0 < lr or not 0.0 < eps:
            raise ValueError("d0, lr and eps must be > 0")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1[")
        defaults = dict(
            lr=lr, betas=betas, beta3=beta3, eps=eps, weight_decay=weight_decay, d=d0, d0=d0, d_max=d0,
            d_numerator=0.0, d_coef=d_coef, k=0, growth_rate=growth_rate, use_bias_correction=use_bias_correction,
            decouple=decouple, safeguard_warmup=safeguard_warmup, slice_p=slice_p,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        beta3 = group["beta3"] if group["beta3"] is not None else math.sqrt(beta2)
        k, d, d_max, d_coef = group["k"], group["d"], group["d_max"], group["d_coef"]
        lr = max(g["lr"] for g in self.param_groups)
        bias_correction = 1.0
        if group["use_bias_correction"]:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        dlr = d * lr * bias_correction
        d_numerator = group["d_numerator"] * beta3
        delta_numerator, d_denom = 0.0, 0.0

        # Statistics of the learning rate estimate
        for group in self.param_groups:
            decay, group_lr, d0, slice_p = group["weight_decay"], group["lr"], group["d0"], group["slice_p"]
            if group_lr not in (lr, 0.0):
                raise RuntimeError("Prodigy only supports one lr for all the parameter groups, or 0 to freeze one")
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Prodigy does not support sparse gradients")
                grad = p.grad
                if decay != 0 and not group["decouple"]:
                    grad.add_(p, alpha=decay)
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["s"] = torch.zeros_like(p.flatten()[::slice_p])
                    if p.any():
                        state["p0"] = p.flatten()[::slice_p].clone()
                    else:
                        state["p0"] = torch.tensor(0, device=p.device, dtype=p.dtype)
                    if beta1 > 0:
                        state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                if group_lr > 0.0:
                    sliced_grad = grad.flatten()[::slice_p]
                    distance = state["p0"] - p.flatten()[::slice_p]
                    delta_numerator += (d / d0) * dlr * torch.dot(sliced_grad, distance).item()
                    if beta1 > 0:
                        state["exp_avg"].mul_(beta1).add_(grad, alpha=d * (1 - beta1))
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))
                    alpha = (d / d0) * d if group["safeguard_warmup"] else (d / d0) * dlr
                    state["s"].mul_(beta3).add_(sliced_grad, alpha=alpha)
                    d_denom += state["s"].abs().sum().item()

        if d_denom == 0:
            return loss
        d_hat = d
        global_d_numerator = d_numerator + delta_numerator
        if lr > 0.0:
            d_hat = d_coef * global_d_numerator / d_denom
            if d == self.param_groups[0]["d0"]:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * self.param_groups[0]["growth_rate"])

        # Adam step with the estimated learning rate
        for group in self.param_groups:
            group.update(d_numerator=global_d_numerator, d_denom=d_denom, d=d, d_max=d_max, d_hat=d_hat)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] += 1
                denom = state["exp_avg_sq"].sqrt().add_(d * group["eps"])
                if group["weight_decay"] != 0 and group["decouple"]:
                    p.add_(p, alpha=-group["weight_decay"] * dlr)
                if beta1 > 0:
                    p.addcdiv_(state["exp_avg"], denom, value=-dlr)
                else:
                    p.addcdiv_(p.grad, denom, value=-dlr * d)
            group["k"] += 1
        return loss
