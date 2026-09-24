"""
Title    : Lookahead Optimizer: k steps forward, 1 step back - 2019
Authors  : Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba
Papers   : https://arxiv.org/abs/1907.08610
Source   : https://github.com/lonePatient/lookahead_pytorch
           https://github.com/alphadl/lookahead.pytorch
Note     : Wrapper around any optimizer: the fast weights take k steps of the base optimizer, then the slow
           weights move by alpha towards them and the fast weights restart from the slow ones.
"""
import torch


class Lookahead:
    def __init__(self, base_optimizer, alpha: float = 0.5, k: int = 6):
        """
        :param base_optimizer: inner optimizer, e.g. torch.optim.AdamW or a CombinedOptimizer.
        :param alpha: slow weights step size, in [0, 1].
        :param k: number of fast steps between two slow updates.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid lookahead steps: {k}")
        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.k = k
        self.step_count = 0
        # Slow weights start from the initial weights, as in the paper
        self.slow_weights = [p.detach().clone() for p in self._params()]

    def _params(self):
        return [p for group in self.param_groups for p in group["params"]]

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @property
    def state(self):
        return self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def sync_lookahead(self):
        """Move the slow weights towards the fast ones and restart the fast weights from them."""
        for fast, slow in zip(self._params(), self.slow_weights):
            slow.add_(fast.detach() - slow, alpha=self.alpha)
            fast.copy_(slow)

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.step_count += 1
        if self.step_count % self.k == 0:
            self.sync_lookahead()
        return loss

    def state_dict(self) -> dict:
        # Slow weights stored in parameter order, so that the state can be loaded in another process
        return {
            "base": self.base_optimizer.state_dict(),
            "slow_weights": [w.clone() for w in self.slow_weights],
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: dict):
        self.base_optimizer.load_state_dict(state_dict["base"])
        with torch.no_grad():
            for slow, saved in zip(self.slow_weights, state_dict["slow_weights"]):
                slow.copy_(saved)
        self.step_count = state_dict["step_count"]
