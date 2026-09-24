"""
Title   : Cyclical Learning Rates for Training Neural Networks - 2015
Author  : Leslie N. Smith
Papers  : https://arxiv.org/pdf/1506.01186.pdf
Source  : https://github.com/davidtvs/pytorch-lr-finder, https://github.com/fastai/fastai
Note    : Learning rate range test: train on a few batches while increasing the learning rate, then pick a
          rate where the loss decreases the fastest. The model and optimizer states, learning rates included,
          are restored afterwards, optimizers with a scheduler attached or combining several optimizers
          (CombinedOptimizer) are supported, and the loss is computed as in the PytorchModelTrainer.
"""
import copy
import logging
import math
import os
import random
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm.autonotebook import tqdm


@dataclass
class LRFinderResult:
    """Outcome of a range test: tested rates, smoothed losses, suggestion and early stop."""

    lrs: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    suggested_lr: float | None = None
    method: str = "min_div_10"
    diverged: bool = False


def _call_model(model, inputs):
    # Tuples / lists are positional inputs, mappings keyword inputs
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def compute_training_loss(model, criterion, inputs, labels):
    """
    Training loss as computed by the PytorchModelTrainer.
    :param model: model, its training_loss(inputs, labels) is used when it returns a value (e.g. hierarchical
        softmax), and its regularization_loss() is added when defined (e.g. LEAM).
    :param criterion: loss applied to the model outputs otherwise.
    :param inputs: batch inputs, a tuple / list → model(*inputs), a dict → model(**inputs).
    :param labels: batch labels.
    :returns: the scalar loss tensor.
    """
    loss = model.training_loss(inputs, labels) if hasattr(model, "training_loss") else None
    if loss is None:
        loss = criterion(_call_model(model, inputs), labels)
    if hasattr(model, "regularization_loss"):
        loss = loss + model.regularization_loss()
    return loss


def set_learning_rate(optimizer, lr: float, scheduler=None) -> None:
    """
    Set a new peak learning rate, also as the base rate of an attached scheduler.
    :param optimizer: torch optimizer or CombinedOptimizer.
    :param lr: new learning rate.
    :param scheduler: optional attached scheduler or CombinedScheduler, its base rates are updated too.
    """
    for group in optimizer.param_groups:
        group["lr"] = lr
        if "initial_lr" in group:
            group["initial_lr"] = lr
    if scheduler is not None:
        for s in getattr(scheduler, "schedulers", [scheduler]):
            if hasattr(s, "base_lrs"):
                s.base_lrs = [lr] * len(s.base_lrs)


class LRFinder(object):
    """
    Learning rate range test.

    Example:
        >>> lr_finder = LRFinder(model, optimizer, criterion)
        >>> suggested_lr = lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=10, num_iter=100)
        >>> lr_finder.plot(output_path="lr_finder.png")
        >>> lr_finder.reset()  # done automatically at the end of range_test
    """

    def __init__(self, model, optimizer, criterion, device=None, memory_cache=True, cache_dir=None):
        """
        :param model: model to test.
        :param optimizer: optimizer (or CombinedOptimizer), a scheduler may be attached, it is not stepped.
        :param criterion: loss function, see compute_training_loss.
        :param device: device of the test, the model device if None.
        :param memory_cache: keep the initial states in memory (True) or in cache_dir files.
        :param cache_dir: directory of the state files, the system temporary directory if None.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.suggested_lr = None
        self.result = LRFinderResult()
        self.model_device = next(self.model.parameters()).device
        self._was_training = self.model.training
        self._rng_states = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )
        self.device = device or self.model_device
        self.state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
        self.state_cacher.store("model", self.model.state_dict())
        self.state_cacher.store("optimizer", self.optimizer.state_dict())
        # Learning rates are not always part of the optimizer state_dict (e.g. CombinedOptimizer)
        self._initial_groups = [
            {k: group[k] for k in ("lr", "initial_lr") if k in group} for group in self.optimizer.param_groups
        ]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Remove the cached state files, when memory_cache is False."""
        self.state_cacher.close()

    def reset(self):
        """
        Restore the model, the optimizer state, the learning rates, the train / eval mode and the random
        number generators to their initial states, so that the range test has no side effect.
        """
        self.model.load_state_dict(self.state_cacher.retrieve("model"))
        self.optimizer.load_state_dict(self.state_cacher.retrieve("optimizer"))
        for group, initial in zip(self.optimizer.param_groups, self._initial_groups):
            group.update(initial)
        self.model.to(self.model_device)
        self.model.train(self._was_training)
        py_state, np_state, torch_state, cuda_states = self._rng_states
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    def range_test(
        self,
        train_loader,
        val_loader=None,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        step_mode: str = "exp",
        smooth_f: float = 0.05,
        diverge_th: float = 5,
        accumulation_steps: int = 1,
        suggestion: str = "min_div_10",
        reset: bool = True,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ):
        """
        :param train_loader: training DataLoader, iterated again when exhausted.
        :param val_loader: optional DataLoader, its loss is used instead of the training loss (slower).
        :param start_lr: first learning rate tested.
        :param end_lr: last learning rate tested.
        :param num_iter: number of tested learning rates, one optimizer step each.
        :param step_mode: "exp" (geometric spacing, the usual choice) or "linear".
        :param smooth_f: weight of the new loss in the bias corrected exponential moving average, 0 → none.
        :param diverge_th: stop when the smoothed loss exceeds diverge_th * best loss (or is not finite).
        :param accumulation_steps: batches accumulated per optimizer step.
        :param suggestion: "min_div_10" (lr of the lowest loss / 10, the most reliable on AG News over 5 losses
            and 2 optimizers), "valley" (inside the longest decreasing part of the curve, fastai default) or
            "steepest" (steepest decrease of the smoothed loss).
        :param reset: restore the model and optimizer at the end of the test.
        :param amp: native mixed precision (torch.autocast), with a GradScaler for float16 on cuda.
        :param amp_dtype: autocast dtype, torch.float16 or torch.bfloat16.
        :returns: the suggested learning rate, None if the test is too short to suggest one, the full
            outcome is stored in self.result.
        """
        if not 0 < start_lr < end_lr:
            raise ValueError("start_lr must be positive and lower than end_lr")
        if not 0 <= smooth_f < 1:
            raise ValueError("smooth_f is outside the range [0, 1[")
        if step_mode == "exp":
            lrs = np.geomspace(start_lr, end_lr, num_iter)
        elif step_mode == "linear":
            lrs = np.linspace(start_lr, end_lr, num_iter)
        else:
            raise ValueError(f"expected one of (exp, linear), got {step_mode}")

        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.model.to(self.device)
        device_type = torch.device(self.device).type
        self._amp = (amp, amp_dtype, device_type)
        use_scaler = amp and amp_dtype == torch.float16 and device_type == "cuda"
        # torch.amp.GradScaler exists since PyTorch 2.3, torch.cuda.amp.GradScaler before
        if hasattr(torch.amp, "GradScaler"):
            self._scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)
        else:
            self._scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        diverged = False
        avg_loss, beta = 0.0, 1.0 - smooth_f
        iter_wrapper = DataLoaderIterWrapper(train_loader)
        # Schedule-free optimizers must be in train mode to step, reset restores their mode
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        try:
            for iteration, lr in enumerate(tqdm(lrs)):
                # The recorded learning rate is the one used for this step
                for group in self.optimizer.param_groups:
                    group["lr"] = float(lr)
                loss = self._train_batch(iter_wrapper, accumulation_steps)
                if val_loader is not None:
                    loss = self._validate(val_loader)

                if smooth_f > 0:
                    avg_loss = beta * avg_loss + (1 - beta) * loss
                    loss = avg_loss / (1 - beta ** (iteration + 1))
                self.history["lr"].append(float(lr))
                self.history["loss"].append(loss)
                if not math.isfinite(loss) or (self.best_loss is not None and loss > diverge_th * self.best_loss):
                    logging.info("LR Finder stopping early, the loss has diverged")
                    diverged = True
                    break
                if self.best_loss is None or loss < self.best_loss:
                    self.best_loss = loss
        finally:
            if reset:
                self.reset()

        self.suggested_lr = self.suggestion(suggestion)
        self.result = LRFinderResult(
            list(self.history["lr"]), list(self.history["loss"]), self.suggested_lr, suggestion, diverged
        )
        if self.suggested_lr is not None:
            logging.info("LR Finder suggested learning rate: {:.3g}".format(self.suggested_lr))
        return self.suggested_lr

    def suggestion(
        self, method: str = "min_div_10", skip_start: int = 10, skip_end: int = 5, min_decrease: float = 0.01
    ):
        """
        :param method: "valley", "steepest" or "min_div_10", see range_test.
        :param skip_start: first points ignored by "steepest", the smoothed loss is not meaningful yet.
        :param skip_end: last points ignored by "steepest", usually the divergence.
        :param min_decrease: minimal relative decrease of the loss over the test, below it or below 3 times the
            batch to batch noise of the first points, the curve is flat (the model did not learn at any tested
            rate, e.g. end_lr too low) and nothing is suggested.
        :returns: the suggested learning rate, None if not enough points or a flat curve.
        """
        if method not in ("valley", "steepest", "min_div_10"):
            raise ValueError(f"Unknown suggestion method: {method}")
        lrs = np.asarray(self.history["lr"])
        losses = np.asarray(self.history["loss"])
        finite = np.isfinite(losses)
        lrs, losses = lrs[finite], losses[finite]
        if len(lrs) < 5:
            return None
        head = losses[: max(5, len(losses) // 10)]
        baseline, noise = float(np.median(head)), float(np.std(head))
        if baseline - losses.min() < max(min_decrease * abs(baseline), 3 * noise):
            logging.warning(
                "LR Finder: the loss did not decrease by {:.0%} over the test, no learning rate suggested, "
                "try a higher end_lr or more iterations".format(min_decrease)
            )
            return None
        if method == "min_div_10":
            return float(lrs[losses.argmin()] / 10)
        if method == "valley":
            # Longest decreasing subsequence of the loss, the rate 2/3 into it (fastai valley)
            n = len(losses)
            lds, max_start, max_end = [1] * n, 0, 0
            for i in range(1, n):
                for j in range(i):
                    if losses[i] < losses[j] and lds[i] < lds[j] + 1:
                        lds[i] = lds[j] + 1
                if lds[max_end] < lds[i]:
                    max_end, max_start = i, i - lds[i]
            sections = (max_end - max_start) / 3
            return float(lrs[max_start + int(sections) + int(sections / 2)])
        end = len(lrs) - skip_end
        if end - skip_start < 3:
            return None
        lrs, losses = lrs[skip_start:end], losses[skip_start:end]
        # Steepest descent of the loss against log(lr), the scale of the range test
        slope = np.gradient(losses, np.log(lrs))
        return float(lrs[slope.argmin()])

    def _train_batch(self, iter_wrapper, accumulation_steps):
        self.model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        use_amp, amp_dtype, device_type = self._amp
        total_loss = 0.0
        self.optimizer.zero_grad()
        for _ in range(accumulation_steps):
            inputs, labels = self._move_to_device(*next(iter_wrapper))
            with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                loss = compute_training_loss(self.model, self.criterion, inputs, labels) / accumulation_steps
            self._scaler.scale(loss).backward()
            total_loss += loss.item()
        # A CombinedOptimizer is stepped optimizer by optimizer, the scaler only knows torch optimizers
        for opt in getattr(self.optimizer, "optimizers", [self.optimizer]):
            self._scaler.step(opt)
        self._scaler.update()
        return total_loss

    def _move_to_device(self, inputs, labels):
        def move(obj, device):
            if hasattr(obj, "to"):
                return obj.to(device)
            elif isinstance(obj, tuple):
                return tuple(move(o, device) for o in obj)
            elif isinstance(obj, list):
                return [move(o, device) for o in obj]
            elif isinstance(obj, dict):
                return {k: move(o, device) for k, o in obj.items()}
            return obj

        return move(inputs, self.device), move(labels, self.device)

    def _validate(self, dataloader):
        running_loss, count = 0.0, 0
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()
        with torch.no_grad():
            use_amp, amp_dtype, device_type = self._amp
            for inputs, labels, *_ in dataloader:
                inputs, labels = self._move_to_device(inputs, labels)
                batch_size = len(labels)
                # The criterion is assumed to average over the batch (reduction="mean")
                with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                    loss = compute_training_loss(self.model, self.criterion, inputs, labels)
                running_loss += loss.item() * batch_size
                count += batch_size
        return running_loss / max(count, 1)

    def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None, ax=None, show=False, output_path=None):
        """
        :param skip_start: first points not plotted.
        :param skip_end: last points not plotted.
        :param log_lr: logarithmic learning rate axis.
        :param show_lr: learning rate marked with a vertical line, the suggested one if None.
        :param ax: matplotlib axes to draw on, a new figure otherwise.
        :param show: show the figure.
        :param output_path: optional png file path.
        :returns: the matplotlib axes.
        """
        import matplotlib.pyplot as plt

        if skip_start < 0 or skip_end < 0:
            raise ValueError("skip_start and skip_end cannot be negative")
        lrs = self.history["lr"][skip_start: len(self.history["lr"]) - skip_end]
        losses = self.history["loss"][skip_start: len(self.history["loss"]) - skip_end]
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(lrs, losses)
        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")
        show_lr = self.suggested_lr if show_lr is None else show_lr
        if show_lr is not None:
            ax.axvline(x=show_lr, color="red", linestyle="--", label=f"lr = {show_lr:.3g}")
            ax.legend()
        if output_path is not None:
            ax.figure.savefig(output_path, format="png")
        if fig is not None:
            if show:
                plt.show()
            else:
                plt.close(fig)
        return ax


class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir or tempfile.gettempdir()
        if not os.path.isdir(self.cache_dir):
            raise ValueError("Given `cache_dir` is not a valid directory.")
        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached[key] = copy.deepcopy(state_dict)
        else:
            fn = os.path.join(self.cache_dir, "state_{}_{}.pt".format(key, id(self)))
            self.cached[key] = fn
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError("Target {} was not cached.".format(key))
        if self.in_memory:
            # A copy, so that the cached state survives a reset followed by more training
            return copy.deepcopy(self.cached[key])
        fn = self.cached[key]
        if not os.path.exists(fn):
            raise RuntimeError("Failed to load state in {}. File doesn't exist anymore.".format(fn))
        return torch.load(fn, map_location=lambda storage, location: storage, weights_only=False)

    def close(self):
        """Remove the cached files."""
        if self.in_memory:
            return
        for fn in self.cached.values():
            if os.path.exists(fn):
                os.remove(fn)

    def __del__(self):
        # Fallback only, prefer LRFinder.close() or the LRFinder context manager
        self.close()


class DataLoaderIterWrapper(object):
    """Iterate a DataLoader, starting it again when exhausted."""

    def __init__(self, data_loader, auto_reset=True):
        self.data_loader = data_loader
        self.auto_reset = auto_reset
        self._iterator = iter(data_loader)

    def __next__(self):
        try:
            inputs, labels, *_ = next(self._iterator)
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            inputs, labels, *_ = next(self._iterator)
        return inputs, labels
