import math
import os
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nlplay.models.pytorch.classifiers.fasttext_hashed import CombinedOptimizer, HashedFastText, fasttext_optimizer
from nlplay.models.pytorch.lr_finder import LRFinder


def make_loader(n=512, batch_size=32):
    g = torch.Generator().manual_seed(0)
    x = torch.randint(1, 50, (n, 6), generator=g)
    y = (x == 7).any(1).long()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(50, 8, mode="mean")
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(self.embedding(x))


def state_copy(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def test_lr_boundaries_and_alignment():
    model = TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    used = []
    opt.register_step_pre_hook(lambda o, args, kwargs: used.append(o.param_groups[0]["lr"]))
    finder = LRFinder(model, opt, nn.CrossEntropyLoss())
    finder.range_test(make_loader(), start_lr=1e-3, end_lr=1.0, num_iter=20, diverge_th=1e9)
    assert finder.history["lr"][0] == pytest.approx(1e-3)
    assert finder.history["lr"][-1] == pytest.approx(1.0)
    assert used == pytest.approx(finder.history["lr"])
    lin = LRFinder(model, opt, nn.CrossEntropyLoss())
    lin.range_test(make_loader(), start_lr=0.1, end_lr=1.0, num_iter=10, step_mode="linear", diverge_th=1e9)
    assert np.allclose(np.diff(lin.history["lr"]), 0.1)


def test_state_restored_without_side_effect():
    model = TinyModel().eval()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    weights, opt_state = state_copy(model), opt.state_dict()
    random.seed(1), np.random.seed(1), torch.manual_seed(1)
    expected_draws = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(1), np.random.seed(1), torch.manual_seed(1)
    LRFinder(model, opt, nn.CrossEntropyLoss()).range_test(make_loader(), start_lr=1e-4, end_lr=10, num_iter=30)
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
    assert opt.state_dict()["state"] == opt_state["state"]
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[0]["initial_lr"] == pytest.approx(1e-3)
    assert not model.training
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == pytest.approx(expected_draws)
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_combined_optimizer_and_training_loss_hook():
    model = HashedFastText(49, 2, 8, sparse=True, loss="hs", class_counts=[300, 212])
    opt, _ = fasttext_optimizer(model, lr=1e-2, total_steps=100, optimizer="adam")
    assert isinstance(opt, CombinedOptimizer)

    class MustNotBeCalled(nn.Module):
        def forward(self, *args):
            raise AssertionError("the model training_loss must be used")

    finder = LRFinder(model, opt, MustNotBeCalled())
    finder.range_test(make_loader(), start_lr=1e-5, end_lr=1, num_iter=20)
    assert len(finder.history["lr"]) > 0
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([1e-2, 1e-2])


def test_gradient_accumulation_averages_the_loss():
    model = TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    finder = LRFinder(model, opt, nn.CrossEntropyLoss())
    finder.range_test(make_loader(batch_size=16), start_lr=1e-9, end_lr=2e-9, num_iter=5, smooth_f=0,
                      accumulation_steps=4)
    # Zero output layer → every loss is log(2), whatever the accumulation
    nn.init.zeros_(model.fc.weight), nn.init.zeros_(model.fc.bias)
    finder = LRFinder(model, opt, nn.CrossEntropyLoss())
    finder.range_test(make_loader(batch_size=16), start_lr=1e-9, end_lr=2e-9, num_iter=5, smooth_f=0,
                      accumulation_steps=4)
    assert finder.history["loss"] == pytest.approx([math.log(2)] * 5)


@pytest.mark.parametrize("kind", ["tuple", "dict"])
def test_multi_input_models(kind):
    class TwoInputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.EmbeddingBag(50, 4), nn.EmbeddingBag(50, 4)
            self.fc = nn.Linear(8, 2)

        def forward(self, left, right):
            return self.fc(torch.cat([self.a(left), self.b(right)], 1))

    class PairDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.x = torch.randint(1, 50, (128, 6))
            self.y = (self.x == 7).any(1).long()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            inputs = (self.x[i, :3], self.x[i, 3:])
            return (inputs if kind == "tuple" else {"left": inputs[0], "right": inputs[1]}), self.y[i]

    model = TwoInputs()
    finder = LRFinder(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss())
    finder.range_test(DataLoader(PairDataset(), batch_size=16), start_lr=1e-3, end_lr=1, num_iter=10)
    assert len(finder.history["loss"]) == 10


def test_divergence_and_non_finite_losses_stop_the_test():
    model = TinyModel()
    finder = LRFinder(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss())
    finder.range_test(make_loader(), start_lr=1e-2, end_lr=1e8, num_iter=100)
    assert finder.result.diverged
    assert len(finder.history["loss"]) < 100
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_flat_curve_gives_no_suggestion():
    model = TinyModel()
    finder = LRFinder(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss())
    assert finder.range_test(make_loader(), start_lr=1e-9, end_lr=1e-8, num_iter=30) is None
    assert finder.result.suggested_lr is None


def test_suggestion_on_a_learnable_problem():
    model = TinyModel()
    finder = LRFinder(model, torch.optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss())
    lr = finder.range_test(make_loader(n=2048), start_lr=1e-6, end_lr=10, num_iter=100)
    assert lr is not None and 1e-5 < lr < 10
    assert finder.result.method == "min_div_10"
    for method in ("steepest", "valley"):
        assert finder.suggestion(method) is not None


def test_file_cache_is_removed_on_close(tmp_path):
    model = TinyModel()
    with LRFinder(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss(),
                  memory_cache=False, cache_dir=str(tmp_path)) as finder:
        finder.range_test(make_loader(), start_lr=1e-3, end_lr=1, num_iter=5)
        assert len(os.listdir(tmp_path)) == 2
    assert os.listdir(tmp_path) == []


def test_native_amp_cpu_bfloat16():
    model = TinyModel()
    finder = LRFinder(model, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss())
    finder.range_test(make_loader(), start_lr=1e-3, end_lr=1, num_iter=10, amp=True, amp_dtype=torch.bfloat16)
    assert len(finder.history["loss"]) == 10


def test_plot_on_external_axes(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = TinyModel()
    finder = LRFinder(model, torch.optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss())
    finder.range_test(make_loader(n=2048), start_lr=1e-6, end_lr=10, num_iter=60)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.figure()  # another current figure, the finder must not save it
    finder.plot(ax=ax2, show_lr=1, output_path=str(tmp_path / "plot.png"))
    assert ax2.lines and not ax1.lines
    assert (tmp_path / "plot.png").stat().st_size > 0
