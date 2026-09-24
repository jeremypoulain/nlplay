import importlib
import warnings

import pytest
import torch
from torch import nn

from nlplay.models.pytorch.dropout.dropout import LockedDropout, WeightDrop, WordDropout, embedding_dropout
from nlplay.models.pytorch.optimizer.lookahead import Lookahead


def test_embedding_dropout_padding_and_training_flag():
    emb = nn.Embedding(10, 4)
    embedding_dropout(emb, torch.tensor([[1, 9]]), dropout=0.0).sum().backward()
    # Without padding_idx, the last vocabulary row must be trained like the others
    assert emb.weight.grad[9].abs().sum() > 0
    words = torch.arange(10).unsqueeze(0)
    assert torch.equal(embedding_dropout(emb, words, dropout=0.5, training=False), emb(words))
    torch.manual_seed(0)
    dropped = embedding_dropout(emb, words, dropout=0.5)
    rows_zero = (dropped[0].abs().sum(1) == 0)
    assert rows_zero.any() and not rows_zero.all()
    kept = ~rows_zero
    assert torch.allclose(dropped[0][kept], 2 * emb(words)[0][kept])


def test_locked_and_word_dropout():
    x = torch.ones(4, 7, 5)
    torch.manual_seed(0)
    out = LockedDropout(0.5).train()(x)
    # Same mask at every time step
    assert torch.equal(out[:, :1].expand_as(out), out)
    assert torch.equal(LockedDropout(0.5).eval()(x), x)
    out = WordDropout(0.5).train()(x)
    assert torch.equal(out[..., :1].expand_as(out), out)
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_weight_drop_lstm():
    torch.manual_seed(0)
    lstm = WeightDrop(nn.LSTM(4, 6, batch_first=True), ["weight_hh_l0"], dropout=0.5)
    x = torch.randn(3, 5, 4)
    lstm.train()
    out1, _ = lstm(x)
    out2, _ = lstm(x)
    assert not torch.allclose(out1, out2)
    out1.sum().backward()
    assert lstm.module.weight_hh_l0_raw.grad is not None
    lstm.eval()
    assert torch.allclose(lstm(x)[0], lstm(x)[0])
    names = {n for n, _ in lstm.named_parameters()}
    assert "module.weight_hh_l0_raw" in names and "module.weight_hh_l0" not in names


def test_lookahead():
    torch.manual_seed(0)
    model = nn.Linear(5, 2)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    w0 = initial["weight"]
    opt = Lookahead(torch.optim.SGD(model.parameters(), lr=0.1), alpha=0.5, k=3)
    x, y = torch.randn(16, 5), torch.randn(16, 2)
    fast = []
    for _ in range(3):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        before = model.weight.detach().clone()
        opt.step()
        fast.append(before)
    # After k steps the weights are halfway between the initial (slow) weights and the fast ones
    reference = nn.Linear(5, 2)
    reference.load_state_dict(initial)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
    for _ in range(3):
        ref_opt.zero_grad()
        nn.functional.mse_loss(reference(x), y).backward()
        ref_opt.step()
    assert torch.allclose(model.weight, w0 + 0.5 * (reference.weight - w0), atol=1e-6)
    state = opt.state_dict()
    other = Lookahead(torch.optim.SGD(model.parameters(), lr=0.1), alpha=0.5, k=3)
    other.load_state_dict(state)
    assert other.step_count == 3 and torch.equal(other.slow_weights[0], opt.slow_weights[0])


@pytest.mark.parametrize("module,name", [
    ("adabelief", "AdaBelief"), ("adabound", "AdaBound"), ("adabound", "AdaBoundW"), ("diffgrad", "DiffGrad"),
    ("qhadam", "QHAdam"), ("radam", "RAdam"), ("radam", "PlainRAdam"), ("ranger", "Ranger"),
])
def test_optimizers_learn_without_warnings(module, name):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 3))
    x, y = torch.randn(64, 10), torch.randint(0, 3, (64,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cls = getattr(importlib.import_module(f"nlplay.models.pytorch.optimizer.{module}"), name)
        opt = cls(model.parameters(), lr=1e-2)
        first = nn.functional.cross_entropy(model(x), y).item()
        for _ in range(100):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
    assert nn.functional.cross_entropy(model(x), y).item() < first
