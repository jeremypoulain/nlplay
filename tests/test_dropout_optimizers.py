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


def _toy_problem():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 3))
    return model, torch.randn(64, 10), torch.randint(0, 3, (64,))


@pytest.mark.parametrize("name", ["AdamWScheduleFree", "SGDScheduleFree"])
def test_schedule_free_train_eval_modes(name):
    from nlplay.models.pytorch.optimizer import schedulefree

    model, x, y = _toy_problem()
    opt = getattr(schedulefree, name)(model.parameters(), lr=1e-2 if name.startswith("AdamW") else 0.1)
    with pytest.raises(RuntimeError):
        opt.step()  # train() not called
    opt.train()
    first = nn.functional.cross_entropy(model(x), y).item()
    for _ in range(100):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    train_point = [p.detach().clone() for p in model.parameters()]
    opt.eval()
    assert nn.functional.cross_entropy(model(x), y).item() < first
    assert any(not torch.equal(a, p) for a, p in zip(train_point, model.parameters()))
    opt.train()
    assert all(torch.allclose(a, p, atol=1e-6) for a, p in zip(train_point, model.parameters()))


def test_prodigy_learns_with_default_lr():
    from nlplay.models.pytorch.optimizer.prodigy import Prodigy

    model, x, y = _toy_problem()
    opt = Prodigy(model.parameters())
    first = nn.functional.cross_entropy(model(x), y).item()
    for _ in range(100):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    assert nn.functional.cross_entropy(model(x), y).item() < 0.5 * first
    assert opt.param_groups[0]["d"] > opt.param_groups[0]["d0"]


def test_fasttext_optimizer_new_optimizers_and_trainer(tmp_path, monkeypatch):
    # The trainer writes its checkpoints and plots in the working directory
    monkeypatch.chdir(tmp_path)
    from torch.utils.data import TensorDataset
    from nlplay.models.pytorch.classifiers.fasttext_hashed import (
        CombinedOptimizer, HashedFastText, fasttext_optimizer)
    from nlplay.models.pytorch.optimizer.schedulefree import AdamWScheduleFree
    from nlplay.models.pytorch.trainer import PytorchModelTrainer

    dense = HashedFastText(49, 2, 8, fasttext_init=False)
    opt, sched = fasttext_optimizer(dense, lr=1e-2, total_steps=None, optimizer="adamw_schedulefree", warmup_steps=5)
    assert isinstance(opt, AdamWScheduleFree) and sched is None and opt.param_groups[0]["warmup_steps"] == 5
    sparse = HashedFastText(49, 2, 8, sparse=True, fasttext_init=False)
    opt, sched = fasttext_optimizer(sparse, lr=1e-2, total_steps=100, optimizer="adamw_schedulefree")
    assert isinstance(opt, CombinedOptimizer) and sched is not None
    opt, sched = fasttext_optimizer(dense, lr=1.0, total_steps=100, optimizer="prodigy", warmup_steps=5)
    assert opt.param_groups[0]["safeguard_warmup"]
    with pytest.raises(ValueError):
        fasttext_optimizer(dense, lr=0.1, total_steps=None, optimizer="sgd")

    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 49, (800, 6), generator=g)
    y = (x == 7).any(1).long()
    for name, lr in (("adamw_schedulefree", 2e-2), ("sgd_schedulefree", 2.0), ("prodigy", 1.0)):
        torch.manual_seed(0)
        model = HashedFastText(49, 2, 16, fasttext_init=False)
        opt, sched = fasttext_optimizer(model, lr=lr, total_steps=8 * 25, optimizer=name)
        trainer = PytorchModelTrainer(model, nn.CrossEntropyLoss(), opt, lr_scheduler=sched,
                                      train_ds=TensorDataset(x[:600], y[:600]),
                                      val_ds=TensorDataset(x[600:], y[600:]),
                                      batch_size=32, epochs=8, early_stopping=False)
        trainer.train_evaluate(check_dl=False, run_lr_finder=name == "adamw_schedulefree")
        assert trainer.best_score > 0.9, name
