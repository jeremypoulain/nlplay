import os

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

import nlplay.models.pytorch.trainer as trainer_module
from nlplay.models.pytorch.trainer import PytorchModelTrainer


def _datasets():
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    y = (x[:, 0] > 0).long()
    return TensorDataset(x, y), TensorDataset(x[:32], y[:32])


def _trainer(tmp_path, epochs, val_scores, **kwargs):
    """Trainer whose validation accuracy of each epoch is val_scores[epoch], the model state at each
    validation is recorded in trainer.epoch_states."""
    train_ds, val_ds = _datasets()
    model = nn.Linear(4, 2)
    trainer = PytorchModelTrainer(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.5),
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=16,
        epochs=epochs,
        model_output_folder=str(tmp_path / "models"),
        **kwargs,
    )
    scores = iter(val_scores)
    trainer.epoch_states = []

    def fake_accuracy(model, dl, device=None):
        if dl is trainer.val_dl:
            trainer.epoch_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return torch.tensor(next(scores))
        return torch.tensor(0.5)

    return trainer, fake_accuracy


def test_checkpoint_path_uses_model_name_and_suffix(tmp_path):
    trainer, _ = _trainer(tmp_path, 1, [])
    assert trainer.checkpoint_path == os.path.join(str(tmp_path / "models"), "checkpoint_Linear.pt")
    trainer.checkpoint_file_suffix = "run1"
    assert trainer.checkpoint_path.endswith("checkpoint_Linear_run1.pt")
    trainer.save_checkpoint()
    assert os.path.isfile(trainer.checkpoint_path)


def test_best_epoch_without_early_stopping(tmp_path, monkeypatch):
    trainer, fake_accuracy = _trainer(tmp_path, 4, [0.6, 0.9, 0.7, 0.8], early_stopping=False)
    monkeypatch.setattr(trainer_module, "compute_accuracy", fake_accuracy)
    trainer.train_evaluate(check_dl=False)
    assert trainer.best_epoch == 2 and trainer.best_score == pytest.approx(0.9)
    assert not trainer.early_stop and len(trainer.epoch_states) == 4


def test_early_stopping_and_best_weights_restored(tmp_path, monkeypatch):
    trainer, fake_accuracy = _trainer(
        tmp_path, 10, [0.6, 0.8, 0.8, 0.7, 0.75, 0.9], early_stopping_patience=3
    )
    monkeypatch.setattr(trainer_module, "compute_accuracy", fake_accuracy)
    trainer.train_evaluate(check_dl=False)
    # A tie is not an improvement, the earliest best epoch is kept, stop after 3 epochs without one
    assert trainer.early_stop and len(trainer.epoch_states) == 5
    assert trainer.best_epoch == 2 and trainer.best_score == pytest.approx(0.8)
    best, last = trainer.epoch_states[1], trainer.epoch_states[-1]
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value.cpu(), best[name].cpu())
    assert not all(torch.equal(best[k], last[k]) for k in best)


def test_mixed_precision_without_apex_trains_in_fp32(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "APEX_AVAILABLE", False)
    trainer, fake_accuracy = _trainer(tmp_path, 2, [0.6, 0.7], use_mixed_precision=True)
    monkeypatch.setattr(trainer_module, "compute_accuracy", fake_accuracy)
    assert trainer.apex is False
    trainer.train_evaluate(check_dl=False)
    assert trainer.best_epoch == 2 and os.path.isfile(trainer.checkpoint_path)


def test_load_checkpoint_maps_to_cpu(tmp_path):
    trainer, _ = _trainer(tmp_path, 1, [])
    trainer.save_checkpoint()
    with torch.no_grad():
        trainer.model.weight.zero_()
    trainer.load_checkpoint(trainer.checkpoint_path)
    assert trainer.model.weight.abs().sum() > 0
