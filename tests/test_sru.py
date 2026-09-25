import pytest
import torch

from nlplay.models.pytorch.classifiers.sru import sru_functional as sf
from nlplay.models.pytorch.classifiers.sru.srupp_model import (
    SRUpp,
    SRUppAttention,
    SRUppClassifier,
    SRUppProjectedLinear,
)

pytestmark = pytest.mark.filterwarnings("ignore:Running SRU on CPU")


def tokens(batch=4, seq_len=12, vocab=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, vocab, (batch, seq_len), generator=g)


def test_sru_cell_runs_on_cpu():
    torch.manual_seed(0)
    cell = sf.SRUCell(8, 8, dropout=0.2)
    x = torch.randn(5, 3, 8, requires_grad=True)
    h, c = cell(x)
    h.sum().backward()
    assert h.shape == (5, 3, 8) and c.shape == (3, 8)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_cpu_dropout_matches_cuda_kernel_formula():
    # One step from c0 = 0: c = (1 - f) u0 and h = x + (c - x) * mask * r, as in sru_cuda_kernel.cu
    torch.manual_seed(0)
    d, batch = 4, 2
    u = torch.randn(1, batch, d * 3)
    x = torch.randn(1, batch, d)
    weight_c = torch.randn(2 * d)
    bias = torch.randn(2 * d)
    mask_c = torch.bernoulli(torch.full((batch, d), 0.5)) * 2
    h, _ = sf.SRU_Compute_CPU.apply(u, x, weight_c, bias, x.new_zeros(batch, d), 0, d, False, True, None, mask_c)

    uk = u.view(batch, d, 3)
    f = torch.sigmoid(uk[..., 1] + bias[:d])
    r = torch.sigmoid(uk[..., 2] + bias[d:])
    c = (1 - f) * uk[..., 0]
    expected = x[0] + (c - x[0]) * mask_c * r
    torch.testing.assert_close(h[0], expected)


def test_bidirectional_final_state_is_per_sample():
    # The final state of a sample must not depend on the other samples of the batch
    torch.manual_seed(0)
    cell = sf.SRUCell(6, 3, bidirectional=True)
    x = torch.randn(5, 4, 6)
    _, c = cell(x)
    for i in range(4):
        _, c_i = cell(x[:, i : i + 1])
        torch.testing.assert_close(c[i : i + 1], c_i)


def test_srupp_layer_layout():
    enc = SRUpp(16, 8, 4, num_layers=4, attention_every_n_layers=2)
    kinds = [type(cell.custom_m) for cell in enc.rnn_lst]
    assert kinds == [SRUppProjectedLinear, SRUppAttention, SRUppProjectedLinear, SRUppAttention]
    # First layer projects the skip term since 16 != 8
    assert [cell.num_matrices for cell in enc.rnn_lst] == [4, 3, 3, 3]


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("pooling", ["mean", "max", "mean_max"])
def test_srupp_classifier_shapes_and_grads(bidirectional, pooling):
    torch.manual_seed(0)
    model = SRUppClassifier(3, 50, 16, hidden_size=8, proj_size=8, num_heads=2,
                            bidirectional=bidirectional, pooling=pooling)
    out = model(tokens())
    assert out.shape == (4, 3)
    out.sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing


@pytest.mark.parametrize("grad_enabled", [True, False])
@pytest.mark.parametrize("side", ["pre", "post"])
def test_srupp_ignores_padding(grad_enabled, side):
    torch.manual_seed(0)
    model = SRUppClassifier(2, 50, 16, hidden_size=8, proj_size=8).eval()
    with torch.no_grad():
        for cell in model.encoder.rnn_lst:
            cell.custom_m.alpha.fill_(1.0)
    x = tokens()
    pad = torch.zeros(4, 7, dtype=torch.long)
    padded = torch.cat([pad, x], 1) if side == "pre" else torch.cat([x, pad], 1)
    # grad_enabled False goes through the C++ CPU kernel when it compiles
    with torch.set_grad_enabled(grad_enabled):
        torch.testing.assert_close(model(x), model(padded))


def test_srupp_fully_padded_text_is_finite():
    model = SRUppClassifier(2, 50, 16, hidden_size=8, proj_size=8)
    x = tokens()
    x[1] = 0
    out = model(x)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_srupp_invalid_arguments():
    with pytest.raises(ValueError):
        SRUppClassifier(2, 50, 16, pooling="last")
    with pytest.raises(ValueError):
        SRUppAttention(8, 24, 6, num_heads=4)
