import torch
from esa.swqr import OriginalSWQR, original_swqr_recall_model, chunk_scan


def test_scan_matches_sequential_fp32():
    torch.manual_seed(17)
    a = torch.sigmoid(torch.randn(2, 19, 6, 8)) * .3 + .69
    v = torch.randn(2, 19, 6, 8) * .1
    initial = torch.randn(2, 6, 8) * .02
    values, final = chunk_scan(a, v, initial, chunk_size=7)
    state = initial.clone()
    expected = []
    for t in range(19):
        state = a[:, t] * state + v[:, t]
        expected.append(state)
    expected = torch.stack(expected, 1)
    torch.testing.assert_close(values, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final, state, rtol=1e-5, atol=1e-5)


def test_original_gate_forward_grad_and_cache():
    torch.manual_seed(99)
    layer = OriginalSWQR(embd=48, head=6, chunk_size=8)
    x = torch.randn(2, 25, 48, requires_grad=True)
    y, final = layer(x)
    assert y.shape == x.shape and final.shape == (2, 6, 8)
    (y.square().mean() + final.square().mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in layer.parameters())
    with torch.no_grad():
        a, first = layer(x.detach()[:, :12])
        b, second = layer(x.detach()[:, 12:], first)
        whole, _ = layer(x.detach())
    torch.testing.assert_close(torch.cat([a, b], 1), whole, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(final, second, rtol=2e-4, atol=2e-4)


def test_reference_model_parameter_count_and_io():
    model = original_swqr_recall_model()
    assert sum(p.numel() for p in model.parameters()) == 9_849_168
    x = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        logits, states = model(x)
    assert logits.shape == (2, 256)
    assert len(states) == 6
    assert all(s.shape == (2, 6, 64) for s in states)


def test_model_cache_chunk_equivalence():
    torch.manual_seed(17)
    model = original_swqr_recall_model(dim=384) if False else original_swqr_recall_model(embd=48, layers=2)
    model.eval()
    x = torch.randint(0, 256, (1, 19))
    with torch.no_grad():
        full, full_states = model(x)
        _, s = model(x[:, :8])
        part, part_states = model(x[:, 8:], s)
    torch.testing.assert_close(full, part, rtol=3e-4, atol=3e-4)
    for a, b in zip(full_states, part_states):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-4)
