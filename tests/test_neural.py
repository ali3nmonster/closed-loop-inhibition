"""Causality, padding isolation and trainability checks for neural baselines."""

import pytest

torch = pytest.importorskip("torch")

from closed_loop_inhibition.neural import CausalTransformer, make_model


@pytest.fixture(autouse=True)
def small_deterministic_cpu_models():
    # Tiny matrices are slower with the machine's large default thread pool.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10)
        yield
    torch.set_num_threads(previous_threads)


def test_future_tokens_cannot_change_earlier_transformer_features():
    model = CausalTransformer().eval()
    tokens = torch.randn(2, 11, 10)
    valid = torch.ones(2, 11, dtype=torch.bool)
    changed = tokens.clone()
    changed[:, 5:] = 20 * torch.randn_like(changed[:, 5:])
    with torch.no_grad():
        original_features = model.forward_tokens(tokens, valid)
        changed_features = model.forward_tokens(changed, valid)
    torch.testing.assert_close(original_features[:, :5], changed_features[:, :5])
    assert not torch.allclose(original_features[:, 5:], changed_features[:, 5:])


@pytest.mark.parametrize("kind", ["transformer", "mlp"])
def test_padding_contents_and_padding_length_do_not_change_action(kind):
    model = make_model(kind).eval()
    short = torch.randn(2, 4, 10)
    short_valid = torch.ones(2, 4, dtype=torch.bool)
    padded = torch.full((2, 11, 10), float("nan"))
    padded[:, :4] = short
    padded_valid = torch.arange(11).expand(2, -1) < 4
    with torch.no_grad():
        short_actions = model(short, short_valid)
        padded_actions = model(padded, padded_valid)
    assert torch.isfinite(padded_actions).all()
    torch.testing.assert_close(short_actions, padded_actions, atol=1e-6, rtol=1e-5)


def test_transformer_readout_uses_last_valid_token():
    model = CausalTransformer().eval()
    tokens = torch.randn(3, 11, 10)
    lengths = torch.tensor([1, 5, 11])
    valid = torch.arange(11).unsqueeze(0) < lengths.unsqueeze(1)
    with torch.no_grad():
        batched = model(tokens, valid)
        individual = torch.stack(
            [model(tokens[i:i + 1, :n], valid[i:i + 1, :n])[0] for i, n in enumerate(lengths)]
        )
        features = model.forward_tokens(tokens, valid)
    torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(features[~valid]) == 0


@pytest.mark.parametrize("kind", ["transformer", "mlp"])
def test_other_batch_members_cannot_change_action(kind):
    model = make_model(kind).eval()
    tokens = torch.randn(3, 11, 10)
    valid = torch.arange(11).unsqueeze(0) < torch.tensor([3, 7, 11]).unsqueeze(1)
    changed = tokens.clone()
    changed[1:] *= 100
    with torch.no_grad():
        original_action = model(tokens, valid)[0]
        changed_action = model(changed, valid)[0]
    torch.testing.assert_close(original_action, changed_action)


@pytest.mark.parametrize("kind", ["transformer", "mlp"])
def test_models_have_finite_gradients_and_ignore_padding_in_backward(kind):
    model = make_model(kind).train()
    tokens = torch.randn(4, 11, 10, requires_grad=True)
    valid = torch.arange(11).unsqueeze(0) < torch.tensor([1, 4, 8, 11]).unsqueeze(1)
    output = model(tokens, valid)
    assert output.shape == (4,)
    assert torch.isfinite(output).all()
    loss = (output - torch.tensor([-0.8, 0.2, 0.7, -0.1])).square().mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert torch.count_nonzero(tokens.grad[~valid]) == 0
    assert torch.count_nonzero(tokens.grad[valid]) > 0


@pytest.mark.parametrize("kind", ["transformer", "mlp"])
@pytest.mark.parametrize(
    "valid",
    [torch.zeros(1, 3, dtype=torch.bool), torch.tensor([[True, False, True]])],
)
def test_empty_or_non_prefix_histories_are_rejected(kind, valid):
    model = make_model(kind)
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 10), valid)


@pytest.mark.parametrize("kind", ["transformer", "mlp"])
def test_history_beyond_configured_capacity_is_rejected(kind):
    model = make_model(kind)
    with pytest.raises(ValueError, match="time must be"):
        model(torch.randn(1, 12, 10), torch.ones(1, 12, dtype=torch.bool))


def test_layers_start_with_different_attention_weights():
    model = CausalTransformer()
    assert not torch.equal(
        model.layers[0].self_attn.in_proj_weight,
        model.layers[1].self_attn.in_proj_weight,
    )


def test_unknown_model_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be"):
        make_model("unknown")
