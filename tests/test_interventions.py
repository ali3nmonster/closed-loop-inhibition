"""Check attention pathway isolation without changing the native controller."""

import pytest

torch = pytest.importorskip("torch")

from closed_loop_inhibition.interventions import (
    enumerate_attention_heads,
    head_residuals,
    scale_attention_head,
)
from closed_loop_inhibition.neural import CausalTransformer, HistoryMLP


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2030)
        yield
    torch.set_num_threads(previous_threads)


@pytest.fixture
def inputs():
    tokens = torch.randn(3, 11, 10)
    valid = torch.arange(11).unsqueeze(0) < torch.tensor([1, 6, 11]).unsqueeze(1)
    return tokens, valid


def test_head_identifiers_follow_layer_and_head_order():
    assert enumerate_attention_heads(CausalTransformer()) == [
        (layer, head) for layer in range(2) for head in range(4)
    ]


def test_identity_clone_preserves_exact_predictions_and_original_state(inputs):
    model = CausalTransformer().eval()
    model.layers[0].linear1.weight.requires_grad_(False)
    original_flags = [parameter.requires_grad for parameter in model.parameters()]
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    identity = scale_attention_head(model, 1, 2, 1.0)
    assert not identity.training
    assert not any(parameter.requires_grad for parameter in identity.parameters())
    assert [parameter.requires_grad for parameter in model.parameters()] == original_flags
    for name, value in identity.state_dict().items():
        torch.testing.assert_close(value, original_state[name], atol=0, rtol=0)
        assert value.data_ptr() != model.state_dict()[name].data_ptr()
    with torch.inference_mode():
        torch.testing.assert_close(identity(*inputs), model(*inputs), atol=0, rtol=0)


def test_scaling_changes_only_requested_output_projection_columns(inputs):
    model = CausalTransformer().train()
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    scaled = scale_attention_head(model, 0, 2, 0.5)
    assert model.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    for name, value in scaled.state_dict().items():
        expected = original_state[name].clone()
        if name == "layers.0.self_attn.out_proj.weight":
            expected[:, 32:48] *= 0.5
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
        torch.testing.assert_close(model.state_dict()[name], original_state[name], atol=0, rtol=0)


def test_residuals_reconstruct_native_attention_branch_and_leave_bias_separate(inputs):
    tokens, valid = inputs
    model = CausalTransformer().eval()
    with torch.no_grad():
        # Nonzero biases ensure the decomposition cannot silently assign the
        # shared output bias to any individual head.
        for layer in model.layers:
            layer.self_attn.out_proj.bias.copy_(torch.linspace(-0.3, 0.2, 64))
        residuals = head_residuals(model, tokens, valid)
        assert residuals.shape == (3, 11, 2, 4, 64)
        assert not residuals.requires_grad
        assert torch.count_nonzero(residuals[~valid]) == 0
        clean = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        hidden = model.input_projection(clean) + model.position_embedding(torch.arange(11))
        future = torch.ones(11, 11, dtype=torch.bool).triu(diagonal=1)
        for index, layer in enumerate(model.layers):
            normalized = layer.norm1(hidden)
            branch = layer.self_attn(
                normalized, normalized, normalized,
                attn_mask=future, key_padding_mask=~valid, need_weights=False,
            )[0]
            reconstructed = residuals[:, :, index].sum(dim=2) + layer.self_attn.out_proj.bias
            torch.testing.assert_close(reconstructed[valid], branch[valid], atol=2e-6, rtol=1e-5)
            hidden = layer(hidden, src_mask=future, src_key_padding_mask=~valid)


@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize("scale", [0.0, 0.5, 1.5])
def test_intervention_scales_exactly_one_head_contribution(inputs, layer_index, scale):
    model = CausalTransformer().eval()
    changed = scale_attention_head(model, layer_index, 1, scale)
    original_residuals = head_residuals(model, *inputs)
    changed_residuals = head_residuals(changed, *inputs)
    expected = original_residuals[:, :, layer_index].clone()
    expected[:, :, 1] *= scale
    torch.testing.assert_close(changed_residuals[:, :, layer_index], expected, atol=2e-6, rtol=1e-5)
    if layer_index:
        torch.testing.assert_close(changed_residuals[:, :, :layer_index], original_residuals[:, :, :layer_index])


def test_scaled_pathway_retains_future_mask_and_padding_isolation():
    model = scale_attention_head(CausalTransformer().eval(), 0, 1, 0.5)
    tokens = torch.randn(2, 11, 10)
    valid = torch.ones(2, 11, dtype=torch.bool)
    changed = tokens.clone()
    changed[:, 5:] = 100 * torch.randn_like(changed[:, 5:])
    with torch.inference_mode():
        original_features = model.forward_tokens(tokens, valid)
        changed_features = model.forward_tokens(changed, valid)
    torch.testing.assert_close(original_features[:, :5], changed_features[:, :5])
    torch.testing.assert_close(
        head_residuals(model, tokens, valid)[:, :5], head_residuals(model, changed, valid)[:, :5]
    )
    short = tokens[:, :5]
    short_valid = valid[:, :5]
    padded = tokens.clone()
    padded[:, 5:] = float("nan")
    padded_valid = torch.arange(11).expand(2, -1) < 5
    with torch.inference_mode():
        torch.testing.assert_close(model(short, short_valid), model(padded, padded_valid), atol=1e-6, rtol=1e-5)
    padded_residuals = head_residuals(model, padded, padded_valid)
    assert torch.isfinite(padded_residuals).all()
    torch.testing.assert_close(
        head_residuals(model, short, short_valid), padded_residuals[:, :5], atol=2e-6, rtol=1e-5
    )


def test_restoring_from_original_model_rescues_native_predictions(inputs):
    original = CausalTransformer().eval()
    ablated = scale_attention_head(original, 1, 0, 0.0)
    restored = scale_attention_head(original, 1, 0, 1.0)
    with torch.inference_mode():
        original_actions = original(*inputs)
        assert not torch.allclose(ablated(*inputs), original_actions)
        torch.testing.assert_close(restored(*inputs), original_actions, atol=0, rtol=0)


@pytest.mark.parametrize("layer,head,scale", [
    (-1, 0, 1.0), (2, 0, 1.0), (True, 0, 1.0),
    (0, -1, 1.0), (0, 4, 1.0), (0, 1.5, 1.0),
    (0, 0, -0.1), (0, 0, float("inf")), (0, 0, float("nan")), (0, 0, True),
])
def test_invalid_intervention_is_rejected(layer, head, scale):
    with pytest.raises(ValueError):
        scale_attention_head(CausalTransformer(), layer, head, scale)


def test_non_transformer_model_is_rejected():
    with pytest.raises(ValueError, match="CausalTransformer"):
        enumerate_attention_heads(HistoryMLP())
