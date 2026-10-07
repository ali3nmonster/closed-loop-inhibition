"""Frozen attention-head interventions and diagnostic residual contributions.

Scaling columns of the attention output projection multiplies one head's
projected residual contribution at every token. It leaves the shared output
bias and the native attention, masking, normalization, and downstream operations
intact. This identifies a pathway intervention, not an excitatory/inhibitory
identity: functional suppression must be established from causal effects.
"""

from copy import deepcopy
import math
from numbers import Integral, Real

import torch
from torch import Tensor
from torch.nn import functional as F

from .neural import CausalTransformer, _validate_inputs


def _check_model(model: CausalTransformer) -> None:
    if not isinstance(model, CausalTransformer):
        raise ValueError("model must be a CausalTransformer")


def _index(value: int, size: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < size:
        raise ValueError(f"{name} must be an integer in [0, {size})")
    return int(value)


def enumerate_attention_heads(model: CausalTransformer) -> list[tuple[int, int]]:
    """Return stable, zero-based (layer, head) identifiers in layer order."""
    _check_model(model)
    return [
        (layer_index, head)
        for layer_index, layer in enumerate(model.layers)
        for head in range(layer.self_attn.num_heads)
    ]


def scale_attention_head(
    model: CausalTransformer, layer: int, head: int, scale: float
) -> CausalTransformer:
    """Return an independent, frozen eval clone with one head contribution scaled.

    ``scale=1`` preserves the complete state dict and native predictions exactly.
    Shared attention output bias is never scaled. The original model's weights,
    mode, and gradient flags are unchanged. To restore a pathway, construct the
    ``scale=1`` clone from the original checkpoint model, not an ablated clone.
    """
    _check_model(model)
    layer = _index(layer, len(model.layers), "layer")
    attention = model.layers[layer].self_attn
    head = _index(head, attention.num_heads, "head")
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    clone = deepcopy(model)
    clone.eval()
    clone.requires_grad_(False)
    if scale != 1:
        head_width = attention.head_dim
        columns = slice(head * head_width, (head + 1) * head_width)
        weight = clone.layers[layer].self_attn.out_proj.weight
        with torch.no_grad():
            weight[:, columns].mul_(float(scale))
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("scale produces nonfinite attention projection weights")
    return clone


@torch.no_grad()
def head_residuals(model: CausalTransformer, tokens: Tensor, valid: Tensor) -> Tensor:
    """Return detached per-head projected contributions [B, T, L, H, width].

    Shared attention output biases are excluded; padding is zero. Contributions
    sum to each layer's attention branch output minus that bias, before residual
    addition and subsequent MLP/normalization. Native encoder-layer forwarding
    advances the upstream states. This diagnostic uses the checkpoint's weights
    without installing hooks or changing model state, and assumes the uniform
    head count and dropout-free pre-norm layers of ``CausalTransformer``.

    This tensor is for norm matching and inspection, not additive attribution of
    the final nonlinear action. Use interventions with native downstream
    computation for causal action effects.
    """
    _check_model(model)
    _validate_inputs(tokens, valid, model.input_dim, model.max_tokens)
    head_counts = {layer.self_attn.num_heads for layer in model.layers}
    if len(head_counts) != 1:
        raise ValueError("head_residuals requires the same head count in every layer")
    clean = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    hidden = model.input_projection(clean) + model.position_embedding(positions)
    future = torch.ones(
        tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device
    ).triu(diagonal=1)
    # SDPA boolean masks use True for allowed entries, unlike encoder masks.
    allowed = (~future)[None, None] & valid[:, None, None, :]
    contributions = []
    batch_size, time_steps, width = hidden.shape
    for layer in model.layers:
        attention = layer.self_attn
        normalized = layer.norm1(hidden)
        query, key, value = F.linear(
            normalized, attention.in_proj_weight, attention.in_proj_bias
        ).chunk(3, dim=-1)
        heads = attention.num_heads
        head_width = attention.head_dim

        def split_heads(projected: Tensor) -> Tensor:
            return projected.reshape(batch_size, time_steps, heads, head_width).transpose(1, 2)

        mixed_values = F.scaled_dot_product_attention(
            split_heads(query), split_heads(key), split_heads(value),
            attn_mask=allowed, dropout_p=0.0,
        )
        projected = torch.einsum(
            "bhtd,whd->bthw", mixed_values,
            attention.out_proj.weight.reshape(width, heads, head_width),
        )
        contributions.append(projected)
        hidden = layer(hidden, src_mask=future, src_key_padding_mask=~valid)
    return torch.stack(contributions, dim=2).masked_fill(
        ~valid[:, :, None, None, None], 0.0
    )
