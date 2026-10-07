"""Small history-based neural controllers with explicit causal masking.

Both models receive identical tokens and validity information. Tokens are in
chronological order and right padded; every sequence must contain at least one
valid token. Outputs are unbounded scalars in the training target's units. The
policy adapter is responsible for converting normalized outputs into physical
actions; the simulator applies actuator limits.

PyTorch is an optional dependency, imported only when this module is used.
"""

from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_inputs(
    tokens: Tensor, valid: Tensor, input_dim: int, max_tokens: int
) -> None:
    if tokens.ndim != 3 or tokens.shape[-1] != input_dim:
        raise ValueError(f"tokens must have shape [batch, time, {input_dim}]")
    if not tokens.is_floating_point():
        raise ValueError("tokens must be floating point")
    if tokens.shape[0] < 1 or not 1 <= tokens.shape[1] <= max_tokens:
        raise ValueError(f"batch must be nonempty and time must be in [1, {max_tokens}]")
    if valid.dtype != torch.bool or valid.shape != tokens.shape[:2]:
        raise ValueError("valid must be a boolean tensor with shape [batch, time]")
    if valid.device != tokens.device:
        raise ValueError("tokens and valid must be on the same device")
    if not bool(valid[:, 0].all()):
        raise ValueError("every sequence must have at least one valid leading token")
    if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("valid tokens must form a contiguous prefix (right padding)")


class CausalTransformer(nn.Module):
    """Pre-norm, dropout-free causal transformer with a final-token readout.

    Learned positions encode token order; timestamp features supplied by the
    caller encode physical time. Layers are constructed independently so their
    initial weights are not identical copies. No E/I constraint or suppressive
    component is added to this baseline.
    """

    def __init__(
        self,
        input_dim: int = 10,
        max_tokens: int = 11,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_width: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, "input_dim")
        self.max_tokens = _positive_integer(max_tokens, "max_tokens")
        width = _positive_integer(width, "width")
        heads = _positive_integer(heads, "heads")
        layers = _positive_integer(layers, "layers")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        feedforward_width = _positive_integer(
            2 * width if feedforward_width is None else feedforward_width,
            "feedforward_width",
        )
        self.input_projection = nn.Linear(self.input_dim, width)
        self.position_embedding = nn.Embedding(self.max_tokens, width)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=feedforward_width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, 1)

    def forward_tokens(self, tokens: Tensor, valid: Tensor) -> Tensor:
        """Return [batch, time, width] causal features, zero at padded slots."""
        _validate_inputs(tokens, valid, self.input_dim, self.max_tokens)
        # Replace padding before projection: even NaNs in unused slots cannot
        # contaminate keys, values, or the sample's valid outputs.
        clean = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.input_projection(clean) + self.position_embedding(positions)
        future = torch.ones(
            tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device
        ).triu(diagonal=1)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=future, src_key_padding_mask=~valid)
        return self.final_norm(hidden).masked_fill(~valid.unsqueeze(-1), 0.0)

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        hidden = self.forward_tokens(tokens, valid)
        last = valid.sum(dim=1) - 1
        batch = torch.arange(tokens.shape[0], device=tokens.device)
        return self.readout(hidden[batch, last]).squeeze(-1)


class HistoryMLP(nn.Module):
    """MLP comparator with the same full history and explicit validity mask.

    The default 208-unit hidden layers give 69,057 parameters, compared with
    68,545 for the default transformer. This is a close capacity comparison,
    not a claim of matched compute or identical inductive biases.
    """

    def __init__(
        self,
        input_dim: int = 10,
        max_tokens: int = 11,
        hidden_width: int = 208,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, "input_dim")
        self.max_tokens = _positive_integer(max_tokens, "max_tokens")
        hidden_width = _positive_integer(hidden_width, "hidden_width")
        self.network = nn.Sequential(
            nn.Linear(self.max_tokens * (self.input_dim + 1), hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        _validate_inputs(tokens, valid, self.input_dim, self.max_tokens)
        clean = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        remaining = self.max_tokens - tokens.shape[1]
        clean = F.pad(clean, (0, 0, 0, remaining))
        valid_features = F.pad(valid.to(tokens.dtype), (0, remaining))
        features = torch.cat((clean.flatten(start_dim=1), valid_features), dim=1)
        return self.network(features).squeeze(-1)


def make_model(
    kind: str, input_dim: int = 10, max_tokens: int = 11, width: int = 64
) -> nn.Module:
    """Build a baseline; width sets transformer width or scales the MLP size."""
    width = _positive_integer(width, "width")
    if kind == "transformer":
        return CausalTransformer(input_dim=input_dim, max_tokens=max_tokens, width=width)
    if kind == "mlp":
        return HistoryMLP(
            input_dim=input_dim, max_tokens=max_tokens, hidden_width=round(width * 3.25)
        )
    raise ValueError("kind must be 'transformer' or 'mlp'")
