from __future__ import annotations

from typing import Any, Iterable

from .optional_deps import require_dependency


def _dropout(torch, value: float | None):
    if value is None or float(value) <= 0.0:
        return None
    return torch.nn.Dropout(float(value))


def build_mlp(
    observation_size: int,
    action_size: int,
    hidden_sizes: Iterable[int],
    *,
    dropout: float | None = None,
):
    torch = require_dependency("torch", "ml")
    layers = []
    current_size = int(observation_size)
    for hidden_size in hidden_sizes:
        layers.append(torch.nn.Linear(current_size, int(hidden_size)))
        layers.append(torch.nn.ReLU())
        dropout_layer = _dropout(torch, dropout)
        if dropout_layer is not None:
            layers.append(dropout_layer)
        current_size = int(hidden_size)
    layers.append(torch.nn.Linear(current_size, int(action_size)))
    return torch.nn.Sequential(*layers)


def _build_dense_stack(torch, input_size: int, hidden_sizes: Iterable[int], *, dropout: float | None = None):
    layers = []
    current_size = int(input_size)
    for hidden_size in hidden_sizes:
        layers.append(torch.nn.Linear(current_size, int(hidden_size)))
        layers.append(torch.nn.ReLU())
        dropout_layer = _dropout(torch, dropout)
        if dropout_layer is not None:
            layers.append(dropout_layer)
        current_size = int(hidden_size)
    return torch.nn.Sequential(*layers), current_size


def build_residual_mlp(
    observation_size: int,
    action_size: int,
    *,
    width: int,
    blocks: int,
    dropout: float | None = None,
    layer_norm: bool = False,
):
    torch = require_dependency("torch", "ml")
    residual_block = _residual_block_class(torch)
    layers = [torch.nn.Linear(int(observation_size), int(width)), torch.nn.ReLU()]
    for _ in range(int(blocks)):
        layers.append(residual_block(int(width), dropout=dropout, layer_norm=layer_norm))
    layers.append(torch.nn.Linear(int(width), int(action_size)))
    return torch.nn.Sequential(*layers)


def _observation_layout(observation_size: int, param_count: int, embedding_size: int | None) -> tuple[int, int]:
    if embedding_size is None:
        remaining = int(observation_size) - int(param_count)
        assert remaining > 0 and remaining % 3 == 0, (
            f"Cannot infer embedding size from observation_size={observation_size}, param_count={param_count}."
        )
        embedding_size = remaining // 3
    assert 3 * int(embedding_size) + int(param_count) == int(observation_size), (
        "Observation layout must be [target_embedding, current_embedding, delta_embedding, params]."
    )
    return int(embedding_size), int(param_count)


def build_cnn1d(
    observation_size: int,
    action_size: int,
    *,
    param_count: int,
    embedding_size: int | None = None,
    channels: Iterable[int],
    kernel_sizes: Iterable[int],
    embedding_hidden_size: int,
    param_hidden_sizes: Iterable[int],
    head_hidden_sizes: Iterable[int],
    dropout: float | None = None,
):
    torch = require_dependency("torch", "ml")
    embedding_size, param_count = _observation_layout(observation_size, param_count, embedding_size)
    cnn_class = _cnn_action_value_network_class(torch)
    return cnn_class(
        observation_size=int(observation_size),
        action_size=int(action_size),
        embedding_size=embedding_size,
        param_count=param_count,
        channels=[int(value) for value in channels],
        kernel_sizes=[int(value) for value in kernel_sizes],
        embedding_hidden_size=int(embedding_hidden_size),
        param_hidden_sizes=[int(value) for value in param_hidden_sizes],
        head_hidden_sizes=[int(value) for value in head_hidden_sizes],
        dropout=dropout,
    )


def build_recurrent(
    observation_size: int,
    action_size: int,
    *,
    param_count: int,
    embedding_size: int | None = None,
    hidden_size: int,
    layers: int = 1,
    cell: str = "gru",
    bidirectional: bool = False,
    param_hidden_sizes: Iterable[int] = (),
    head_hidden_sizes: Iterable[int] = (),
    dropout: float | None = None,
):
    torch = require_dependency("torch", "ml")
    embedding_size, param_count = _observation_layout(observation_size, param_count, embedding_size)
    recurrent_class = _recurrent_action_value_network_class(torch)
    return recurrent_class(
        observation_size=int(observation_size),
        action_size=int(action_size),
        embedding_size=embedding_size,
        param_count=param_count,
        hidden_size=int(hidden_size),
        layers=int(layers),
        cell=str(cell),
        bidirectional=bool(bidirectional),
        param_hidden_sizes=[int(value) for value in param_hidden_sizes],
        head_hidden_sizes=[int(value) for value in head_hidden_sizes],
        dropout=dropout,
    )


def build_network(
    spec: dict[str, Any],
    *,
    observation_size: int,
    action_size: int,
    param_count: int | None = None,
    embedding_size: int | None = None,
):
    network_type = str(spec.get("type", "mlp"))
    if network_type == "mlp":
        return build_mlp(
            observation_size,
            action_size,
            spec["hidden_sizes"],
            dropout=spec.get("dropout"),
        )
    if network_type == "residual_mlp":
        return build_residual_mlp(
            observation_size,
            action_size,
            width=int(spec["width"]),
            blocks=int(spec["blocks"]),
            dropout=spec.get("dropout"),
            layer_norm=bool(spec.get("layer_norm", False)),
        )
    if network_type in {"cnn1d", "hybrid_cnn_mlp"}:
        assert param_count is not None, f"{network_type} requires param_count."
        head_key = "head_hidden_sizes" if network_type == "cnn1d" else "fusion_hidden_sizes"
        return build_cnn1d(
            observation_size,
            action_size,
            param_count=int(param_count),
            embedding_size=embedding_size,
            channels=spec["channels"],
            kernel_sizes=spec["kernel_sizes"],
            embedding_hidden_size=int(spec.get("embedding_hidden_size", spec.get("fusion_embedding_size", 128))),
            param_hidden_sizes=spec["param_hidden_sizes"],
            head_hidden_sizes=spec[head_key],
            dropout=spec.get("dropout"),
        )
    if network_type in {"rnn", "gru", "lstm"}:
        assert param_count is not None, f"{network_type} requires param_count."
        return build_recurrent(
            observation_size,
            action_size,
            param_count=int(param_count),
            embedding_size=embedding_size,
            hidden_size=int(spec["hidden_size"]),
            layers=int(spec.get("layers", 1)),
            cell=str(spec.get("cell", network_type)),
            bidirectional=bool(spec.get("bidirectional", False)),
            param_hidden_sizes=spec.get("param_hidden_sizes", []),
            head_hidden_sizes=spec.get("head_hidden_sizes", []),
            dropout=spec.get("dropout"),
        )
    raise ValueError(f"Unsupported architecture type: {network_type}")


def _residual_block_class(torch):
    class _ResidualBlock(torch.nn.Module):
        def __init__(self, width: int, *, dropout: float | None = None, layer_norm: bool = False):
            super().__init__()
            layers = [torch.nn.Linear(width, width), torch.nn.ReLU()]
            dropout_layer = _dropout(torch, dropout)
            if dropout_layer is not None:
                layers.append(dropout_layer)
            layers.append(torch.nn.Linear(width, width))
            self.net = torch.nn.Sequential(*layers)
            self.norm = torch.nn.LayerNorm(width) if layer_norm else torch.nn.Identity()
            self.activation = torch.nn.ReLU()

        def forward(self, x):
            return self.activation(self.norm(x + self.net(x)))

    return _ResidualBlock


def _cnn_action_value_network_class(torch):
    class _CnnActionValueNetwork(torch.nn.Module):
        def __init__(
            self,
            *,
            observation_size: int,
            action_size: int,
            embedding_size: int,
            param_count: int,
            channels: list[int],
            kernel_sizes: list[int],
            embedding_hidden_size: int,
            param_hidden_sizes: list[int],
            head_hidden_sizes: list[int],
            dropout: float | None,
        ):
            super().__init__()
            assert channels, "CNN architecture requires at least one channel size."
            assert len(channels) == len(kernel_sizes), "channels and kernel_sizes must have the same length."
            self.observation_size = observation_size
            self.embedding_size = embedding_size
            self.param_count = param_count
            conv_layers = []
            in_channels = 3
            for out_channels, kernel_size in zip(channels, kernel_sizes):
                padding = int(kernel_size) // 2
                conv_layers.append(torch.nn.Conv1d(in_channels, out_channels, kernel_size=int(kernel_size), padding=padding))
                conv_layers.append(torch.nn.ReLU())
                dropout_layer = _dropout(torch, dropout)
                if dropout_layer is not None:
                    conv_layers.append(dropout_layer)
                in_channels = out_channels
            conv_layers.append(torch.nn.AdaptiveAvgPool1d(1))
            conv_layers.append(torch.nn.Flatten())
            conv_layers.append(torch.nn.Linear(in_channels, embedding_hidden_size))
            conv_layers.append(torch.nn.ReLU())
            self.embedding_branch = torch.nn.Sequential(*conv_layers)
            self.param_branch, param_out = _build_dense_stack(torch, param_count, param_hidden_sizes, dropout=dropout)
            if not param_hidden_sizes:
                self.param_branch = torch.nn.Identity()
                param_out = param_count
            self.head, head_out = _build_dense_stack(torch, embedding_hidden_size + param_out, head_hidden_sizes, dropout=dropout)
            self.output = torch.nn.Linear(head_out, action_size)

        def forward(self, observation):
            target = observation[:, : self.embedding_size]
            current = observation[:, self.embedding_size : 2 * self.embedding_size]
            delta = observation[:, 2 * self.embedding_size : 3 * self.embedding_size]
            params = observation[:, 3 * self.embedding_size :]
            embedding_input = torch.stack([target, current, delta], dim=1)
            embedding_features = self.embedding_branch(embedding_input)
            param_features = self.param_branch(params)
            features = torch.cat([embedding_features, param_features], dim=1)
            return self.output(self.head(features))

    return _CnnActionValueNetwork


def _recurrent_action_value_network_class(torch):
    class _RecurrentActionValueNetwork(torch.nn.Module):
        def __init__(
            self,
            *,
            observation_size: int,
            action_size: int,
            embedding_size: int,
            param_count: int,
            hidden_size: int,
            layers: int,
            cell: str,
            bidirectional: bool,
            param_hidden_sizes: list[int],
            head_hidden_sizes: list[int],
            dropout: float | None,
        ):
            super().__init__()
            assert layers >= 1, f"Recurrent layers must be >= 1, got {layers}."
            self.observation_size = observation_size
            self.embedding_size = embedding_size
            self.param_count = param_count
            recurrent_dropout = float(dropout or 0.0) if int(layers) > 1 else 0.0
            cell_name = str(cell).lower()
            if cell_name == "rnn":
                recurrent_type = torch.nn.RNN
            elif cell_name == "gru":
                recurrent_type = torch.nn.GRU
            elif cell_name == "lstm":
                recurrent_type = torch.nn.LSTM
            else:
                raise ValueError(f"Unsupported recurrent cell: {cell}")
            self.recurrent = recurrent_type(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
            )
            recurrent_out = hidden_size * (2 if bidirectional else 1)
            self.param_branch, param_out = _build_dense_stack(torch, param_count, param_hidden_sizes, dropout=dropout)
            if not param_hidden_sizes:
                self.param_branch = torch.nn.Identity()
                param_out = param_count
            self.head, head_out = _build_dense_stack(torch, recurrent_out + param_out, head_hidden_sizes, dropout=dropout)
            self.output = torch.nn.Linear(head_out, action_size)

        def forward(self, observation):
            target = observation[:, : self.embedding_size]
            current = observation[:, self.embedding_size : 2 * self.embedding_size]
            delta = observation[:, 2 * self.embedding_size : 3 * self.embedding_size]
            params = observation[:, 3 * self.embedding_size :]
            sequence = torch.stack([target, current, delta], dim=1)
            recurrent_output, _hidden = self.recurrent(sequence)
            embedding_features = recurrent_output[:, -1, :]
            param_features = self.param_branch(params)
            features = torch.cat([embedding_features, param_features], dim=1)
            return self.output(self.head(features))

    return _RecurrentActionValueNetwork
