from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable

import numpy as np

from .config import DQNConfig
from .optional_deps import require_dependency


class RandomAgent:
    def __init__(self, action_size: int, seed: int = 7):
        self.action_size = action_size
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        _ = observation
        return int(self._rng.integers(0, self.action_size))


@dataclass(slots=True)
class ReplayTransition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    target_id: str


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: Deque[ReplayTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, transition: ReplayTransition) -> None:
        self._data.append(transition)

    def sample(self, batch_size: int, seed: int | None = None) -> list[ReplayTransition]:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self._data), size=batch_size, replace=False)
        return [self._data[int(index)] for index in indices]


class DQNAgent:
    def __init__(self, observation_size: int, action_size: int, config: DQNConfig):
        torch = require_dependency("torch", "ml")
        self._torch = torch
        self.config = config
        self.action_size = action_size
        self.observation_size = observation_size
        self.online_network = self._build_network(observation_size, action_size, config.hidden_sizes)
        self.target_network = self._build_network(observation_size, action_size, config.hidden_sizes)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        self.replay = ReplayBuffer(config.replay_capacity)
        self.total_steps = 0

    def _build_network(self, observation_size: int, action_size: int, hidden_sizes: Iterable[int]):
        torch = self._torch
        layers = []
        current_size = observation_size
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(current_size, hidden_size))
            layers.append(torch.nn.ReLU())
            current_size = hidden_size
        layers.append(torch.nn.Linear(current_size, action_size))
        return torch.nn.Sequential(*layers)

    def epsilon(self) -> float:
        progress = min(1.0, self.total_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, observation: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon():
            return int(np.random.randint(0, self.action_size))
        torch = self._torch
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            q_values = self.online_network(obs)
        return int(torch.argmax(q_values, dim=1).item())

    def observe(self, transition: ReplayTransition) -> None:
        self.replay.add(transition)
        self.total_steps += 1

    def train_step(self) -> float | None:
        torch = self._torch
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return None
        batch = self.replay.sample(self.config.batch_size)
        obs = torch.tensor(np.stack([item.observation for item in batch]), dtype=torch.float32)
        actions = torch.tensor([item.action for item in batch], dtype=torch.int64)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32)
        next_obs = torch.tensor(np.stack([item.next_observation for item in batch]), dtype=torch.float32)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32)

        q_values = self.online_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = self.target_network(next_obs).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.config.gamma * target_q
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.config.target_sync_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return float(loss.item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(self.online_network.state_dict(), path)

    def load(self, path: Path) -> None:
        state = self._torch.load(path, map_location="cpu")
        self.online_network.load_state_dict(state)
        self.target_network.load_state_dict(state)
