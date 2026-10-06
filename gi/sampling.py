"""Trajectory sampling for the tabular slippery-maze policy."""

from __future__ import annotations
from dataclasses import dataclass
from gi.exact import softmax
from gi.mdp import Maze, N_ACTIONS
import numpy as np

@dataclass(frozen=True)
class Episode:
    """Raw trajectory data shared by all gradient estimators."""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    log_probabilities: np.ndarray
    terminated: bool
    truncated: bool

    @property
    def length(self) -> int:
        return len(self.rewards)


def sample_episode(
    maze: Maze, theta: np.ndarray, rng: np.random.Generator, max_steps: int = 200,
) -> Episode:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    transitions, transition_rewards = maze.transition_model()
    policy = softmax(theta)
    state = maze.start_state
    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[int] = []
    log_probabilities: list[float] = []
    terminated = False

    for _ in range(max_steps):
        if state in maze.terminal_states:
            terminated = True
            break

        action = int(rng.choice(N_ACTIONS, p=policy[state]))
        next_state = int(rng.choice(maze.n_states, p=transitions[state, action]))
        reward = float(transition_rewards[state, action, next_state])

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        log_probabilities.append(float(np.log(policy[state, action])))

        state = next_state
        if state in maze.terminal_states:
            terminated = True
            break

    return Episode(
        states=np.asarray(states, dtype=int),
        actions=np.asarray(actions, dtype=int),
        rewards=np.asarray(rewards, dtype=float),
        next_states=np.asarray(next_states, dtype=int),
        log_probabilities=np.asarray(log_probabilities, dtype=float),
        terminated=terminated,
        truncated=not terminated,
    )

def sample_batch(
    maze: Maze, theta: np.ndarray, batch_size: int, rng: np.random.Generator, max_steps: int = 200,
) -> list[Episode]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    return [sample_episode(maze, theta, rng, max_steps) for _ in range(batch_size)]

def discounted_returns(rewards: np.ndarray, discount: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=float)
    running_return = 0.0
    for timestep in range(len(rewards) - 1, -1, -1):
        running_return = rewards[timestep] + discount * running_return
        returns[timestep] = running_return
    return returns
