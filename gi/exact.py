"""Exact objective and policy-gradient calculations."""

from __future__ import annotations

import numpy as np

from gi.mdp import Maze


def softmax(theta: np.ndarray) -> np.ndarray:
    shifted = theta - np.max(theta, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def exact_objective_and_gradient(
    maze: Maze, theta: np.ndarray, initial_state_distribution: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    transitions, transition_rewards = maze.transition_model()
    policy = softmax(theta)
    policy_transition = np.einsum("sa,san->sn", policy, transitions)
    policy_reward = np.einsum("sa,san,san->s", policy, transitions, transition_rewards)

    system = np.eye(maze.n_states) - maze.discount * policy_transition
    values = np.linalg.solve(system, policy_reward)
    action_values = np.einsum("san,n->sa", transitions, values) * maze.discount
    action_values += np.einsum("san,san->sa", transitions, transition_rewards)

    if initial_state_distribution is None:
        initial_state_distribution = np.zeros(maze.n_states)
        initial_state_distribution[maze.start_state] = 1.0
    discounted_visitation = np.linalg.solve(system.T, initial_state_distribution)
    advantages = action_values - values[:, None]
    gradient = discounted_visitation[:, None] * policy * advantages
    objective = float(initial_state_distribution @ values)
    return objective, gradient
