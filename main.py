"""Exact policy-gradient calculations for the slippery maze prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


N_ACTIONS = 4
UP, DOWN, LEFT, RIGHT = range(N_ACTIONS)


@dataclass(frozen=True)
class Maze:
	rows: int = 3
	cols: int = 4
	slip_probability: float = 0.2
	discount: float = 0.9
	start_state: int = 8

	@property
	def n_states(self) -> int:
		return self.rows * self.cols

	@property
	def treasure_state(self) -> int:
		return 3

	@property
	def pit_states(self) -> frozenset[int]:
		return frozenset({7, 11})

	@property
	def terminal_states(self) -> frozenset[int]:
		return self.pit_states | {self.treasure_state}

	def state(self, row: int, col: int) -> int:
		return row * self.cols + col

	def coordinate(self, state: int) -> tuple[int, int]:
		return divmod(state, self.cols)

	def transition_model(self) -> tuple[np.ndarray, np.ndarray]:
		transitions = np.zeros((self.n_states, N_ACTIONS, self.n_states))
		rewards = np.zeros((self.n_states, N_ACTIONS, self.n_states))

		for state in range(self.n_states):
			if state in self.terminal_states:
				transitions[state, :, state] = 1.0
				continue

			row, col = self.coordinate(state)
			for action in range(N_ACTIONS):
				moved_row, moved_col = row, col
				if action == UP:
					moved_row -= 1
				elif action == DOWN:
					moved_row += 1
				elif action == LEFT:
					moved_col -= 1
				elif action == RIGHT:
					moved_col += 1

				moved_row = int(np.clip(moved_row, 0, self.rows - 1))
				moved_col = int(np.clip(moved_col, 0, self.cols - 1))
				intended_state = self.state(moved_row, moved_col)

				slipped_row = min(moved_row + 1, self.rows - 1)
				slipped_state = self.state(slipped_row, moved_col)
				transitions[state, action, intended_state] += 1.0 - self.slip_probability
				transitions[state, action, slipped_state] += self.slip_probability

				if intended_state == self.treasure_state:
					rewards[state, action, intended_state] = 10.0
				elif intended_state in self.pit_states:
					rewards[state, action, intended_state] = -5.0
				if slipped_state == self.treasure_state:
					rewards[state, action, slipped_state] = 10.0
				elif slipped_state in self.pit_states:
					rewards[state, action, slipped_state] = -5.0

		return transitions, rewards


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


def finite_difference_gradient(maze: Maze, theta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	numerical_gradient = np.zeros_like(theta)
	for index in np.ndindex(theta.shape):
		plus = theta.copy()
		minus = theta.copy()
		plus[index] += epsilon
		minus[index] -= epsilon
		plus_objective, _ = exact_objective_and_gradient(maze, plus)
		minus_objective, _ = exact_objective_and_gradient(maze, minus)
		numerical_gradient[index] = (plus_objective - minus_objective) / (2.0 * epsilon)
	return numerical_gradient


def validate_exact_gradient() -> float:
	maze = Maze()
	rng = np.random.default_rng(7)
	theta = rng.normal(scale=0.4, size=(maze.n_states, N_ACTIONS))
	_, analytic_gradient = exact_objective_and_gradient(maze, theta)
	numerical_gradient = finite_difference_gradient(maze, theta)
	error = float(np.max(np.abs(analytic_gradient - numerical_gradient)))
	if error >= 1e-7:
		raise AssertionError(f"finite-difference error too large: {error:.3e}")
	return error


def main() -> None:
	error = validate_exact_gradient()
	maze = Maze()
	theta = np.zeros((maze.n_states, N_ACTIONS))
	objective, gradient = exact_objective_and_gradient(maze, theta)
	print(f"exact gradient check: max absolute error = {error:.3e}")
	print(f"uniform-policy objective: {objective:.6f}")
	print(f"gradient norm: {np.linalg.norm(gradient):.6f}")


if __name__ == "__main__":
	main()
