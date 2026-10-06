"""Command-line smoke check for the exact-gradient implementation."""

from __future__ import annotations

import numpy as np

from gi.exact import exact_objective_and_gradient
from gi.mdp import Maze, N_ACTIONS


def main() -> None:
	maze = Maze()
	theta = np.zeros((maze.n_states, N_ACTIONS))
	objective, gradient = exact_objective_and_gradient(maze, theta)
	print(f"uniform-policy objective: {objective:.6f}")
	print(f"gradient norm: {np.linalg.norm(gradient):.6f}")


if __name__ == "__main__":
	main()
