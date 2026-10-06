import unittest

import numpy as np

from gi.exact import exact_objective_and_gradient
from gi.mdp import Maze, N_ACTIONS


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


class ExactGradientTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self) -> None:
        maze = Maze()
        rng = np.random.default_rng(19)
        theta = rng.normal(scale=0.35, size=(maze.n_states, N_ACTIONS))

        _, analytic = exact_objective_and_gradient(maze, theta)
        numerical = finite_difference_gradient(maze, theta)

        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
