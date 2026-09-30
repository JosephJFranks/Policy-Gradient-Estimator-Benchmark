import unittest

import numpy as np

from main import Maze, N_ACTIONS, exact_objective_and_gradient, finite_difference_gradient


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
