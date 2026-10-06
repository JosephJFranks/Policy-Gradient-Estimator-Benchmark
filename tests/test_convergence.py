import unittest

import numpy as np

from gi.estimators import reward_to_go_gradient
from gi.exact import exact_objective_and_gradient
from gi.mdp import Maze, N_ACTIONS
from gi.sampling import sample_batch


class ConvergenceTests(unittest.TestCase):
    def test_reward_to_go_approaches_exact_gradient(self) -> None:
        maze = Maze()
        theta = np.zeros((maze.n_states, N_ACTIONS))
        _, exact_gradient = exact_objective_and_gradient(maze, theta)
        rng = np.random.default_rng(123)
        estimates = []

        for _ in range(500):
            episodes = sample_batch(maze, theta, batch_size=32, rng=rng)
            estimates.append(reward_to_go_gradient(maze, theta, episodes))

        mean_estimate = np.mean(estimates, axis=0)
        error = np.linalg.norm(mean_estimate - exact_gradient)
        cosine = np.sum(mean_estimate * exact_gradient) / (
            np.linalg.norm(mean_estimate) * np.linalg.norm(exact_gradient)
        )

        self.assertLess(error, 0.08)
        self.assertGreater(cosine, 0.99)


if __name__ == "__main__":
    unittest.main()
