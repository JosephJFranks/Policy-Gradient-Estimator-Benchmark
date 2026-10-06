import unittest

import numpy as np

from gi.estimators import oracle_critic_gradient, reinforce_gradient, reward_to_go_gradient
from gi.exact import exact_state_values
from gi.mdp import Maze, N_ACTIONS
from gi.sampling import sample_batch


class EstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.maze = Maze()
        self.theta = np.zeros((self.maze.n_states, N_ACTIONS))
        self.episodes = sample_batch(
            self.maze,
            self.theta,
            batch_size=4,
            rng=np.random.default_rng(23),
        )

    def test_estimators_return_gradient_shaped_arrays(self) -> None:
        state_values = exact_state_values(self.maze, self.theta)

        estimates = [
            reinforce_gradient(self.maze, self.theta, self.episodes),
            reward_to_go_gradient(self.maze, self.theta, self.episodes),
            oracle_critic_gradient(
                self.maze, self.theta, self.episodes, state_values
            ),
        ]

        for estimate in estimates:
            self.assertEqual(estimate.shape, self.theta.shape)
            self.assertTrue(np.all(np.isfinite(estimate)))

    def test_estimators_reject_empty_batches(self) -> None:
        with self.assertRaises(ValueError):
            reward_to_go_gradient(self.maze, self.theta, [])


if __name__ == "__main__":
    unittest.main()