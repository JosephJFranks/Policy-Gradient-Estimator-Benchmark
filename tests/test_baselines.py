import unittest

import numpy as np

from gi.estimators import (
    grpo_gradient,
    group_mean_gradient,
    noisy_critic_gradient,
    oracle_critic_gradient,
    rloo_gradient,
)
from gi.exact import exact_state_values
from gi.mdp import Maze, N_ACTIONS
from gi.sampling import sample_batch


class BaselineEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.maze = Maze()
        self.theta = np.zeros((self.maze.n_states, N_ACTIONS))
        self.episodes = sample_batch(
            self.maze,
            self.theta,
            batch_size=4,
            rng=np.random.default_rng(31),
        )
        self.state_values = exact_state_values(self.maze, self.theta)

    def test_critic_variants_return_finite_gradients(self) -> None:
        oracle = oracle_critic_gradient(
            self.maze, self.theta, self.episodes, self.state_values
        )
        noisy = noisy_critic_gradient(
            self.maze,
            self.theta,
            self.episodes,
            self.state_values + 0.25,
        )

        self.assertEqual(oracle.shape, self.theta.shape)
        self.assertEqual(noisy.shape, self.theta.shape)
        self.assertTrue(np.all(np.isfinite(oracle)))
        self.assertTrue(np.all(np.isfinite(noisy)))

    def test_group_variants_return_finite_gradients(self) -> None:
        estimates = [
            group_mean_gradient(self.maze, self.theta, self.episodes),
            rloo_gradient(self.maze, self.theta, self.episodes),
            grpo_gradient(self.maze, self.theta, self.episodes),
        ]

        for estimate in estimates:
            self.assertEqual(estimate.shape, self.theta.shape)
            self.assertTrue(np.all(np.isfinite(estimate)))

    def test_rloo_requires_multiple_episodes(self) -> None:
        with self.assertRaises(ValueError):
            rloo_gradient(self.maze, self.theta, self.episodes[:1])


if __name__ == "__main__":
    unittest.main()
