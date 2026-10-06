import unittest

import numpy as np

from gi.mdp import Maze, N_ACTIONS
from gi.sampling import discounted_returns, sample_batch, sample_episode


class SamplingTests(unittest.TestCase):
    def test_episode_records_are_aligned_and_seeded(self) -> None:
        maze = Maze()
        theta = np.zeros((maze.n_states, N_ACTIONS))

        first = sample_episode(maze, theta, np.random.default_rng(42))
        second = sample_episode(maze, theta, np.random.default_rng(42))

        np.testing.assert_array_equal(first.states, second.states)
        np.testing.assert_array_equal(first.actions, second.actions)
        np.testing.assert_array_equal(first.rewards, second.rewards)
        self.assertEqual(first.length, len(first.actions))
        self.assertEqual(first.length, len(first.next_states))
        self.assertEqual(first.length, len(first.log_probabilities))
        self.assertLessEqual(first.length, 200)

    def test_discounted_returns_are_reward_to_go(self) -> None:
        rewards = np.array([1.0, 2.0, 3.0])

        returns = discounted_returns(rewards, discount=0.5)

        np.testing.assert_allclose(returns, [2.75, 3.5, 3.0])

    def test_batch_size_must_be_positive(self) -> None:
        maze = Maze()
        theta = np.zeros((maze.n_states, N_ACTIONS))

        with self.assertRaises(ValueError):
            sample_batch(maze, theta, batch_size=0, rng=np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()