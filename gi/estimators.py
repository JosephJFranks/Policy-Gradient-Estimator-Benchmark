"""Monte Carlo policy-gradient estimators for sampled episodes."""

from __future__ import annotations
from gi.exact import softmax
from gi.mdp import Maze
from gi.sampling import Episode, discounted_returns
import numpy as np

def _estimate_from_scores(
    maze: Maze, theta: np.ndarray, episodes: list[Episode], score_builder,
) -> np.ndarray:
    '''Helper Function'''
    if not episodes:
        raise ValueError("episodes must not be empty")

    policy = softmax(theta)
    gradient = np.zeros_like(theta, dtype=float)
    for episode in episodes:
        scores = score_builder(episode)
        if len(scores) != episode.length:
            raise ValueError("score builder must return one score per timestep")
        for timestep, (state, action) in enumerate(zip(episode.states, episode.actions)):
            score_function = -policy[state].copy()
            score_function[action] += 1.0
            gradient[state] += (
                maze.discount**timestep * scores[timestep] * score_function
            )
    return gradient / len(episodes)

'''The Different Methods'''

def reinforce_gradient(
    maze: Maze, theta: np.ndarray, episodes: list[Episode]
) -> np.ndarray:
    """Estimate the gradient using the discounted whole-episode return."""

    def whole_return(episode: Episode) -> np.ndarray:
        value = discounted_returns(episode.rewards, maze.discount)[0]
        return np.full(episode.length, value)

    return _estimate_from_scores(
        maze, 
        theta, 
        episodes, 
        whole_return
    )

def reward_to_go_gradient(
    maze: Maze, theta: np.ndarray, episodes: list[Episode]
) -> np.ndarray:
    """Estimate the gradient using discounted reward-to-go."""

    return _estimate_from_scores(
        maze,
        theta,
        episodes,
        lambda episode: discounted_returns(episode.rewards, maze.discount),
    )

def oracle_critic_gradient(
    maze: Maze, theta: np.ndarray, episodes: list[Episode], state_values: np.ndarray,
) -> np.ndarray:
    """Estimate the gradient using the exact state-value function as a baseline."""
    if state_values.shape != (maze.n_states,):
        raise ValueError("state_values must contain one value per maze state")

    return _estimate_from_scores(
        maze,
        theta,
        episodes,
        lambda episode: discounted_returns(episode.rewards, maze.discount) - state_values[episode.states],
    )

def noisy_critic_gradient(
    maze: Maze, theta: np.ndarray, episodes: list[Episode], critic_values: np.ndarray,
) -> np.ndarray:
    """Estimate the gradient using a supplied imperfect state-value critic."""
    if critic_values.shape != (maze.n_states,):
        raise ValueError("critic_values must contain one value per maze state")

    return _estimate_from_scores(
        maze,
        theta,
        episodes,
        lambda episode: discounted_returns(episode.rewards, maze.discount) - critic_values[episode.states],
    )

def _episode_returns(episodes: list[Episode], discount: float) -> np.ndarray:
    if not episodes:
        raise ValueError("episodes must not be empty")
    return np.asarray(
        [discounted_returns(episode.rewards, discount)[0] for episode in episodes]
    )

def group_mean_gradient(
    maze: Maze, theta: np.ndarray, episodes: list[Episode]
) -> np.ndarray:
    """Estimate the gradient with a batch return baseline including each sample."""
    returns = _episode_returns(episodes, maze.discount)
    baseline = float(np.mean(returns))
    scores = returns - baseline
    score_by_episode = {id(episode): score for episode, score in zip(episodes, scores)}
    return _estimate_from_scores(
        maze,
        theta,
        episodes,
        lambda episode: np.full(episode.length, score_by_episode[id(episode)]),
    )

def rloo_gradient(
    maze: Maze, theta: np.ndarray, episodes: list[Episode]
) -> np.ndarray:
    """Estimate the gradient with a leave-one-out return baseline."""
    returns = _episode_returns(episodes, maze.discount)
    if len(returns) < 2:
        raise ValueError("RLOO requires at least two episodes")
    scores = (len(returns) * returns - np.sum(returns)) / (len(returns) - 1)
    scores = returns - scores
    score_by_episode = {id(episode): score for episode, score in zip(episodes, scores)}
    return _estimate_from_scores(
        maze,
        theta,
        episodes,
        lambda episode: np.full(episode.length, score_by_episode[id(episode)]),
    )

def grpo_gradient(
    maze: Maze,
    theta: np.ndarray,
    episodes: list[Episode],
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Estimate the gradient with a standardized group-relative return."""
    returns = _episode_returns(episodes, maze.discount)
    scores = (returns - np.mean(returns)) / (np.std(returns) + epsilon)
    score_by_episode = {id(episode): score for episode, score in zip(episodes, scores)}
    return _estimate_from_scores(
        maze,
        theta,
        episodes,
        lambda episode: np.full(episode.length, score_by_episode[id(episode)]),
    )