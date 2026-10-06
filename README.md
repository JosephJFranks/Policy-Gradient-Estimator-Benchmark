# Policy Gradient Estimator Benchmark

A study of how baseline choice affects the quality of policy-gradient estimates. It compares REINFORCE-, critic-, and group-based estimators against an analytically computed gradient in a small tabular maze.

The benchmark is designed to answer:

1. How do whole-return REINFORCE, reward-to-go, critic-based advantages, group baselines, and leave-one-out baselines compare?
2. How much value-function error can a critic tolerate before a group baseline becomes preferable?
3. Does including each sample in its own group baseline produce the predicted magnitude shrinkage?
4. How much does omitting the trajectory-time factor `gamma^t` change the discounted gradient?
5. Does a cleaner gradient estimate lead to faster improvement in the exact objective?

This project uses a small, structured slippery maze whose policy objective and exact gradient can be calculated through dynamic programming. The maze is not intended to represent a realistic control benchmark - its purpose is to provide a simple environment in which sampled gradient estimates can be compared with a known reference. That makes it possible to measure bias, variance, directional error, and gradient magnitude directly.

## The idea

For a policy $\pi_\theta$, the policy-gradient theorem expresses the gradient of the expected discounted return as an expectation over sampled trajectories:

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_t \gamma^t
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, S_t
\right].
$$

Here, $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ is the score-function term for the sampled action, and $S_t$ is the scalar assigned to that action. With $N$ sampled episodes, the Monte Carlo estimator is

$$
\widehat{\nabla_\theta J}
= \frac{1}{N}
\sum_{i=1}^{N}
\sum_t \gamma^t
\nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})\, S_{i,t}.
$$

The choice of $S_{i,t}$ defines the estimator. It may be the complete episode return, a reward-to-go $G_{i,t}$, or an advantage such as $G_{i,t} - V(s_{i,t})$. Group-based methods instead construct $S_{i,t}$ from the returns of several episodes, while leave-one-out methods exclude the current episode from that baseline.

REINFORCE, critic-based PPO-style estimators, and GRPO-style group baselines differ primarily in how they construct the score $S_{i,t}$. This project compares those scoring rules directly against the analytic gradient, before asking the separate question of how their updates affect learning over time.

Episodes are sampled as REINFORCE, PPO, or GRPO would be, and each Monte Carlo gradient estimate is compared with this analytic gradient rather than judged through a noisy final reward. This isolates the quality of the local update signal: bias measures systematic deviation from the target, variance measures sensitivity to the sampled batch, cosine similarity measures directional agreement, and gradient magnitude exposes scaling effects such as group-baseline self-inclusion.

## Estimators

- Whole-return REINFORCE
- Discounted reward-to-go
- Oracle state-value critic
- Noisy state-value critic at controlled error levels
- Group-mean baseline
- Leave-one-out baseline
- Optional standard-deviation normalization ablation

Every estimator uses the same sampled episodes within a paired repetition. Differences should therefore come from the scoring rule rather than different random trajectories.

## Measurements and visual outputs

For several batch sizes, the experiments measure:

- Bias relative to the exact gradient
- Monte Carlo variance and mean squared error
- Mean cosine similarity with the exact gradient
- Fraction of estimates pointing in the wrong direction
- Mean estimated gradient magnitude

Planned visuals include:

- A diagram of the maze, policy, and exact gradient
- A parallel/perpendicular gradient scatter plot
- Error versus batch size on log-log axes
- A critic-quality crossover plot against the group baseline
- A group-mean versus leave-one-out magnitude-bias plot
- An estimator summary table
- A regime map showing which estimator wins across critic quality and batch size
- Exact-objective learning curves using the different sampled gradient estimates

## Scope

This project studies the gradient-estimation ideas associated with REINFORCE, critic-based PPO-style advantages, and GRPO-style group baselines. Here, "PPO-style" means a state-value critic and "GRPO-style" means a group-relative return baseline.

It does not benchmark PPO clipping, old-policy ratios, multiple epochs, neural policies, or full training behaviour. Those mechanisms belong in a later project on a larger task. This project asks whether an estimator provides a good learning signal; a later project can ask whether that signal produces better learning in practice.

## Run the current checks

```bash
python3 -m unittest discover -s tests -v
python3 main.py
```

## Current status

The repository currently contains the exact objective, analytic gradient, and finite-difference validation. Episode sampling, estimator comparisons, metrics, and visualizations are the next implementation slices.
