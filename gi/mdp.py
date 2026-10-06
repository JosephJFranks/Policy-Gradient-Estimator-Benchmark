"""The tabular slippery-maze environment."""

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
