import numpy as np
from abstract_types import (
    OneDimensionArray,
    StateActionValueDict,
    Allocation,
    TurnNumber,
)
import random
from logger import logger


class Agent:
    possible_actions: OneDimensionArray = np.linspace(0, 1, 2)
    state_action_value_dict: StateActionValueDict
    epsilon: float
    alpha: float
    default_value: float

    def __init__(
        self, total_turns: int, epsilon: float, alpha, default_value: float = 1.0
    ):
        self.state_action_value_dict = {
            turn: {action: default_value for action in self.possible_actions}
            for turn in range(total_turns)
        }
        self.epsilon = epsilon
        self.alpha = alpha
        self.default_value = default_value  # default value of Q(s, a)

    def get_greedy_allocation(self, turn_number: TurnNumber) -> Allocation:
        action_value_dict = self.state_action_value_dict[turn_number]
        # logger.debug(action_value_dict) # CR-soon kleung: consider removing this line
        max_value = max(action_value_dict.values())

        possible_actions = []
        for key, val in action_value_dict.items():
            if val == max_value:
                possible_actions.append(key)

        return random.choice(possible_actions)

    def get_allocation(self, turn_number: TurnNumber) -> tuple[Allocation, bool]:
        if random.random() < self.epsilon:
            return random.choice(self.possible_actions), False
        else:
            return self.get_greedy_allocation(turn_number=turn_number), True

    def debug_state_action_value_dict(self, show_non_default_only: bool):
        for turn, action_value_dict in self.state_action_value_dict.items():
            if show_non_default_only:
                action_value_dict = {
                    action: value
                    for action, value in action_value_dict.items()
                    if value != self.default_value
                }
            logger.debug(f"Turn: {turn}")

            for action, value in action_value_dict.items():
                logger.debug(f"Action: {action}, Value: {value}")
            logger.debug("\n")

    def is_optimal_strategy(self, optimal_strategy=0):
        for _, action_value_dict in self.state_action_value_dict.items():
            if (
                max(action_value_dict.items(), key=lambda x: x[1])[0]
                != optimal_strategy
            ):
                return False

        return True
