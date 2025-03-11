import numpy as np
from abstract_types import (
    OneDimensionArray,
    StateActionValueDict,
    Allocation,
    TurnNumber,
    Wealth,
    Mode,
)
import random
from logger import logger


class Agent:
    possible_actions: OneDimensionArray = np.linspace(0, 1, 2)
    state_action_value_dict: StateActionValueDict
    epsilon: float
    alpha: float
    default_value: float

    def __init__(self, epsilon: float, alpha, default_value: float = 1.0):
        self.state_action_value_dict = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.default_value = default_value  # default value of Q(s, a)

    def get_greedy_allocation(
        self,
        turn_number: TurnNumber,
        wealth: Wealth,
        mode: Mode,
    ) -> Allocation:
        # inialize action_value_dict if not present
        if mode == "Prod" and (turn_number, wealth) not in self.state_action_value_dict:
            self.state_action_value_dict[(turn_number, wealth)] = {
                action: self.default_value for action in self.possible_actions
            }

        action_value_dict = self.state_action_value_dict[(turn_number, wealth)]
        # logger.debug(action_value_dict) # CR-soon kleung: consider removing this line
        max_value = max(action_value_dict.values())

        possible_actions = []
        for key, val in action_value_dict.items():
            if val == max_value:
                possible_actions.append(key)

        return random.choice(possible_actions)

    def get_allocation(
        self, turn_number: TurnNumber, wealth: Wealth, all_greedy: bool = False
    ) -> tuple[Allocation, bool]:
        if not all_greedy and random.random() < self.epsilon:
            return random.choice(self.possible_actions), False
        else:
            return self.get_greedy_allocation(
                turn_number=turn_number, wealth=wealth, mode="Prod"
            ), True

    def debug_state_action_value_dict(self, show_non_default_only: bool):
        for (turn, wealth), action_value_dict in self.state_action_value_dict.items():
            if show_non_default_only:
                action_value_dict = {
                    action: value
                    for action, value in action_value_dict.items()
                    if value != self.default_value
                }
            logger.debug(f"Turn: {turn} Wealth: {wealth}")

            for action, value in action_value_dict.items():
                logger.debug(f"Action: {action}, Value: {value}")
            logger.debug("\n")

    def is_optimal_strategy(self, optimal_strategy):
        for _, action_value_dict in self.state_action_value_dict.items():
            if (
                max(action_value_dict.items(), key=lambda x: x[1])[0]
                != optimal_strategy
            ):
                return False

        return True
