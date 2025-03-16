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

    def __init__(
        self,
        epsilon: float,
        alpha,
    ):
        self.state_action_value_dict = {}
        self.epsilon = epsilon
        self.alpha = alpha

    def get_greedy_allocation(
        self,
        turn_number: TurnNumber,
        wealth: Wealth,
        mode: Mode,
    ) -> Allocation:
        # inialize action_value_dict if not present
        if mode == "Prod" and (turn_number, wealth) not in self.state_action_value_dict:
            self.state_action_value_dict[(turn_number, wealth)] = {
                # taking the current wealth as the default q value
                action: wealth
                for action in self.possible_actions
            }

        action_value_dict = self.state_action_value_dict[(turn_number, wealth)]
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

    def debug_state_action_value_dict(self):
        for (turn, wealth), action_value_dict in self.state_action_value_dict.items():
            logger.debug(f"Turn: {turn} Wealth: {wealth}")

            for action, value in action_value_dict.items():
                logger.debug(f"Action: {action}, Value: {value}")
            logger.debug("\n")
