"""
This file contains the implementation of the Agent class, which is used in the
discrete-time asset allocation environment.

The Agent class is responsible for making decisions about how to allocate assets
based on the current state of the environment. It uses a policy to map the state of
the environment to an allocation of assets. The policy is an epsilon-greedy policy
that selects the greedy allocation with probability 1 - epsilon and selects the
other allocation with probability epsilon. The agent also has an alpha parameter
that determines the step size of the TD update.

The Agent class also contains a method to get the allocation of assets based on
the state of the environment, and a method to update the state-action value
dictionary using the TD update.
"""

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
    """
    The Agent class is used to make decisions in the discrete-time asset allocation
    environment. The agent uses an epsilon-greedy policy to select the allocation of
    assets, and uses the TD update to update the state-action value dictionary.

    Attributes:
        possible_actions: OneDimensionArray
            The possible actions of the agent (i.e. the possible allocations of assets).
        state_action_value_dict: StateActionValueDict
            The state-action value dictionary used by the agent to make decisions.
        epsilon: float
            The probability of selecting a non-greedy action.
        alpha: float
            The step size of the TD update.
    """

    possible_actions: OneDimensionArray = np.linspace(0, 1, 2)
    state_action_value_dict: StateActionValueDict
    epsilon: float
    alpha: float

    def __init__(
        self,
        epsilon: float,
        alpha,
    ):
        """
        Initialize the agent with an epsilon-greedy policy.

        Parameters
        ----------
        epsilon : float
            The probability of selecting a non-greedy action.
        alpha : float
            The step size of the TD update.
        """
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
        """
        Get the greedy allocation based on the state-action value dictionary.
        If there are two greedy allocations, select one at random.

        Parameters
        ----------
        turn_number : TurnNumber
            The current turn number.
        wealth : Wealth
            The current wealth.
        mode : Mode
            The mode of the agent, either "Prod" or "Test".

        Returns
        -------
        Allocation
            The greedy allocation based on the current state-action value dictionary.
        """
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
        """
        Get the allocation of assets based on the current state and the epsilon-greedy
        policy (i.e. select the greedy action with probability 1 - epsilon, otherwise random).


        Parameters
        ----------
        turn_number : TurnNumber
            The current turn number.
        wealth : Wealth
            The current wealth.
        all_greedy : bool, optional
            Whether to always select the greedy action, by default False.

        Returns
        -------
        tuple[Allocation, bool]
            A tuple containing the allocation and a boolean indicating whether the
            allocation was selected greedily.
        """
        if not all_greedy and random.random() < self.epsilon:
            return random.choice(self.possible_actions), False
        else:
            return self.get_greedy_allocation(
                turn_number=turn_number, wealth=wealth, mode="Prod"
            ), True

    def debug_state_action_value_dict(self) -> None:
        """Prints the state-action value dictionary to the logger with level DEBUG."""
        for (turn, wealth), action_value_dict in self.state_action_value_dict.items():
            logger.debug(f"Turn: {turn} Wealth: {wealth}")

            for action, value in action_value_dict.items():
                logger.debug(f"Action: {action}, Value: {value}")
            logger.debug("\n")
