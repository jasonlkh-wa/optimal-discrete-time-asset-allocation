"""
This file contains the implementation of the Environment class, which is used in the
discrete-time asset allocation problem.

The Environment class is responsible for simulating the environment and providing the
yields of the risky and riskless assets, and the probability of each yield. It also
keeps track of the policy of the agent.
"""

import random
from print_utils import print_box
from agent import Agent
from abstract_types import Allocation, StateActionValueDict, ExpectedReturn
import joblib
import numpy as np
from typing import cast


class Environment:
    """
    The Environment class is responsible for simulating the environment and
    providing the yields of the risky and riskless assets, and the probability of
    each yield. It also keeps track of the policy of the agent.

    Attributes:
        yield_a (float): The first yield of the risky asset.
        yield_b (float): The second yield of the risky asset.
        yield_r (float): The yield of the riskless asset.
        probability_of_yield_a (float): The probability of the first yield of the risky asset.
        probability_of_yield_b (float): The probability of the second yield of the risky asset.
        total_turns (int): The total number of turns in the environment.
        agent (Agent): The agent that is interacting with the environment.
    """

    yield_a: float
    yield_b: float
    yield_r: float
    probability_of_yield_a: float
    probability_of_yield_b: float
    total_turns: int
    agent: Agent

    def __init__(
        self,
        yield_a: float,
        yield_b: float,
        yield_r: float,
        probability_of_yield_1: float,
        epsilon: float,
        alpha: float,
        total_turns: int = 10,
    ):
        """
        Initialize the environment with the given parameters.

        Parameters
        ----------
        yield_a : float
            The first yield of the risky asset.
        yield_b : float
            The second yield of the risky asset.
        yield_r : float
            The yield of the riskless asset.
        probability_of_yield_1 : float
            The probability of the first yield of the risky asset.
        epsilon : float
            The probability of selecting a non-greedy action.
        alpha : float
            The step size of the TD update.
        total_turns : int, optional
            The total number of turns in the environment. Defaults to 10.
        """
        self.yield_a = yield_a
        self.yield_b = yield_b
        self.yield_r = yield_r
        self.probability_of_yield_a = probability_of_yield_1
        self.probability_of_yield_b = 1 - probability_of_yield_1
        self.total_turns = total_turns
        self.agent = Agent(
            epsilon=epsilon,
            alpha=alpha,
        )

    def expected_risky_return(self) -> float:
        """Calculate the expected return of the risky asset."""
        return (
            self.yield_a * self.probability_of_yield_a
            + self.yield_b * self.probability_of_yield_b
        )

    @print_box()
    def show_environment(self) -> None:
        """Print the parameters of the environment in a box."""
        print(
            f"Environment:\n"
            f"yield_a: {self.yield_a}\n"
            f"yield_b: {self.yield_b}\n"
            f"yield_r: {self.yield_r}\n"
            f"probability_of_yield_a: {self.probability_of_yield_a}\n"
            f"probability_of_yield_b: {self.probability_of_yield_b}\n"
            f"total_turns: {self.total_turns}\n"
        )

    def get_random_return_of_risky_asset(self) -> float:
        """Return a random return from (yield_a, yield_b) with the given probabilities."""
        return random.choices(
            [self.yield_a, self.yield_b],
            weights=[self.probability_of_yield_a, self.probability_of_yield_b],
        )[0]

    def calculate_new_wealth_with_random_risky_return(
        self,
        current_wealth: float,
        risky_asset_allocation: Allocation,
    ) -> float:
        """
        Calculate the new wealth after the current turn given the current wealth and the allocation of the risky asset.

        The new wealth is calculated as the sum of the returns of the risky asset and the riskless asset.

        Parameters
        ----------
        current_wealth : float
            The current wealth.
        risky_asset_allocation : float
            The allocation of the risky asset.

        Returns
        -------
        float
            The new wealth after the current turn.
        """
        risky_wealth = current_wealth * risky_asset_allocation
        riskless_wealth = current_wealth * (1 - risky_asset_allocation)
        return risky_wealth * (
            1 + self.get_random_return_of_risky_asset()
        ) + riskless_wealth * (1 + self.yield_r)

    def calculate_optimal_strategy_and_expected_return(
        self,
    ) -> tuple[Allocation, ExpectedReturn]:
        """
        Calculate the optimal strategy and expected return of the environment.

        The optimal strategy is the allocation of the risky asset that maximizes the expected return.
        The expected return is the expected value of the wealth after the current turn.

        Returns
        -------
        tuple[float, float]
            A tuple containing the optimal allocation and the expected return.
        """
        max_value = 0
        max_value_action: float = self.agent.possible_actions[0]
        for allocation in self.agent.possible_actions:
            expected_return = (
                self.yield_a * self.probability_of_yield_a
                + self.yield_b * self.probability_of_yield_b
            ) * allocation + self.yield_r * (1 - allocation)
            if expected_return > max_value:
                max_value = expected_return
                max_value_action = allocation

        return (max_value_action, (1 + max_value) ** self.total_turns)

    def export_environment(self, filename: str) -> None:
        joblib.dump(self, filename)

    def calculate_max_q_function_percent_diff(
        self, state_action_value_dict: StateActionValueDict
    ) -> float:
        """Calculates the average percent difference between the current Q function and the optimal Q function.

        Args:
            state_action_value_dict: A dictionary mapping states to their action value dictionaries.

        Returns:
            The average percent difference between the current Q function and the optimal Q function.
        """
        optimal_allocation, _ = self.calculate_optimal_strategy_and_expected_return()
        expected_optimal_return = optimal_allocation * (
            self.expected_risky_return()
        ) + (1 - optimal_allocation) * (
            self.yield_r
        )  # under optimal allocation, expected return of a single time step

        percent_diff_arr = np.array([])  # [optimal q* - q] / optimal q*

        for (turn, wealth), action_value_dict in state_action_value_dict.items():
            for allocation, expected_wealth in action_value_dict.items():
                optimal_q = (
                    wealth
                    * (
                        allocation * (1 + self.expected_risky_return())
                        + (1 - allocation) * (1 + self.yield_r)
                    )
                    * (1 + expected_optimal_return) ** (self.total_turns - turn - 1)
                )  # under optimal policy, expected q-value of the current state

                percent_diff_arr = np.append(
                    percent_diff_arr,
                    [abs(optimal_q - expected_wealth) / optimal_q],
                )

        if len(percent_diff_arr) == 0:
            return 1  # if state_action_value_dict is empty, return 1 as a dummy value

        return cast(float, percent_diff_arr.max())

    def is_optimal_strategy(
        self,
    ) -> bool:
        """
        Check if the current policy is the optimal strategy.

        The optimal strategy is the allocation of the risky asset that maximizes the expected return.
        This function checks if the current policy is the optimal strategy by checking if the
        greedy action of each state is the same as the optimal allocation.

        Note that this function only checks the potential states if optimal action is taken, it does not
        check states which are only reachable through suboptimal actions.

        Returns:
            bool: True if the current policy is the optimal strategy, False otherwise.
        """
        optimal_strategy = self.calculate_optimal_strategy_and_expected_return()[0]

        current_turn = 0
        possible_state: set[float] = {1}  # initial wealth
        next_turn_possible_state: set[float] = set()

        while current_turn < self.total_turns:
            state = possible_state.pop()

            if (
                self.agent.get_greedy_allocation(current_turn, state, mode="Prod")
                == optimal_strategy
            ):
                next_turn_possible_state.add(state * (1 + self.yield_a))
                next_turn_possible_state.add(state * (1 + self.yield_b))
            else:
                return False

            if len(possible_state) == 0:
                possible_state = next_turn_possible_state
                next_turn_possible_state = set()
                current_turn += 1

        return True
