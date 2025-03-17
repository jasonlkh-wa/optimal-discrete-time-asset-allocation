"""
A run means an epoch of the training/simulation with [total_turns] turns of the asset
allocatoin environment.

This file contains the implementation of the SingleRunInternalState class, which is used
internally by the Environment class to simulate a single run of the environment over a
specified number of turns.

The SingleRunInternalState class is responsible for simulating a single run of the
environment over a specified number of turns. It uses the Agent's policy to determine the
allocation of assets for each turn, and updates the current wealth based on the allocation
and the random return from the risky asset. It also stores the state of each turn in a
dictionary for future reference.

The SingleRunInternalState class also contains methods to train the Agent using the TD
update, and to get the allocation of assets based on the Agent's policy.
"""

from typing import Optional
from environment import Environment
from agent import Agent
from abstract_types import (
    TurnNumber,
    Allocation,
    StateActionValueDict,
    Mode,
    ActionValueDict,
)
import copy


class State:
    """
    Class to represent the state of the environment at a single turn.

    The state of the environment at a single turn is represented by the current turn
    number, the current wealth, the selected allocation of assets, and whether the
    allocation was selected greedily.

    Attributes:
        turn: TurnNumber
            The current turn number.
        wealth: float
            The current wealth.
        selected_allocation: Optional[Allocation]
            The allocation of assets selected by the Agent's policy.
        is_greedy_allocation: Optional[bool]
            Whether the allocation was selected greedily.
    """

    turn: TurnNumber
    wealth: float
    selected_allocation: Optional[Allocation]
    is_greedy_allocation: Optional[bool]

    def __init__(self, turn: int, wealth: float):
        """
        Initialize the state with the given turn and wealth.

        Parameters
        ----------
        turn : int
            The current turn number.
        wealth : float
            The current wealth.
        """

        self.turn = turn
        self.wealth = wealth
        self.selected_allocation = None
        self.is_greedy_allocation = None

    def set_allocation_and_is_greedy(
        self, allocation: Allocation, is_greedy_allocation: bool
    ) -> None:
        """
        Set the allocation and greediness status for the current state.

        Parameters
        ----------
        allocation : Allocation
            The allocation of assets selected by the Agent's policy.
        is_greedy_allocation : bool
            Whether the allocation was selected greedily.

        Asserts
        -------
        AssertionError
            If the allocation or is_greedy_allocation is already set.
        """
        assert self.selected_allocation is None, (
            "Allocation already set, this should never be reached"
        )
        assert self.is_greedy_allocation is None, (
            "[is_greedy_allocation] already set, this should never be reached"
        )

        self.selected_allocation = allocation
        self.is_greedy_allocation = is_greedy_allocation


class SingleRunInternalState:
    """
    The SingleRunInternalState class is used to store the state of a single run (epoch) of the
    environment. It contains the state-action value dictionary, the total number of
    turns, and a dictionary of states for each turn.

    Attributes
    ----------
    agent : Agent
        The agent used in the environment.
    total_turns : int
        The total number of turns in the environment.
    turn_state_dict : dict[TurnNumber, State]
        A dictionary of states for each turn.
    """

    agent: Agent
    total_turns: int
    turn_state_dict: dict[TurnNumber, State]

    def __init__(self, environment: Environment, wealth: float = 1.0):
        """
        Initialize the SingleRunInternalState object.

        Parameters
        ----------
        environment : Environment
            The Environment object.
        wealth : float, optional
            The initial wealth of the agent. Defaults to 1.0.

        Attributes
        ----------
        environment : Environment
            The Environment object.
        agent : Agent
            The Agent object.
        total_turns : int
            The total number of turns in the environment.
        turn_state_dict : dict[TurnNumber, State]
            A dictionary of states for each turn.
        current_wealth : float
            The current wealth of the agent.
        """
        self.environment = environment
        self.agent = environment.agent
        self.total_turns = environment.total_turns
        self.turn_state_dict = {}
        self.current_wealth = wealth

    def forward_step(self, mode: Mode, all_greedy: bool = False) -> float:
        """
        Simulate a single run of the environment over a specified number of turns.

        For each turn, the function:
        1. Initializes the state with the current wealth.
        2. Determines the allocation of assets and whether the allocation is greedy
        by querying the agent's policy.
        3. Updates the current wealth based on the allocation and the random return
        from the risky asset.
        4. Stores the state in the turn_state_dict for future reference.

        Asserts:
            - The allocation and is_greedy_allocation must be set for each state.

        This function modifies the [current_wealth] and populates the [turn_state_dict]
        with the state of each turn.

        After this function, the [turn_state_dict] will contain the state of each turn.
        """

        for turn in range(self.total_turns):
            state = State(turn=turn, wealth=self.current_wealth)

            if (
                mode == "Prod"
                and (state.turn, state.wealth)
                not in self.agent.state_action_value_dict.keys()
            ):
                # Initialize the state-action value dictionary, disale when the mode is ["Test"]
                self.agent.state_action_value_dict[(state.turn, state.wealth)] = {
                    action: self.current_wealth
                    for action in self.agent.possible_actions
                }

            state.set_allocation_and_is_greedy(
                *self.agent.get_allocation(
                    turn_number=turn, wealth=state.wealth, all_greedy=all_greedy
                )
            )

            assert (
                state.selected_allocation is not None
                and state.is_greedy_allocation is not None
            ), "Allocation and is_greedy_allocation should always be set here"

            self.current_wealth = (
                self.environment.calculate_new_wealth_with_random_risky_return(
                    current_wealth=self.current_wealth,
                    risky_asset_allocation=state.selected_allocation,
                )
            )

            # Store the state in the turn_state_dict
            self.turn_state_dict[turn] = state

        return self.current_wealth

    def backup_agent_from_turn_state_dict(
        self,
    ) -> tuple[StateActionValueDict, float, bool]:
        """
        Back up the agent's state-action value dictionary using the TD update.
        This function is expected to return the new action value dictionary to be
        passed back to the [Agent] class

        The update is defined as:
        Q(s_t, a_t) = Q(s_t, a_t) + alpha * (Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))

        The TD update is applied in reverse order of the turns, i.e. from the last turn
        to the first turn. For each turn, the function:
        1. If the current turn is the last turn, updates the expected return of the selected
        allocation using the TD update with the current wealth as the target.
        2. Otherwise, updates the expected return of the selected allocation using the TD update
        with the expected return of the next state as the target, if the next state is selected
        greedily. Skip backup if the next state is not selected greedily (i.e. randomly).

        Asserts:
            - The selected allocation and is_greedy_allocation of each state must be set.
            - The next state's allocation and is_greedy_allocation must be set if the next state
            is selected greedily.

        Returns:
            tuple[StateActionValueDict, float]:
                StateActionValueDict: The updated state-action value dictionary.
                float: the abs maximum percent difference between the old and new state action value dictionary
                bool: is the policy stable
        """

        def get_reversed_turn_numbers(
            turn_state_dict: dict[TurnNumber, State],
        ) -> list[TurnNumber]:
            """Get a list of reversed turn numbers from the turn_state_dict."""
            return list(reversed(sorted(turn_state_dict.keys())))

        original_state_action_value_dict = copy.deepcopy(
            self.agent.state_action_value_dict
        )
        state_action_value_dict = self.agent.state_action_value_dict
        turns = get_reversed_turn_numbers(self.turn_state_dict)

        for idx, turn in enumerate(turns):
            state = self.turn_state_dict[turn]
            assert (
                state.selected_allocation is not None
                and state.is_greedy_allocation is not None
            )
            selected_allocation_expected_return = state_action_value_dict[
                (turn, state.wealth)
            ][state.selected_allocation]

            # Update the expected return of the last turn
            if idx == 0:
                if turn == self.total_turns - 1:
                    state_action_value_dict[(turn, state.wealth)][
                        state.selected_allocation
                    ] += self.agent.alpha * (
                        self.current_wealth - selected_allocation_expected_return
                    )
            else:
                next_state = self.turn_state_dict[turn + 1]
                assert (
                    next_state.wealth is not None
                    and next_state.is_greedy_allocation is not None
                ), "Allocation and is_greedy_allocation should always be set here"

                # Update the expected return of the current turn if the next state is selected greedily
                if next_state.is_greedy_allocation:
                    assert next_state.selected_allocation is not None, (
                        "Next state's allocation should always be set here"
                    )
                    next_state_return = state_action_value_dict[
                        (turn + 1, next_state.wealth)
                    ][next_state.selected_allocation]

                    state_action_value_dict[(turn, state.wealth)][
                        state.selected_allocation
                    ] += self.agent.alpha * (
                        next_state_return - selected_allocation_expected_return
                    )

                else:
                    # skipping update as the next state is not selected greedily
                    continue

        # Calculate the max percent difference between old and new state_value_dict
        max_percent_diff: float = (
            self.calcualte_max_percent_diff_between_state_action_value_dict(
                state_action_value_dict, original_state_action_value_dict
            )
        )

        # Evaluate if the policy is stable
        is_policy_stable = self.is_policy_stable(
            state_action_value_dict, original_state_action_value_dict
        )

        return state_action_value_dict, max_percent_diff, is_policy_stable

    @staticmethod
    def calcualte_max_percent_diff_between_state_action_value_dict(
        state_action_value_dict: StateActionValueDict,
        original_state_action_value_dict: StateActionValueDict,
    ) -> float:
        """
        Calculate the maximum percent difference between the old and new state-action value dictionaries.

        Args:
            state_action_value_dict (StateActionValueDict): The new state-action value dictionary.
            original_state_action_value_dict (StateActionValueDict): The old state-action value dictionary.

        Returns:
            float: The maximum percent difference between the old and new state-action value dictionaries.
        """
        max_percent_diff: float = 0
        for key in state_action_value_dict.keys():
            if key not in original_state_action_value_dict:
                max_percent_diff = (
                    float(
                        "inf"
                    )  # the dictionary is not stable, we return inf for the max_diff
                )
                break
            for action in state_action_value_dict[key].keys():
                if action not in original_state_action_value_dict[key]:
                    max_percent_diff = (
                        float(
                            "inf"
                        )  # the dictionary is not stable, we return inf for the max_diff
                    )

                    break
                # We use percent diff to normalize the wealth level difference
                max_percent_diff = max(
                    max_percent_diff,
                    abs(
                        original_state_action_value_dict[key][action]
                        - state_action_value_dict[key][action]
                    )
                    / original_state_action_value_dict[key][action]
                    if original_state_action_value_dict[key][action] != 0
                    else float("inf"),
                )
        return max_percent_diff

    @staticmethod
    def is_policy_stable(
        state_action_value_dict: StateActionValueDict,
        original_state_action_value_dict: StateActionValueDict,
    ) -> bool:
        """
        Evaluate if the policy is stable.

        Args:
            state_action_value_dict: The new state-action value dictionary after the backup.
            original_state_action_value_dict: The original state-action value dictionary before the backup.

        Returns:
            bool: True if the policy is stable, False otherwise.
        """

        def max_value_action(action_value_dict: ActionValueDict) -> float:
            max_allocation = list(action_value_dict.keys())[0]
            max_value = -float("inf")
            for allocation, value in action_value_dict.items():
                if value > max_value:
                    max_value = value
                    max_allocation = allocation
            return max_allocation

        is_policy_stable = True
        for key in state_action_value_dict.keys():
            for action in state_action_value_dict[key].keys():
                if (
                    key not in original_state_action_value_dict.keys()
                    or action not in original_state_action_value_dict[key].keys()
                ):
                    is_policy_stable = False
                    break

                if (
                    max_value_action(
                        state_action_value_dict[key],
                    )  # type: ignore
                    != max_value_action(
                        original_state_action_value_dict[key],
                    )
                ):
                    is_policy_stable = False
                    break

        return is_policy_stable

    def train_one_step(self, mode: Mode) -> tuple[StateActionValueDict, float, bool]:
        """Train the agent for one step in the environment.
        Runs the environment for one step, and then back up the agent's state-action

        Read the [forward_step] and [backup_agent_from_turn_state_dict] functions
        for detail implementation."""
        self.forward_step(mode=mode)
        return self.backup_agent_from_turn_state_dict()
