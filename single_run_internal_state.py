from typing import Optional
from environment import Environment
from agent import Agent
from abstract_types import TurnNumber, Allocation, StateActionValueDict, Mode


class State:
    turn: TurnNumber
    wealth: float
    selected_allocation: Optional[Allocation]
    is_greedy_allocation: Optional[bool]

    def __init__(self, turn: int, wealth: float):
        self.turn = turn
        self.wealth = wealth
        self.selected_allocation = None
        self.is_greedy_allocation = None

    def set_allocation_and_is_greedy(
        self, allocation: Allocation, is_greedy_allocation: bool
    ):
        assert self.selected_allocation is None, (
            "Allocation already set, this should never be reached"
        )
        assert self.is_greedy_allocation is None, (
            "[is_greedy_allocation] already set, this should never be reached"
        )

        self.selected_allocation = allocation
        self.is_greedy_allocation = is_greedy_allocation


class SingleRunInternalState:
    agent: Agent
    total_turns: int
    turn_state_dict: dict[TurnNumber, State]

    def __init__(self, environment: Environment, wealth: float = 1.0):
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

            self.turn_state_dict[turn] = state

        return self.current_wealth

    def backup_agent_from_turn_state_dict(self) -> StateActionValueDict:
        def get_reversed_turn_numbers(turn_state_dict: dict[TurnNumber, State]):
            return list(reversed(sorted(turn_state_dict.keys())))

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

        return state_action_value_dict

    def train_one_step(self, mode: Mode) -> StateActionValueDict:
        self.forward_step(mode=mode)
        return self.backup_agent_from_turn_state_dict()
