import pytest
from environment import Environment
from single_run_internal_state import SingleRunInternalState
import numpy as np

GREEDY_ACTION = 0.5
BASE_WEALTH = 1


@pytest.fixture
def simple_environment():
    environment = Environment(
        yield_a=0.1,
        yield_b=0.2,
        probability_of_yield_1=1,  # always yield_a
        yield_r=0.15,
        total_turns=10,
        alpha=0.1,
        epsilon=0,  # always greedy
    )
    environment.agent.possible_actions = np.array(
        [0, GREEDY_ACTION]
    )  # adding 0.5 as the possible action for testing
    # CR-soon kleung: add doc here for how we calculate this dict
    environment.agent.state_action_value_dict = {
        (
            turn,
            BASE_WEALTH
            * ((1 + (environment.yield_r + environment.yield_a) / 2) ** turn),
        ): {
            0: 0,
            GREEDY_ACTION: 1,
        }
        for turn in range(environment.total_turns)
    }  # action [0.5] is always optimal
    return environment


@pytest.fixture
def simple_single_run_internal_state(simple_environment):
    return SingleRunInternalState(simple_environment)


def test_simulate_run_with_all_greedy_calculation(
    simple_environment, simple_single_run_internal_state
):
    simple_single_run_internal_state.simulate_run(mode="Test")

    assert (
        len(simple_single_run_internal_state.turn_state_dict)
        == simple_environment.total_turns
    )
    assert (
        simple_single_run_internal_state.current_wealth
        == (
            1
            + (simple_environment.yield_a + simple_environment.yield_r) * GREEDY_ACTION
        )
        ** simple_environment.total_turns
    )  # testing wealth calculation

    # Testing each state
    current_wealth = 1
    for turn, state in simple_single_run_internal_state.turn_state_dict.items():
        assert turn == state.turn
        assert state.selected_allocation == GREEDY_ACTION
        assert state.is_greedy_allocation
        assert state.wealth == current_wealth
        current_wealth = current_wealth * (
            1
            + (simple_environment.yield_a + simple_environment.yield_r) * GREEDY_ACTION
        )


# CR kleung: not here, but all turn may need to break into (turn, wealth)
def test_train_one_step(simple_environment, simple_single_run_internal_state):
    new_state_action_value_dict = simple_single_run_internal_state.train_one_step(
        mode="Test"
    )
    final_wealth = simple_single_run_internal_state.current_wealth
    for (turn, wealth), action_value_dict in new_state_action_value_dict.items():
        assert action_value_dict[0] == 0
        if turn == simple_environment.total_turns - 1:
            assert (
                action_value_dict[GREEDY_ACTION]
                == 1 + (final_wealth - 1) * simple_environment.agent.alpha
            )
        else:
            assert (
                action_value_dict[GREEDY_ACTION]
                == 1
                + (
                    new_state_action_value_dict[
                        turn + 1,
                        simple_single_run_internal_state.turn_state_dict[
                            turn + 1
                        ].wealth,
                    ][GREEDY_ACTION]
                    - 1
                )
                * simple_environment.agent.alpha
            )
