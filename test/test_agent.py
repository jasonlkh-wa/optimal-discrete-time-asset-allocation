import pytest
from agent import Agent
import random
import logging
from logger import logger

BASE_WEALTH = 1
TOTAL_TURNS = 10


@pytest.fixture
def base_agent() -> Agent:
    """
    A fixture providing a base agent with a state-action value dictionary.

    The base agent is created with 10 turns, epsilon = 0.1, and alpha = 0.1.
    The state-action value dictionary is set to {0: 0, 1: 1} at every turn, resulting in action [1] being the optimal action every turn.
    """
    agent = Agent(
        epsilon=0.1,
        alpha=0.1,
    )
    for turn in range(TOTAL_TURNS):
        agent.state_action_value_dict[(turn, BASE_WEALTH)] = {0: 0, 1: 1}
    return agent


def test_get_greedy_allocation_with_absoulte_best_action(base_agent: Agent):
    random.seed(0)  # fix the seed for testing purpose

    base_agent.state_action_value_dict[0, 1] = {0: 0, 1: 1}
    assert base_agent.get_greedy_allocation(0, BASE_WEALTH, "Test") == 1


def test_get_greedy_allocation_with_non_absoulte_best_action(base_agent: Agent):
    """Test [get_greedy_allocation] with non-absolute best action"""
    random.seed(0)  # fix the seed for testing purpose
    base_agent.state_action_value_dict[0, BASE_WEALTH] = {0: 0, 1: 0}
    assert (
        base_agent.get_greedy_allocation(0, BASE_WEALTH, "Test") == 1
    )  # best action can be 0 or 1, 1 is determinsistic due to random seed selected

    random.seed(1)
    assert (
        base_agent.get_greedy_allocation(0, BASE_WEALTH, "Test") == 0
    )  # run with another random state which results in 0


def test_get_allocation_with_non_greedy_epsilon(base_agent: Agent):
    random.seed(0)  # fix the seed for testing purpose
    base_agent.epsilon = float(
        "inf"
    )  # guarantee random value selected is smaller than epsilon, triggering random selection
    assert base_agent.get_allocation(0, BASE_WEALTH) == (
        1,
        False,
    )  # non greedy random selection

    random.seed(1)
    assert base_agent.get_allocation(0, BASE_WEALTH) == (
        0,
        False,
    )  # another random seed which results in 0


def test_get_allocation_with_greedy_epsilon_and_absoulte_best_action(base_agent: Agent):
    """Test [get_allocation] with a greedy selection and absolute best action"""
    base_agent.epsilon = -float(
        "inf"
    )  # guarantee greedy selection since [random_value] > epsilon
    assert base_agent.get_allocation(0, BASE_WEALTH) == (1, True)


def test_get_allocation_with_greedy_epsilon_and_non_absoulte_best_action(
    base_agent: Agent,
):
    """Test [get_allocation] with a greedy selection and absolute best action"""
    random.seed(0)  # fix the seed for testing purpose
    base_agent.state_action_value_dict[0, BASE_WEALTH] = {0: 0, 1: 0}
    base_agent.epsilon = -float(
        "inf"
    )  # guarantee greedy selection since [random_value] > epsilon
    assert base_agent.get_allocation(0, BASE_WEALTH) == (1, True)

    random.seed(1)
    assert base_agent.get_allocation(0, BASE_WEALTH) == (
        0,
        True,
    )  # another random seed with 0 as random selection


def test_debug_state_action_value_dict(base_agent: Agent, caplog):
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        base_agent.debug_state_action_value_dict()

        assert "Turn: 0 Wealth: 1" in caplog.text
        assert "Action: 0, Value: 0" in caplog.text
        assert "Action: 1, Value: 1" in caplog.text
