from environment import Environment
from collections import Counter
import random
import pytest


@pytest.fixture
def env() -> Environment:
    return Environment(
        yield_a=0.1,
        yield_b=0.2,
        probability_of_yield_1=0.3,
        yield_r=0.15,
        total_turns=10,
        alpha=0.1,
        epsilon=0.1,
    )


def test_get_random_return(env: Environment) -> None:
    """
    Test [get_random_return] method of [Environment] class with 1000 runs
    and fix the seed at [0].
    """
    random.seed(0)  # fix the seed for testing purpose
    n_run = 1_000
    percent_tolerance = 0.001

    random_counter = Counter()
    for _ in range(n_run):
        random_counter[env.get_random_return_of_risky_asset()] += 1

    assert (random_counter[env.yield_a] / n_run) - percent_tolerance < 0.296
    assert (random_counter[env.yield_b] / n_run) - percent_tolerance < 0.704


def test_calculate_new_wealth_with_random_risk_return(env: Environment):
    random.seed(0)  # fix the seed for testing purpose
    new_wealth = env.calculate_new_wealth_with_random_risky_return(1, 1)
    assert new_wealth == 1 + env.yield_b

    random.seed(1)
    new_wealth = env.calculate_new_wealth_with_random_risky_return(1, 1)
    assert new_wealth == 1 + env.yield_a  # case when random result = [yield_a]

    random.seed(1)
    new_wealth = env.calculate_new_wealth_with_random_risky_return(1, 0.5)
    assert (
        new_wealth == 1 + (env.yield_a + env.yield_r) / 2
    )  # case when random result = [yield_a] and 0.5 riskless asset


def test_calculate_optimal_strategy_and_expected_return(env: Environment):
    optimal_allocation, expected_return = (
        env.calculate_optimal_strategy_and_expected_return()
    )
    assert optimal_allocation == 1
    assert expected_return == (1 + env.expected_risky_return()) ** env.total_turns
