"""
Tests for validating the TD-0 solution of the discrete-time asset allocation problem.
For each case, we would validate (check [README.md] for details):
* Optimal policy reached
* Policy is stable
* Q function update is smaller than threshold
* low q vs q* difference

The tests in this file are used to validate the TD-0 solution of the
discrete-time asset allocation problem. The tests use the benchmark
environment and agent to compare the q functions and the policy of the
agent with the optimal q functions and the optimal policy.

The tests are run with a fixed seed for reproducibility, and the tests
are run for a specified number of epochs. The maximum difference between
the q functions and the policy are checked to ensure that the solution
is stable.

The tests are run for a specified number of epochs, and the maximum
difference between the q functions and the policy are checked to ensure
that the solution is stable.
"""

from environment import Environment
from single_run_internal_state import SingleRunInternalState
import random


def full_test_run(environment: Environment, epoch: int):
    random.seed(1)
    max_state_action_value_dict_diffs = []
    max_state_action_value_dict_diff = float("inf")
    is_policy_stable = False
    q_function_diff_threshold = 1e-4
    for _ in range(epoch):
        single_run_internal_state = SingleRunInternalState(environment=environment)

        # Update the agent's state-action value dictionary
        (
            environment.agent.state_action_value_dict,
            max_state_action_value_dict_diff,
            is_policy_stable,
        ) = single_run_internal_state.train_one_step(mode="Prod")
        max_state_action_value_dict_diffs.append(max_state_action_value_dict_diff)

        # Although we have epoch, we could break the loop if the training has
        # potentially passed all the tests
        if (
            max_state_action_value_dict_diff < q_function_diff_threshold
            and environment.is_optimal_strategy()
        ):
            break

    assert (
        min(max_state_action_value_dict_diffs) < q_function_diff_threshold
    )  # assert the max diff between q function update is less than threshold
    assert is_policy_stable  # assert the policy is stable
    assert (
        environment.is_optimal_strategy()
    )  # assert the policy is reaching the theoretical optimal
    assert (
        environment.calculate_avg_q_function_percent_diff(
            environment.agent.state_action_value_dict
        )
        < 0.05  # assert the max diff between q and q* is less than 5%
    )


def test_two_timestep_first_setup():
    environment = Environment(
        yield_a=0.3,
        yield_b=0.1,
        yield_r=0.1,
        probability_of_yield_1=0.5,
        total_turns=2,
        alpha=0.3,
        epsilon=0.4,
    )
    full_test_run(environment, 10_000)


def test_two_timestep_second_setup():
    environment = Environment(
        yield_a=0.4,
        yield_b=0.2,
        yield_r=-0.1,
        probability_of_yield_1=0.5,
        total_turns=2,
        alpha=0.3,
        epsilon=0.4,
    )
    full_test_run(environment, 10_000)


def test_three_timestep_first_setup():
    environment = Environment(
        yield_a=0.3,
        yield_b=0.1,
        yield_r=0.1,
        probability_of_yield_1=0.5,
        total_turns=3,
        alpha=0.3,
        epsilon=0.4,
    )
    full_test_run(environment, 10_000)


def test_three_timestep_second_setup():
    environment = Environment(
        yield_a=0.4,
        yield_b=0.2,
        yield_r=-0.1,
        probability_of_yield_1=0.5,
        total_turns=3,
        alpha=0.3,
        epsilon=0.4,
    )
    full_test_run(environment, 10_000)


def test_five_timestep_first_setup():
    environment = Environment(
        yield_a=0.3,
        yield_b=0.1,
        yield_r=0.1,
        probability_of_yield_1=0.5,
        total_turns=5,
        alpha=0.3,
        epsilon=0.4,
    )
    full_test_run(environment, 10_000)


def test_five_timestep_second_setup():
    environment = Environment(
        yield_a=0.4,
        yield_b=0.2,
        yield_r=-0.1,
        probability_of_yield_1=0.5,
        total_turns=5,
        alpha=0.3,
        epsilon=0.4,
    )
    full_test_run(environment, 10_000)


# Comment out the code if you're interested with the result on
# ten timesteps, note that it takes a long time to complete the tests
# *** It is recommended to run the [main.py] to see the full run since
# the set up here is simplified with less flexibility to epsilon and alpha.

# def test_ten_timestep_first_setup():
#     environment = Environment(
#         yield_a=0.3,
#         yield_b=0.1,
#         yield_r=0.1,
#         probability_of_yield_1=0.5,
#         total_turns=10,
#         alpha=0.3,
#         epsilon=0.4,
#     )
#     full_test_run(environment, 100_000)


# def test_ten_timestep_second_setup():
#     environment = Environment(
#         yield_a=0.4,
#         yield_b=0.2,
#         yield_r=-0.1,
#         probability_of_yield_1=0.5,
#         total_turns=10,
#         alpha=0.3,
#         epsilon=0.4,
#     )
#     full_test_run(environment, 100_000)
