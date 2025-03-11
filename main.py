# CR kleung: make test cases of t=2,3,4 for showing the algo works
# CR kleung: improve the [is_optimal_strategy] function since now
#            it's very hard given if i run large t
# CR kleung: something to export value functions for inspection and reuse
# CR kleung: setup breakpoints for debugging and learn about how to use
# CR kleung: add analysis for q* vs q in state-action-value dict (which is E(yield) + 1 ** n_turn)
from environment import Environment
import click
from single_run_internal_state import SingleRunInternalState
import print_utils
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from logger import update_logger, logger, LogLevel
import logging
from typing import Optional, Tuple


@click.command()
@click.option(
    "-a",
    "--yield-a",
    type=float,
    default=0.05,
    show_default=True,
    help="Yield of risky asset with probability [probability-of-yield-a]",
)
@click.option(
    "-b",
    "--yield-b",
    type=float,
    default=0.01,
    show_default=True,
    help="Yield of risky asset with probability [1 - probability-of-yield-a]",
)
@click.option(
    "-p",
    "--probability-of-yield-a",
    type=float,
    default=0.4,
    show_default=True,
    help="Probability of [yield-a] of risky asset",
)
@click.option(
    "-r",
    "--yield-r",
    type=float,
    default=0.02,
    show_default=True,
    help="Yield of riskless asset",
)
@click.option(
    "-t",
    "--total-turns",
    type=int,
    default=10,
    show_default=True,
    help="Number of turns per epoch",
)
@click.option(
    "--epoch",
    "n_epoch",
    type=int,
    default=None,
    show_default=True,
    help="Number of epochs",
)
@click.option(
    "--alpha",
    type=float,
    default=0.1,
    show_default=True,
    help="Learning rate of TD-0 algorithm",
)
@click.option(
    "--epsilon",
    type=float,
    default=0.01,
    show_default=True,
    help="Epsilon of TD-0 algorithm",
)
@click.option(
    "--epsilon-decay",
    type=float,
    default=0.9999,
    show_default=True,
    help="Epsilon decay multiplier per turn",
)
@click.option(
    "--log-level",
    type=click.Choice(list(logging._nameToLevel.keys()), case_sensitive=False),
    default="INFO",
)
def main(
    yield_a: float,
    yield_b: float,
    probability_of_yield_a: float,
    yield_r: float,
    total_turns: int,
    n_epoch: Optional[int],
    alpha: float,
    epsilon: float,
    epsilon_decay: float,
    log_level: LogLevel,
):
    update_logger(logger, log_level)
    environment = Environment(
        yield_a=yield_a,
        yield_b=yield_b,
        yield_r=yield_r,
        probability_of_yield_1=probability_of_yield_a,
        total_turns=total_turns,
        alpha=alpha,
        epsilon=epsilon,
    )
    environment.show_environment()

    avg_return_of_epochs = []
    returns_of_epochs = []
    optimal_stratgy, optimal_expected_return = (
        environment.calculate_optimal_strategy_and_expected_return()
    )

    axes: list[Axes]
    plt.ion()
    _fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes_dict = {
        "average_return_ax": axes[0],
        "epsilon_ax": axes[1],
        "alpha_ax": axes[2],
    }

    axes[0].set_xlabel("Epoch")
    axes_dict["average_return_ax"].set_ylabel("Average Return")
    axes_dict["average_return_ax"].set_title("Average return over epochs")
    (average_return_line,) = axes_dict["average_return_ax"].plot([], [])
    (optimal_expected_return_line,) = axes_dict["average_return_ax"].plot(
        [], [], label="optimal", color="r"
    )
    axes_dict["average_return_ax"].legend()

    axes_dict["epsilon_ax"].set_xlabel("Epoch")
    axes_dict["epsilon_ax"].set_ylabel("Episilon")
    axes_dict["epsilon_ax"].set_title("Epsilon over epochs")
    (epsilon_line,) = axes_dict["epsilon_ax"].plot([], [], label="Epsilon", color="r")

    axes_dict["alpha_ax"].set_xlabel("Epoch")
    axes_dict["alpha_ax"].set_ylabel("Alpha")
    axes_dict["alpha_ax"].set_title("Alpha over epochs")
    (alpha_line,) = axes_dict["alpha_ax"].plot([], [], label="Alpha", color="b")

    epoch = 0
    epsilon_values = []
    alpha_values = []

    epoch = 0
    while True:
        if n_epoch is not None and epoch == n_epoch:
            break

        logger.info(f"{print_utils.rule}\nEpoch: {epoch} {epsilon=}")
        environment.agent.debug_state_action_value_dict(show_non_default_only=True)
        single_run_internal_state = SingleRunInternalState(environment=environment)

        environment.agent.state_action_value_dict = (
            single_run_internal_state.train_one_step(mode="Prod")
        )

        # CR kleung: have a plot of optimal policy rate and a plot of policy selected per turn per run
        # Simulate return with this policy
        test_returns = []
        total_test_run = 100
        for _ in range(total_test_run):
            single_run_internal_state = SingleRunInternalState(environment=environment)

            new_return = single_run_internal_state.forward_step(
                mode="Prod", all_greedy=True
            )
            test_returns.append(new_return)

        # Calculate and append average return of this epoch
        avg_return = sum(test_returns) / len(test_returns)
        logger.debug(f"Average return: {avg_return} of epoch {epoch}")
        avg_return_of_epochs.append(avg_return)
        returns_of_epochs.append(test_returns)

        # Append epsilon and alpha value
        epsilon_values.append(epsilon)
        alpha_values.append(environment.agent.alpha)

        average_return_line.set_xdata(range(epoch + 1))
        average_return_line.set_ydata(avg_return_of_epochs)
        optimal_expected_return_line.set_xdata(range(epoch + 1))
        optimal_expected_return_line.set_ydata(
            [optimal_expected_return for _ in range(len(avg_return_of_epochs))]
        )
        axes_dict["average_return_ax"].relim()
        axes_dict["average_return_ax"].autoscale_view()

        epsilon_line.set_xdata(range(epoch + 1))
        epsilon_line.set_ydata(epsilon_values)

        axes_dict["epsilon_ax"].relim()
        axes_dict["epsilon_ax"].autoscale_view()

        alpha_line.set_xdata(range(epoch + 1))
        alpha_line.set_ydata(alpha_values)
        axes_dict["alpha_ax"].relim()
        axes_dict["alpha_ax"].autoscale_view()

        # refresh graph
        plt.draw()
        plt.pause(1e-10)

        if environment.agent.is_optimal_strategy(optimal_strategy=optimal_stratgy):
            logger.info("Optimal strategy found!")
            if n_epoch is None:
                break

        epsilon *= epsilon_decay
        environment.agent.alpha = max(environment.agent.alpha * epsilon_decay, 0.1)
        epoch += 1

    plt.ioff()
    plt.show()
    print(environment.agent.state_action_value_dict)
    print(returns_of_epochs[-1])
    print(f"{yield_a=},{yield_b=},{probability_of_yield_a=},{yield_r=}")


if __name__ == "__main__":
    main()
