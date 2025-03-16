# CR kleung: make test cases of t=2,3,4 for showing the algo works, use stable V (0.05) and stable policy to proof
# CR kleung: test is ~100% coverage, do documentation and readme
from environment import Environment
import click
from single_run_internal_state import SingleRunInternalState
import print_utils
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike
from logger import update_logger, logger, LogLevel
import logging
from typing import Optional
import time
import datetime
import joblib
import os
import textwrap

plt.rcParams.update(
    {
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


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
    default=0.5,
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
    help="Epsilon decay multiplier per turn, new = old * decay",
)
@click.option(
    "--min-epsilon",
    type=float,
    default=0.2,
    show_default=True,
    help="Minimum epsilon of TD-0 algorithm after applying decay",
)
@click.option(
    "--log-level",
    type=click.Choice(list(logging._nameToLevel.keys()), case_sensitive=False),
    default="INFO",
)
@click.option(
    "--env-path",
    type=click.Path(exists=True),
    help="Import environment from a file",
)
@click.option(
    "--disable-train",
    is_flag=True,
    help="Disable agent training steps",
)
@click.option(
    "--disable-simulation", is_flag=True, help="Disable agent simulation tests and plot"
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
    min_epsilon: float,
    log_level: LogLevel,
    env_path: Optional[os.PathLike],
    disable_train: bool,
    disable_simulation: bool,
):
    update_logger(logger, log_level)

    if env_path is None:
        environment = Environment(
            yield_a=yield_a,
            yield_b=yield_b,
            yield_r=yield_r,
            probability_of_yield_1=probability_of_yield_a,
            total_turns=total_turns,
            alpha=alpha,
            epsilon=epsilon,
        )
    else:
        environment: Environment = joblib.load(env_path)

    assert isinstance(environment, Environment), (
        "Environment cannot be loaded from [env_path]"
    )
    environment.show_environment()

    avg_return_of_epochs = []
    avg_optimal_action_count_per_run_list = []
    optimal_stratgy, optimal_expected_return = (
        environment.calculate_optimal_strategy_and_expected_return()
    )

    # Initialize variables for surpress type hinting
    epsilon_values: Optional[list[float]] = None
    axes_dict: Optional[dict[str, Axes]] = None
    q_function_diffs: Optional[list[float]] = None
    optimal_expected_return_line: Optional[Line2D] = None
    average_return_line: Optional[Line2D] = None
    epsilon_line: Optional[Line2D] = None
    optimal_action_count_line: Optional[Line2D] = None
    q_function_diff_line: Optional[Line2D] = None

    # We don't initialize graphs if simulation is disabled to save resources
    if not disable_simulation:
        axes: list[list[Axes]]
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 5))
        fig.tight_layout(pad=5)

        axes_dict = {
            "average_return_ax": axes[0][0],
            "optimal_action_count_ax": axes[0][1],
            "q_function_diff_ax": axes[1][0],
            "epsilon_ax": axes[1][1],
        }

        # Average return plot
        update_ax_properties(
            axes_dict["average_return_ax"],
            x_label="Epoch",
            y_label="Average Return",
            title="Average return over epochs",
        )

        (average_return_line,) = axes_dict["average_return_ax"].plot([], [])
        (optimal_expected_return_line,) = axes_dict["average_return_ax"].plot(
            [], [], label="optimal", color="r"
        )
        axes_dict["average_return_ax"].legend()

        # Optimal action count plot
        update_ax_properties(
            axes_dict["optimal_action_count_ax"],
            x_label="Epoch",
            y_label="Optimal Action Count",
            title=wrap_title("Average optimal action count over epochs", width=40),
        )

        (optimal_action_count_line,) = axes_dict["optimal_action_count_ax"].plot(
            [], [], color="b"
        )

        # Optimal q function vs current q function plot
        update_ax_properties(
            axes_dict["q_function_diff_ax"],
            x_label="Epoch",
            y_label="Max percent difference",
            title=wrap_title(
                "Max percent difference of Q function against Q* function over epochs",
                width=40,
            ),
        )
        (q_function_diff_line,) = axes_dict["q_function_diff_ax"].plot(
            [], [], color="b"
        )

        # Epsilon plot
        update_ax_properties(
            axes_dict["epsilon_ax"],
            x_label="Epoch",
            y_label="Epsilon",
            title="Epsilon over epochs",
        )
        (epsilon_line,) = axes_dict["epsilon_ax"].plot(
            [], [], label="Epsilon", color="r"
        )

        q_function_diffs = []
        epsilon_values = []

    epoch = 0
    while True:
        # If [n_epoch] is not defined, train agent until reaching the optimal
        # strategy, i.e. selecting the optimal action at every turn greedily
        if n_epoch is not None and epoch == n_epoch:
            break

        logger.info(
            f"{print_utils.rule}\nEpoch: {epoch} {epsilon=} {alpha=} {environment.calculate_max_q_function_percent_diff(environment.agent.state_action_value_dict)=}"
        )
        environment.agent.debug_state_action_value_dict()
        single_run_internal_state = SingleRunInternalState(environment=environment)

        if not disable_train:
            environment.agent.state_action_value_dict = (
                single_run_internal_state.train_one_step(mode="Prod")
            )

        if not disable_simulation:
            try:
                assert avg_optimal_action_count_per_run_list is not None
                assert q_function_diffs is not None
                assert epsilon_values is not None
                assert axes_dict is not None
                assert optimal_expected_return_line is not None
                assert average_return_line is not None
                assert epsilon_line is not None
                assert optimal_action_count_line is not None
                assert q_function_diff_line is not None
            except AssertionError as e:
                logger.error("Unexpected plot variable being None")
                raise e

            avg_return, avg_optimal_action_count_per_run = run_test_simulation(
                environment=environment,
                optimal_stratgy=optimal_stratgy,
                n_run=100,
            )

            # Calculate and append average return of this epoch
            logger.debug(f"Average return: {avg_return} of epoch {epoch}")
            avg_return_of_epochs.append(avg_return)
            avg_optimal_action_count_per_run_list.append(
                avg_optimal_action_count_per_run
            )

            # Append epsilon value
            epsilon_values.append(epsilon)

            # Update average return plot
            update_realtime_graph(
                axes_dict["average_return_ax"],
                average_return_line,
                xdata=range(epoch + 1),
                ydata=avg_return_of_epochs,
            )
            # Update optimal expected return plot
            update_realtime_graph(
                axes_dict["average_return_ax"],
                optimal_expected_return_line,
                xdata=range(epoch + 1),
                ydata=[
                    optimal_expected_return for _ in range(len(avg_return_of_epochs))
                ],
            )

            # Update optimal action count plot
            update_realtime_graph(
                axes_dict["optimal_action_count_ax"],
                optimal_action_count_line,
                xdata=range(epoch + 1),
                ydata=avg_optimal_action_count_per_run_list,
            )

            # Update optimal action count plot
            q_function_diffs.append(
                environment.calculate_max_q_function_percent_diff(
                    environment.agent.state_action_value_dict
                )
            )
            update_realtime_graph(
                axes_dict["q_function_diff_ax"],
                q_function_diff_line,
                xdata=range(epoch + 1),
                ydata=q_function_diffs,
            )

            # Update epsilon plot
            update_realtime_graph(
                axes_dict["epsilon_ax"],
                epsilon_line,
                xdata=range(epoch + 1),
                ydata=epsilon_values,
            )

            # refresh graph
            plt.draw()
            plt.pause(1e-5)

        if environment.is_optimal_strategy():
            logger.info("Optimal strategy found!")
            if n_epoch is None:
                break

        # epsilon decay
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        epoch += 1

        logger.info(
            f"{environment.calculate_max_q_function_percent_diff(environment.agent.state_action_value_dict)=}"
        )

    # Show final graph to avoid realtime graph closing automatically
    if not disable_simulation:
        plt.ioff()
        plt.show()

    # Export environment
    environment.export_environment(
        f"./data/env_{datetime.datetime.today().strftime('%Y%m%d')}_{time.time()}_total_turns_{environment.total_turns}.joblib"
    )


def run_test_simulation(
    environment: Environment, optimal_stratgy: float, n_run: int
) -> tuple[float, float]:
    test_returns = []
    optimal_strategy_count = 0
    for _ in range(n_run):
        single_run_internal_state = SingleRunInternalState(environment=environment)

        new_return = single_run_internal_state.forward_step(
            mode="Prod", all_greedy=True
        )
        optimal_strategy_count += sum(
            [
                1 if state.selected_allocation == optimal_stratgy else 0
                for state in single_run_internal_state.turn_state_dict.values()
            ]
        )
        test_returns.append(new_return)

    avg_return = sum(test_returns) / n_run
    avg_optimal_action_count_per_run = optimal_strategy_count / n_run
    return (avg_return, avg_optimal_action_count_per_run)


def update_realtime_graph(ax: Axes, line: Line2D, xdata: ArrayLike, ydata: ArrayLike):
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    ax.relim()
    ax.autoscale_view()


def update_ax_properties(
    ax: Axes,
    title: str,
    x_label: str,
    y_label: str,
):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        title,
        fontsize=8,
    )


def wrap_title(title: str, width=40) -> str:
    return "\n".join(textwrap.wrap(title, width=width))


if __name__ == "__main__":
    main()
