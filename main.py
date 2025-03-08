from environment import Environment
import click
from single_run_internal_state import SingleRunInternalState
import print_utils
import matplotlib.pyplot as plt
from logger import update_logger, logger, LogLevel
import logging
from typing import Optional


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
    "-a",
    "--alpha",
    type=float,
    default=0.1,
    show_default=True,
    help="Learning rate of TD-0 algorithm",
)
@click.option(
    "--epsilon",
    type=float,
    default=0.001,
    show_default=True,
    help="Epsilon of TD-0 algorithm",
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

    epoch = 0
    while True:
        logger.info(f"{print_utils.rule}\nEpoch: {epoch}")

        environment.agent.debug_state_action_value_dict(show_non_default_only=True)
        single_run_internal_state = SingleRunInternalState(environment=environment)

        environment.agent.state_action_value_dict = (
            single_run_internal_state.train_one_step()
        )
        if environment.agent.is_optimal_strategy():
            logger.info("Optimal strategy found!")
            if n_epoch is None:
                break

        epoch += 1

        if n_epoch is not None and epoch == n_epoch:
            break

        # CR kleung: have a plot of optimal policy rate and a plot of policy selected per turn per run
        # CR kleung: refactor it to realtime plot instead
        # Simulate return with this policy
        avg_return = 0
        for i in range(1, 101):
            single_run_internal_state = SingleRunInternalState(environment=environment)

            new_return = single_run_internal_state.simulate_run()
            avg_return = (avg_return * (i - 1) + (new_return - avg_return)) / i
        logger.debug(f"Average return: {avg_return} of epoch {epoch}")
        avg_return_of_epochs.append(avg_return)

    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.title("Average Return over Epochs")
    plt.plot(avg_return_of_epochs)
    plt.show()


if __name__ == "__main__":
    main()
