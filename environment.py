import random
from print_utils import print_box
from agent import Agent
from abstract_types import Allocation


class Environment:
    """
    Simulate a discrete-time asset allocation problem environment.

    The environment keeps track of the yields of the risky and riskless assets, and
    the probability of each yield.

    Attributes
    ----------
    yield_a, yield_b: float
        The two possible yields of the risky asset.
    yield_r: float
        The yield of the riskless asset.
    probability_of_yield_a: float
        The probability of the first yield.
    probability_of_yield_b: float
        The probability of the second yield, which is [1 - probability_of_yield_1].
    total_turns: int
        The total number of turns in the simulation. Defaults to 10.
    policy: Policy
        The policy of the agent.
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
        self.yield_a = yield_a
        self.yield_b = yield_b
        self.yield_r = yield_r
        self.probability_of_yield_a = probability_of_yield_1
        self.probability_of_yield_b = 1 - probability_of_yield_1
        self.total_turns = total_turns
        self.agent = Agent(
            epsilon=epsilon,
            alpha=alpha,
            default_value=(1 + yield_r) ** total_turns,
        )

    @print_box()
    def show_environment(self) -> None:
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
        risky_wealth = current_wealth * risky_asset_allocation
        riskless_wealth = current_wealth * (1 - risky_asset_allocation)
        return risky_wealth * (
            1 + self.get_random_return_of_risky_asset()
        ) + riskless_wealth * (1 + self.yield_r)

    def calculate_optimal_strategy_and_expected_return(self) -> tuple[float, float]:
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
