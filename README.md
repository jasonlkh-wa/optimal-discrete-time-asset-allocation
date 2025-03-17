# MDBD6000M - Optimal Discrete-Time Asset Allocation with TD-0

## For teaching assistant

It's recommended to go to the [Approach to evaluate the solution] section to
look for the final solution and proof of the solution.

## Introduction

This project is to solve the Optimal Discrete-Time Asset Allocation problem
using the TD-0 method. The problem is defined as follows:

Suppose the single-time-step return of the risky asset as
$Y_t=a, p rob=a,  and b, p rob=(1-a)$. Suppose that T =10, use the TD method to
find the Q function, and hence the optimal strategy.

In this project, we would use TD-0 approach with epsilon-greedy to find the
optimal allocation.

### Parameters of the Environment

The parameters of the environment are as follows:

-   `yield_a`: The first yield of the risky asset.
-   `yield_b`: The second yield of the risky asset.
-   `yield_r`: The yield of the riskless asset.
-   `probability_of_yield_a`: The probability of the first yield (a) of the
    risky asset, the second yield hence has 1 - p(a) chance
-   `total_turns`: The total number of turns in the environment.

### Parameters of the Agent

The agent uses an epsilon-greedy policy to select an action, and uses the TD
update to update the state-action value dictionary.

The TD update is defined as follows:

$Q(s, a) = Q(s, a) + \alpha [ V(s') - Q(s, a)]$

where Q(s, a) is the state-action value of taking action a in state s, , V(s')
is the value of the next state s', and $\alpha$ is the step size of the TD
update.

-   `epsilon`: The probability of selecting a non-greedy action.
-   `alpha`: The step size of the TD update.

## Approach to evaluate the TD-0 result

In order to evaluate the result policy, we would define a few benchmarks to
evaluate and verify the result.

### Theoretical optimal solution

Considering total timestep = 1, the expected return is given by
$$E(R) = allocation * E(R_{risky}) + (1-allocation) * yield_r$$
$$E(R_{risky}) = p * yield_a + (1-p) * yield_b$$  
Since E(risky) and E(riskless) = $yield_r$ are static within the environment. To
maximize the return, we would allocate all wealth to risky or riskless asset for
the one with higher expected return.

i.e. if $yield_a=0.5, yield_b=0, p_a=1, yield_r=0$, we always allocate to risky
asset since $E(risky) = 0.5 > 0 = E(riskless)$

Given this knowledge, we would simplify the action space. Instead of using a
continuous allocation space, we define possible allocation to be either 0 or 1,
i.e. either we allocate all wealth to risky or riskless asset.

### Policy convergence and validate policy

If the max change in q function is smaller than a small positive number, we
could check if the policy is stable.

Policy stable: if the policy does not change over an entire epoch.If, during the
policy improvement step, the policy remains the same for all states (i.e., no
action updates occur), then policy-stable remains true

Here, since we know the optimal solution of the problem, we would compare the
policy to the optimal policy to validate the solution from TD-0 approach.

### Compare q function to optimal q\*

Again, since we know the optimal allocation for a given environment, we could
compare the q function to the optimal q function of all state.

For example, $total\_turns=2, timestep=0, yield_a=0.1, p_a =1,yield_r = 0$  
we expect the q function is mapped to:  
allocate 0: 1 \* 1 \* 1.1 = 1.1  
allocate 1: 1 \* 1.1 \* 1.1 = 1.21

### Test module

To consolidate the evaluation methods mentioned above, we would set up tests in
[test_solution_proof.py], you can run by calling:

```bash
cd <repo>
pytest .
```

Considering time to run all tests, we would run all benchmark analysis on cases
with total turns = 2 / 3 / 5, and 2 set of yield combinations to show the
algorithm can adapt to different environment.

Lastly, we would add a test for total turns = 10, but we'd only compare if it
can reach the optimal policy given the run time constraint. Note that you could
still verify by running the [main.py] manually to see the performance of other
benchmarks in realtime plots.

## How to run the training

Firstly, download the dependency through  
`pip install -r requirements.txt`

The simulation can be executed using the `main.py` script with the following
command-line options to configure the environment and agent parameters:

-   `--yield-a`: Sets the yield of the risky asset with probability
    `probability-of-yield-a` (default is 0.05).
-   `--yield-b`: Sets the yield of the risky asset with probability
    `1 - probability-of-yield-a` (default is 0.01).
-   `--probability-of-yield-a`: Sets the probability of `yield-a` for the risky
    asset (default is 0.5).
-   `--yield-r`: Sets the yield of the riskless asset (default is 0.02).
-   `--total-turns`: Number of turns per epoch (default is 10).
-   `--epoch`: Number of epochs to run the simulation.

Example usage:

```bash
python main.py --yield-a 0.1 --yield-b 0.05 --probability-of-yield-a 0.6 --yield-r 0.02 --total-turns 10 --epoch 100
```

This command sets up an environment where the risky asset has a yield of 0.1
with a 60% probability, a yield of 0.05 otherwise, and the riskless asset has a
yield of 0.02. The simulation runs for 10 turns per epoch and iterates over 100
epochs.

Note that there are more args to control the behavior, run
`python main.py --help` to check out the definitions
