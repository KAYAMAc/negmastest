#------------------------------------------------------------------------
# Based on Rubinstein Bargaining Protocol (Stateful Mechanism)
#------------------------------------------------------------------------

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from matplotlib.ticker import StrMethodFormatter

from abc import abstractmethod, ABC
from attr import define, field
from random import random
from negmas import Mechanism, MechanismRoundResult, Negotiator, PolyAspiration
from negmas import MechanismState
from negmas import NegotiatorMechanismInterface
from typing import Callable, Tuple, Optional, List, Any, Dict
from negmas import (
    Outcome,
    make_issue,
    UtilityFunction,
    LinearUtilityFunction,
    ExpDiscountedUFun,
)
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

@define
class RubinsteinMechanismState(MechanismState):
    discounts = field(default=list)

# We can then define the mechanism class itself
class RubinsteinMechanism(Mechanism):
    # Simplified Rubinstein's Mechanism with Exponential discounting

    def __init__(self, num_issues, extended=False, **kwargs):
        kwargs.update(
            dict(
                issues=[make_issue(values=(0.0, 1.0), name="issue%d" % i) for i in range(num_issues)],
                max_n_agents=2,
                dynamic_entry=False,
                state_factory=RubinsteinMechanismState,
            )
        )
        super().__init__(**kwargs)
        self.add_requirements(dict(propose=True, set_index=True))
        self.state.discounts = []
        self.proposals = []
        self.extended = extended

    def add(self, negotiator: "Negotiator", *, discount: float = 0.95,  **kwargs,) -> Optional[bool]:
        weights = [1, 0] if len(self.negotiators) == 0 else [0, 1]
        ufun = ExpDiscountedUFun(
            LinearUtilityFunction(weights, outcome_space=self.outcome_space),
            outcome_space=self.outcome_space,
            discount=discount)
        added = super().add(negotiator, ufun=ufun, role=None, **kwargs)
        if added:
            self.state.discounts.append(discount)

    def round(self) -> MechanismRoundResult:
        """One round of the mechanism"""
        if self.current_step == 0:
            if len(self.negotiators) != 2:
                return MechanismRoundResult(
                    error=True,
                    error_details=f"Got {len(self.negotiators)} negotiators!!",
                    broken=True,
                )
            for i, n in enumerate(self.negotiators):
                n.set_index(i)

        outcomes = list(n.propose(self.state) for n in self.negotiators)
        self.proposals.append(outcomes)
        if any(o is None for o in outcomes):
            return MechanismRoundResult(broken=True)
        if sum(outcomes[0]) <= 1 + 1e-3:
            if self.extended:
                if (outcomes[0][0] <= outcomes[1][0] + 1e-5 and outcomes[1][1] <= outcomes[0][1] + 1e-5):
                    return MechanismRoundResult(
                        agreement=(
                            min(outcomes[0][0], outcomes[1][0]),
                            min(outcomes[0][1], outcomes[1][1]),
                        )
                    )
            elif max(abs(outcomes[0][i] - outcomes[1][i]) for i in range(2)) < 1e-3:
                return MechanismRoundResult(
                    agreement=tuple(
                        0.5 * (outcomes[0][i] + outcomes[1][i]) for i in range(2)
                    )
                )

        return MechanismRoundResult()


def plot_a_run(mechanism: RubinsteinMechanism, name : str) -> None:
    result = mechanism.state
    fig = plt.figure(figsize=(6, 6))
    x = np.linspace(0.0, 1.0, 101, endpoint=True)
    first = np.array([_[0] for _ in mechanism.proposals])
    second = np.array([_[1] for _ in mechanism.proposals])
    plt.plot(x, 1 - x, color="gray", label="Pareto-front")
    plt.xlabel("Agent 1's utility")
    plt.ylabel("Agent 2's utility")
    plt.scatter(
        first[:, 0], first[:, 1], marker="x", color="green", label="Proposals from 1"
    )
    plt.scatter(
        second[:, 0], second[:, 1], marker="+", color="blue", label="Proposals from 2"
    )
    if result.agreement is not None:
        plt.scatter(
            [result.agreement[0]],
            [result.agreement[1]],
            marker="o",
            color="red",
            label="Agreement",
        )
    plt.legend()
    plt.title(name)
    plt.grid(color='gray', linestyle='dashed', linewidth=0.1)
    plt.savefig("figures/" + name + ".png", format='png', dpi=500)
    #plt.savefig(name + ".png", format='png', dpi=500)
    plt.cla()

# Base negotiator type for this mechanism
class RubinsteinNegotiator(Negotiator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_capabilities(dict(propose=True, set_index=True))
        self.my_index = -1

    def set_index(self, indx) -> None:
        self.my_index = indx

    @abstractmethod
    def propose(self, state: RubinsteinMechanismState) -> Outcome:
        """Proposes an outcome which is a tuple of two numbers between zero and one"""

#------------------------------------------------------------------------------------------------------------
# Random negotiator that ends the negotiation if it finds that it is impossible to get a positive utility anymore (due to discounting) and otherwise returns a random apportionment of the pie.
#------------------------------------------------------------------------------------------------------------

class RandomRubinsteinNegotiator(RubinsteinNegotiator):
    def propose(self, state: RubinsteinMechanismState) -> Outcome:
        if self.ufun((1.0, 1.0)) < 0.0:
            return None
        r = random()
        return r, 1 - r

# Run negotiations using the protocol and negotiator:

mechanism = RubinsteinMechanism(num_issues=2, extended=False)
mechanism.add(RandomRubinsteinNegotiator(), discount=0.75)
mechanism.add(RandomRubinsteinNegotiator(), discount=0.75)
print(f"Agreed to: {mechanism.run().agreement} after {mechanism.current_step} steps")
plot_a_run(mechanism, "Random Rubinstein Negotiator")

#------------------------------------------------------------------------------------------------------------
# Rubinstein showed in 1982 that there is a single perfect game equilibrium of single round
# We can implement the optimal negotiator for this mechanism as follows:
#------------------------------------------------------------------------------------------------------------
class OptimalRubinsteinNegotiator(RubinsteinNegotiator):
    def propose(self, state: RubinsteinMechanismState) -> Outcome:
        first = (1 - state.discounts[1]) / (1 - state.discounts[1] * state.discounts[0])
        return first, 1 - first

mechanism = RubinsteinMechanism(num_issues=2)
mechanism.add(OptimalRubinsteinNegotiator())
mechanism.add(OptimalRubinsteinNegotiator())
print(f"Agreed to: {mechanism.run().agreement} in {mechanism.current_step} steps")
plot_a_run(mechanism, "Optimal Rubinstein Negotiator")

#------------------------------------------------------------------------------------------------------------
# Let's try to make an agent that does not use the information about the other agent's
#------------------------------------------------------------------------------------------------------------
class AspirationRubinsteinNegotiator(RubinsteinNegotiator):
    def __init__(self, *args, aspiration_type="linear", max_aspiration=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._asp = PolyAspiration(max_aspiration, aspiration_type) #A polynomially conceding curve

    def propose(self, state: RubinsteinMechanismState) -> Outcome:
        if self.ufun((1.0, 1.0)) < 0.0:
            return None
        r = self._asp.utility_at(state.relative_time)
        return (r, 1.0 - r) if self.my_index == 0 else (1.0 - r, r)

mechanism = RubinsteinMechanism(num_issues=2, n_steps=100, extended=True)
mechanism.add(AspirationRubinsteinNegotiator())
mechanism.add(AspirationRubinsteinNegotiator())
result = mechanism.run()
print(f"Agreed to: {result.agreement} in {mechanism.current_step} steps")
plot_a_run(mechanism, "Aspiration Rubinstein Negotiator")

print ("\nCase where the first negotiator is a conceder")
print ("------------------------------------------------------------------------------")
mechanism = RubinsteinMechanism(num_issues=2, n_steps=100, extended=True)
mechanism.add(AspirationRubinsteinNegotiator(aspiration_type="conceder"))
mechanism.add(AspirationRubinsteinNegotiator(aspiration_type="linear"))
print(f"Agreed to: {mechanism.run().agreement} in {mechanism.current_step} steps")
plot_a_run(mechanism, "Aspiration Rubinstein Negotiator : Conceder vs Linear")

print ("\nCase where the first negotiator is a boulware")
print ("------------------------------------------------------------------------------")
mechanism = RubinsteinMechanism(num_issues=2, n_steps=100, extended=True)
mechanism.add(AspirationRubinsteinNegotiator(aspiration_type="boulware"))
mechanism.add(AspirationRubinsteinNegotiator(aspiration_type="linear"))
print(f"Agreed to: {mechanism.run().agreement} in {mechanism.current_step} steps")
plot_a_run(mechanism, "Aspiration Rubinstein Negotiator : Boulware vs Linear")

print ("\nCase where the first negotiator is a hardheaded")
print ("------------------------------------------------------------------------------")
mechanism = RubinsteinMechanism(num_issues=2, n_steps=100, extended=True)
mechanism.add(AspirationRubinsteinNegotiator(aspiration_type="hardheaded"))
mechanism.add(AspirationRubinsteinNegotiator(aspiration_type="linear"))
print(f"Agreed to: {mechanism.run().agreement} in {mechanism.current_step} steps")
plot_a_run(mechanism, "Aspiration Rubinstein Negotiator : Hardheaded vs Linear")

print ("\nCase where the first negotiator is a hardheaded and second is boulware")
print ("------------------------------------------------------------------------------")
mechanism = RubinsteinMechanism(num_issues=2, n_steps=100, extended=True)
mechanism.add(AspirationRubinsteinNegotiator(aspiration_type="hardheaded"))
mechanism.add(AspirationRubinsteinNegotiator(aspiration_type="boulware"))
print(f"Agreed to: {mechanism.run().agreement} in {mechanism.current_step} steps")
plot_a_run(mechanism, "Aspiration Rubinstein Negotiator : Hardheaded vs Boulware")
