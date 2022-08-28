from negmas.common import NegotiatorMechanismInterface
from negmas import OutcomeSpace
from negmas import Mechanism
from typing import TYPE_CHECKING, Any, Iterable, Protocol, Union, runtime_checkable
from attr import asdict, define, field

class Novelnmi(NegotiatorMechanismInterface):
    partial_agreement:int
    partial_digits:list
    partial_offer:list
    id: str
    """Mechanism session ID. That is unique for all mechanisms"""
    n_outcomes: int | float
    """Number of outcomes which may be `float('inf')` indicating infinity"""
    outcome_space: OutcomeSpace
    """Negotiation agenda as as an `OutcomeSpace` object. The most common type is `CartesianOutcomeSpace` which represents the cartesian product of a list of issues"""
    time_limit: float
    """The time limit in seconds for this negotiation session. None indicates infinity"""
    step_time_limit: float
    """The time limit in seconds for each step of ;this negotiation session. None indicates infinity"""
    negotiator_time_limit: float
    """The time limit in seconds to wait for negotiator responses of this negotiation session. None indicates infinity"""
    n_steps: int | None
    """The allowed number of steps for this negotiation. None indicates infinity"""
    dynamic_entry: bool
    """Whether it is allowed for agents to enter/leave the negotiation after it starts"""
    max_n_agents: int | None
    """Maximum allowed number of agents in the session. None indicates no limit"""
    mechanism: Mechanism
    """A reference to the mechanism. MUST NEVER BE USED BY NEGOTIATORS. **must be treated as a private member**"""
    annotation: dict[str, Any] = field(default=dict)
    """An arbitrary annotation as a `dict[str, Any]` that is always available for all agents"""