"""
TODO:
- parametrise all utility functions using variable ranges, within each simulation

"""


from __future__ import annotations

import functools
import random
import copy
import sys
import time
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from saomechanism import SAOMechanism
from randomeone import BetterRandomNegotiator
from negmas import make_issue
from negmas import TimeBasedConcedingNegotiator

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun
from random import randint, uniform

# https://github.com/yasserfarouk/negmas ---
from negmas.mechanisms import Mechanism, MechanismRoundResult
from negmas import warnings

from negmas.events import Event
from negmas.helpers import TimeoutCaller, TimeoutError, exception2str
from negmas.outcomes.common import Outcome
from negmas.outcomes.outcome_ops import cast_value_types, outcome_types_are_ok

from negmas import SAOMechanism # Old one
from negmas.sao.common import ResponseType, SAOResponse, SAOState
from collections import defaultdict, namedtuple

"""
By implementing the single round() function, a new protocol is created.
New negotiators can be added to the negotiation using add() and removed using remove().
See the documentation for a full description of Mechanism available functionality out of the box.
"""

class MyNovelSAOMechanism(SAOMechanism):
    """One round of the protocol"""
    def round(self) -> MechanismRoundResult:
        """implements a round of the Stacked Alternating Offers Protocol."""
        state = self._current_state
        if self._frozen_neg_list is None:
            state.new_offers = []
        negotiators: list[SAONegotiator] = self.negotiators
        n_negotiators = len(negotiators)
        # times = dict(zip([_.id for _ in negotiators], itertools.repeat(0.0)))
        times = defaultdict(float, self._waiting_time)
        exceptions = dict(
            zip([_.id for _ in negotiators], [list() for _ in negotiators])
        )

        def _safe_counter(negotiator, *args, **kwargs) -> tuple[SAOResponse | None, bool]:
            assert (
                not state.waiting or negotiator.id == state.current_proposer
            ), f"We are waiting with {state.current_proposer} as the last offerer but we are asking {negotiator.id} to offer\n{state}"
            rem = self.remaining_time
            if rem is None:
                rem = float("inf")
            timeout = min(
                self.nmi.negotiator_time_limit - times[negotiator.id],
                self.nmi.step_time_limit,
                rem,
                self._hidden_time_limit - self.time,
            )
            if timeout is None or timeout == float("inf") or self._sync_calls:
                __strt = time.perf_counter()
                try:
                    if (
                        negotiator == self._current_proposer
                    ) and self._offering_is_accepting:
                        state.n_acceptances = 0
                        response = negotiator.counter(*args, **kwargs)
                    else:
                        response = negotiator.counter(*args, **kwargs)
                except TimeoutError:
                    response = None
                    try:
                        negotiator.cancel()
                    except:
                        pass
                except Exception as ex:
                    exceptions[negotiator.id].append(exception2str())
                    if self.ignore_negotiator_exceptions:
                        self.announce(
                            Event(
                                "negotiator_exception",
                                {"negotiator": negotiator, "exception": ex},
                            )
                        )
                        times[negotiator.id] += time.perf_counter() - __strt
                        return SAOResponse(ResponseType.END_NEGOTIATION, None), True
                    else:
                        raise ex
                times[negotiator.id] += time.perf_counter() - __strt
            else:
                fun = functools.partial(negotiator.counter, *args, **kwargs)
                __strt = time.perf_counter()
                try:
                    if (
                        negotiator == self._current_proposer
                    ) and self._offering_is_accepting:
                        state.n_acceptances = 0
                        response = TimeoutCaller.run(fun, timeout=timeout)
                    else:
                        response = TimeoutCaller.run(fun, timeout=timeout)
                except TimeoutError:
                    response = None
                except Exception as ex:
                    exceptions[negotiator.id].append(exception2str())
                    if self.ignore_negotiator_exceptions:
                        self.announce(
                            Event(
                                "negotiator_exception",
                                {"negotiator": negotiator, "exception": ex},
                            )
                        )
                        times[negotiator.id] += time.perf_counter() - __strt
                        return SAOResponse(ResponseType.END_NEGOTIATION, None), True
                    else:
                        raise ex
                times[negotiator.id] += time.perf_counter() - __strt
            if (
                self.check_offers
                and response is not None
                and response.outcome is not None
            ):
                if not self.outcome_space.is_valid(response.outcome):
                    return SAOResponse(response.response, None), False
                # todo: do not use .issues here as they are not guaranteed to exist (if it is not a cartesial outcome space)
                if self._enforce_issue_types and hasattr(self.outcome_space, "issues"):
                    if outcome_types_are_ok(
                        response.outcome, self.outcome_space.issues  # type: ignore
                    ):
                        return response, False
                    elif self._cast_offers:
                        return (
                            SAOResponse(
                                response.response,
                                cast_value_types(
                                    response.outcome, self.outcome_space.issues  # type: ignore
                                ),
                            ),
                            False,
                        )
                    return SAOResponse(response.response, None), False
            return response, False

        proposers, proposer_indices = [], []
        for i, neg in enumerate(negotiators):
            if not neg.capabilities.get("propose", False):
                continue
            proposers.append(neg)
            proposer_indices.append(i)
        n_proposers = len(proposers)
        if n_proposers < 1:
            if not self.dynamic_entry:
                return MechanismRoundResult(
                    broken=True,
                    timedout=False,
                    agreement=None,
                    error=True,
                    error_details="No proposers and no dynamic entry",
                    times=times,
                    exceptions=exceptions,
                )
            else:
                return MechanismRoundResult(
                    broken=False,
                    timedout=False,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                )
        # if this is the first step (or no one has offered yet) which means that there is no _current_offer
        if (
            state.current_offer is None
            and n_proposers > 1
            and self._avoid_ultimatum
            and not self._ultimatum_avoided
        ):
            if not self.dynamic_entry and not self.state.step == 0:
                if self.end_negotiation_on_refusal_to_propose:
                    return MechanismRoundResult(
                        broken=True,
                        times=times,
                        exceptions=exceptions,
                    )
            # if we are trying to avoid an ultimatum, we take an offer from everyone and ignore them but one.
            # this way, the agent cannot know its order. For example, if we have two agents and 3 steps, this will
            # be the situation after each step:
            #
            # Case 1: Assume that it ignored the offer from agent 1
            # Step, Agent 0 calls received  , Agent 1 calls received    , relative time during last call
            # 0   , counter(None)->offer1*  , counter(None) -> offer0   , 0/3
            # 1   , counter(offer2)->offer3 , counter(offer1) -> offer2 , 1/3
            # 2   , counter(offer4)->offer5 , counter(offer3) -> offer4 , 2/3
            # 3   ,                         , counter(offer5)->offer6   , 3/3
            #
            # Case 2: Assume that it ignored the offer from agent 0
            # Step, Agent 0 calls received  , Agent 1 calls received    , relative time during last call
            # 0   , counter(None)->offer1   , counter(None) -> offer0*  , 0/3
            # 1   , counter(offer0)->offer2 , counter(offer2) -> offer3 , 1/3
            # 2   , counter(offer3)->offer4 , counter(offer4) -> offer5 , 2/3
            # 3   , counter(offer5)->offer6 ,                           , 3/3
            #
            # in both cases, the agent cannot know whether its last offer going to be passed to the other agent
            # (the ultimatum scenario) or not.
            responses = []
            responders = []
            for i, neg in enumerate(proposers):
                if not neg.capabilities.get("propose", False):
                    continue
                strt = time.perf_counter()
                resp, has_exceptions = _safe_counter(neg, state=self.state, offer=None)
                if has_exceptions:
                    return MechanismRoundResult(
                        broken=True,
                        timedout=False,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                        error=True,
                        error_details=str(exceptions[neg.id]),
                    )
                if resp is None:
                    return MechanismRoundResult(
                        broken=False,
                        timedout=True,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                        error=False,
                        error_details="",
                    )
                if resp.response != ResponseType.WAIT:
                    self._waiting_time[neg.id] = 0.0
                    self._waiting_start[neg.id] = float("inf")
                    self._frozen_neg_list = None
                else:
                    self._waiting_start[neg.id] = min(self._waiting_start[neg.id], strt)
                    self._waiting_time[neg.id] += (
                        time.perf_counter() - self._waiting_start[neg.id]
                    )
                if resp is None:
                    return MechanismRoundResult(
                        broken=False,
                        timedout=True,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                    )
                if time.perf_counter() - strt > self.nmi.step_time_limit:
                    return MechanismRoundResult(
                        broken=False,
                        timedout=True,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                    )
                if resp.response == ResponseType.END_NEGOTIATION:
                    return MechanismRoundResult(
                        broken=True,
                        timedout=False,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                    )
                if resp.response in (ResponseType.NO_RESPONSE, ResponseType.WAIT):
                    continue
                if (
                    resp.response == ResponseType.REJECT_OFFER
                    and resp.outcome is None
                    and self.end_negotiation_on_refusal_to_propose
                ):
                    continue
                responses.append(resp)
                responders.append(i)
            if len(responses) < 1:
                if not self.dynamic_entry:
                    return MechanismRoundResult(
                        broken=True,
                        timedout=False,
                        agreement=None,
                        error=True,
                        error_details="No proposers and no dynamic entry. This may happen if no negotiators responded to their first proposal request with an offer",
                        times=times,
                        exceptions=exceptions,
                    )
                else:
                    return MechanismRoundResult(
                        broken=False,
                        timedout=False,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                    )
            # choose a random negotiator and set it as the current negotiator
            self._ultimatum_avoided = True
            selected = random.randint(0, len(responses) - 1)
            resp = responses[selected]
            neg = proposers[responders[selected]]
            _first_proposer = proposer_indices[responders[selected]]
            self._selected_first = _first_proposer
            self._last_checked_negotiator = _first_proposer
            state.current_offer = resp.outcome
            state.new_offers.append((neg.id, resp.outcome))
            self._current_proposer = neg
            state.current_proposer = neg.id
            state.n_acceptances = 1 if self._offering_is_accepting else 0
            if self._last_checked_negotiator >= 0:
                state.last_negotiator = self.negotiators[
                    self._last_checked_negotiator
                ].name
            else:
                state.last_negotiator = ""
            (
                self._current_proposer_agent,
                state.new_offerer_agents,
            ) = self._agent_info()

            # current_proposer_agent=current_proposer_agent,
            # new_offerer_agents=new_offerer_agents,
            return MechanismRoundResult(
                broken=False,
                timedout=False,
                agreement=None,
                times=times,
                exceptions=exceptions,
            )

        # this is not the first round. A round will get n_negotiators responses
        if self._frozen_neg_list is not None:
            ordered_indices = self._frozen_neg_list
        else:
            ordered_indices = [
                (_ + self._last_checked_negotiator + 1) % n_negotiators
                for _ in range(n_negotiators)
            ]

        for _, neg_indx in enumerate(ordered_indices):
            self._last_checked_negotiator = neg_indx
            neg = self.negotiators[neg_indx]
            strt = time.perf_counter()
            resp, has_exceptions = _safe_counter(
                neg, state=self.state, offer=state.current_offer
            )
            if has_exceptions:
                return MechanismRoundResult(
                    broken=True,
                    timedout=False,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                    error=True,
                    error_details=str(exceptions[neg.id]),
                )
            if resp is None:
                return MechanismRoundResult(
                    broken=False,
                    timedout=True,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                    error=False,
                    error_details="",
                )
            if resp.response == ResponseType.WAIT:
                self._waiting_start[neg.id] = min(self._waiting_start[neg.id], strt)
                self._waiting_time[neg.id] += time.perf_counter() - strt
                self._last_checked_negotiator = (neg_indx - 1) % n_negotiators
                offered = {self._negotiator_index[_[0]] for _ in state.new_offers}
                did_not_offer = sorted(
                    list(set(range(n_negotiators)).difference(offered))
                )
                assert neg_indx in did_not_offer
                indx = did_not_offer.index(neg_indx)
                assert (
                    self._frozen_neg_list is None
                    or self._frozen_neg_list[0] == neg_indx
                )
                self._frozen_neg_list = did_not_offer[indx:] + did_not_offer[:indx]
                self._n_waits += 1
            else:
                self._stop_waiting(neg.id)

            if resp is None:
                return MechanismRoundResult(
                    broken=False,
                    timedout=True,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                )
            if time.perf_counter() - strt > self.nmi.step_time_limit:
                return MechanismRoundResult(
                    broken=False,
                    timedout=True,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                )
            if self._extra_callbacks:
                if state.current_offer is not None:
                    for other in self.negotiators:
                        if other is not neg:
                            other.on_partner_response(
                                state=self.state,
                                partner_id=neg.id,
                                outcome=state.current_offer,
                                response=resp.response,
                            )
            if resp.response == ResponseType.NO_RESPONSE:
                continue
            if resp.response == ResponseType.WAIT:
                if self._n_waits > self._n_max_waits:
                    self._stop_waiting(neg.id)
                    return MechanismRoundResult(
                        broken=False,
                        timedout=True,
                        agreement=None,
                        waiting=False,
                        times=times,
                        exceptions=exceptions,
                    )
                return MechanismRoundResult(
                    broken=False,
                    timedout=False,
                    agreement=None,
                    waiting=True,
                    times=times,
                    exceptions=exceptions,
                )
            if resp.response == ResponseType.END_NEGOTIATION:
                return MechanismRoundResult(
                    broken=True,
                    timedout=False,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                )
            if resp.response == ResponseType.ACCEPT_OFFER:
                state.n_acceptances += 1
                if state.n_acceptances == n_negotiators:
                    return MechanismRoundResult(
                        broken=False,
                        timedout=False,
                        agreement=state.current_offer,
                        times=times,
                        exceptions=exceptions,
                    )
            if resp.response == ResponseType.REJECT_OFFER:
                proposal = resp.outcome
                if (
                    not self.allow_offering_just_rejected_outcome
                    and proposal == state.current_offer
                ):
                    proposal = None
                if proposal is None:
                    if (
                        neg.capabilities.get("propose", True)
                        and self.end_negotiation_on_refusal_to_propose
                    ):
                        return MechanismRoundResult(
                            broken=True,
                            timedout=False,
                            agreement=None,
                            times=times,
                            exceptions=exceptions,
                        )
                    state.n_acceptances = 0
                else:
                    state.n_acceptances = 1 if self._offering_is_accepting else 0
                    if self._extra_callbacks:
                        for other in self.negotiators:
                            if other is neg:
                                continue
                            other.on_partner_proposal(
                                partner_id=neg.id, offer=proposal, state=self.state
                            )
                state.current_offer = proposal
                self._current_proposer = neg
                state.current_proposer = neg.id
                state.new_offers.append((neg.id, proposal))
                if self._last_checked_negotiator >= 0:
                    state.last_negotiator = self.negotiators[
                        self._last_checked_negotiator
                    ].name
                else:
                    state.last_negotiator = ""
                (
                    self._current_proposer_agent,
                    state.new_offerer_agents,
                ) = self._agent_info()

        return MechanismRoundResult(
            broken=False,
            timedout=False,
            agreement=None,
            times=times,
            exceptions=exceptions,
        )

"""
A = [1, 2, 3]
B = [4, 5, 6]
C = [7, 8, 9, 10]
print (list(zip(A,B,C)))
# --> [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
A = [1, 2, 3]
B = [4, 5, 6]
C = [7, 8]
print (list(zip(A,B,C)))
# --> [(1, 4, 7), (2, 5, 8)]
"""

#----
# create negotiation agenda (issues)

# create the mechanism

# define buyer and seller utilities


result1,result2,result3=[],[],[]
issues=[]

# define 'number_of_issues' as static variables
# try to use comprehensive variable names, x -> number_issues
number_of_issues = 5 # --> 10
number_rounds = 10
number_values_per_issue = 5

def create_profile(num_issues):
    return {uniform(0, 1) : randint(1, 5) for _ in range(num_issues)}

# generate the full profiles of the agents (parameters)
profile1 = create_profile(number_of_issues)
profile2 = create_profile(number_of_issues)

for issue in range(number_of_issues): # number of issues, 'number_of_issues' issues, her
    issues.append(make_issue(name="issue"+str(issue), values=number_values_per_issue))

    print ("____________________________________________________________________________________")

    print ("Issues, under negotiation: ", issues)

    # TODO two exploential functions with different tau,

    values_u1 = list(map(LinearFun, profile1.keys()))
    values_w1 = list(profile1.values())

    session = SAOMechanism(issues=issues, time_limit=number_rounds) # deadline
    # session = SAOMechanism(issues=issues, time_limit=number_rounds) # deadline
    seller_utility = LUFun(
        values=values_u1[:issue+1],
        weights=values_w1[:issue+1],
        outcome_space=session.outcome_space,
    )

    values_u2 = list(map(LinearFun, profile2.keys()))
    values_w2 = list(profile2.values())

    buyer_utility = LUFun(
        values=values_u2[:issue+1],
        weights=values_w2[:issue+1],
        outcome_space = session.outcome_space,
    )
    seller_utility = seller_utility.scale_max(1.0) # normalization
    buyer_utility  = buyer_utility.scale_max(1.0)

    # create and add buyer and sell
    # session.add(BetterRandomNegotiator(name="buyer"), preferences=buyer_utility)
    # session.add(TimeBasedConcedingNegotiator(name="seller"), ufun=seller_utility)

    session.add(BetterRandomNegotiator(name="buyer"), preferences=buyer_utility)
    session.add(BetterRandomNegotiator(name="seller"), preferences=seller_utility)

    # Agreements
    result = session.run()
    print ("u_buyer(bid = {}) = {}".format(result["current_offer"], buyer_utility(result["current_offer"])) )
    print ("u_seller(bid = {}) = {}".format(result["current_offer"], seller_utility(result["current_offer"])) )

    if buyer_utility(result["current_offer"])==float("-inf"):
        result3.append([issue + 1, 0])
    else:
        result3.append([issue + 1,buyer_utility(result["current_offer"])])
    result1.append([issue+1,result["time"]])

    if seller_utility(result["current_offer"])==float("-inf"):
        result2.append([issue + 1, 0])
    else:
        result2.append([issue+1, seller_utility(result["current_offer"])])


# run the negotiation and show the results
#print(result3)
x1=[]
y1=[]
x2=[]
y2=[]
y4=[]
x3=[]
y3=[]
x4=[]
for r1 in result1:
    x1.append(r1[0])
    y1.append(r1[1])

for r2 in result2:
    x2.append(r2[0])
    y2.append(r2[1])

for r3 in result3:
    x3.append(r3[0])
    y3.append(r3[1])

for r in range(len(result2)):
    x4.append(result2[r][0])
    y4.append(result2[r][1]+result3[r][1])
#plt.plot(x1,y1,label="time/issues",marker="o")
plt.plot(x2,y2,label="Agreement utility of agent 1",marker="x")
plt.plot(x3,y3,label="Agreement utility of agent 2",marker="o")
plt.plot(x4,y4,label="Social Welfare",marker="o")

plt.xlabel("Number of issues")
plt.ylabel("Utilities")
#plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
#print(result1,result2)
plt.legend()
#print(session.run())
#session.plot(show_reserved=False)
plt.grid(color='gray', linestyle='dashed', linewidth=0.1)
plt.show()
