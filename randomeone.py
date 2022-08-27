from negmas import SAONegotiator,ResponseType
from random import randint
from negmas.preferences.value_fun import LinearFun
class RandomNegotiator(SAONegotiator):
    def propose(self, state):
        if self.nmi.partial_agreement==0:
            divide_point=len(self.ufun.issues)//2
            randig=randint(0,len(self.ufun.issues)-1)
            self.nmi.partial_digits.append(randig)
            rdm=self.nmi.random_outcomes(1)[0]
            rdmlist=list(rdm)
            for x in range(divide_point):
                rdmlist[x]=9527
            #rdmlist[randig]=9527
            return tuple(rdmlist)
        else:
            rdm = self.nmi.random_outcomes(1)[0]
            return rdm
class BetterRandomNegotiator(RandomNegotiator):
    def respond(self, state, offer):
        if 9527 in offer or self.nmi.partial_offer!=[]:
            if self.ufun(offer) > 0.99:
                return ResponseType.PARTIAL_AGREEMENT
            return ResponseType.REJECT_OFFER
        if self.ufun(offer) > 0.99:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

class BestRandomNegotiator(BetterRandomNegotiator):
    def propose(self, state):
        if self.nmi.partial_agreement==0:
            randig=randint(0,len(self.ufun.issues)-1)
            randig=2
            self.nmi.partial_digits.append(randig)
            rdm=self.nmi.random_outcomes(1)[0]
            rdmlist=list(rdm)
            rdmlist[randig]=9527
            print(rdmlist)
            return tuple(rdmlist)
        else:
            rdm = self.nmi.random_outcomes(1)[0]

            return rdm
"""
from random import randint, uniform
number_of_issues = 5 
def create_profile(num_issues):
    return {uniform(0, 1) : randint(1, 5) for _ in range(num_issues)}
profile1 = create_profile(number_of_issues)
values_u1 = list(map(LinearFun, profile1.keys()))
seller_utility = LUFun(
        values=values_u1[:issue+1],
        weights=values_w1[:issue+1],
        outcome_space=session.outcome_space,
    )
nego1=RandomNegotiator()
print(nego1.propose())
"""