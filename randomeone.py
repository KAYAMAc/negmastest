from negmas import SAONegotiator,ResponseType
class RandomNegotiator(SAONegotiator):
    def propose(self, state):
        return self.nmi.random_outcomes(1)[0]
class BetterRandomNegotiator(RandomNegotiator):
    def respond(self, state, offer):
        print(self.ufun(offer))
        if self.ufun(offer) > 0.8:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

