import copy
from randomeone import BetterRandomNegotiator
from negmas import (
    make_issue,
    SAOMechanism,
    NaiveTitForTatNegotiator,
    TimeBasedConcedingNegotiator,
    AspirationNegotiator,
)
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun

# create negotiation agenda (issues)


# create the mechanism

# define buyer and seller utilities




result1,result2,result3=[],[],[]
issues=[]

# define 10 as static variables
# try to use comprehensive variable names, x -> number_issues
for x in range(10):
    issues.append(make_issue(name="issue"+str(x), values=5))
    session = SAOMechanism(issues=issues,time_limit=10)
    seller_utility = LUFun(
            values=[IdentityFun(), LinearFun(0.15), LinearFun(0.1),LinearFun(0.2),LinearFun(0.2),LinearFun(0.15),LinearFun(0.15),LinearFun(0.13),LinearFun(0.2),LinearFun(0.2),
                # AffineFun(-1, bias=9.0)
         LinearFun(0.2)
        , LinearFun(0.2)
        , LinearFun(0.2)
        , LinearFun(0.2)
        , LinearFun(0.2)
                ],
        weights=[2, 4, 1, 1, 1, 3, 1, 1, 1, 1, 0.5, 1],
        outcome_space=session.outcome_space,
    )

    buyer_utility = LUFun(
        values=[LinearFun(0.19),LinearFun(0.19),LinearFun(0.19),LinearFun(0.19),LinearFun(0.19),LinearFun(0.19),LinearFun(0.19),LinearFun(0.19),LinearFun(0.19),LinearFun(0.19),
                LinearFun(0.19),
                LinearFun(0.19),
                LinearFun(0.19),
                LinearFun(0.19),
                LinearFun(0.19)],
        weights=[1,1,2,1,2.5,1,3,1,1,2,1,1],
        outcome_space = session.outcome_space,
    )
    seller_utility = seller_utility.scale_max(3.0)
    buyer_utility = buyer_utility.scale_max(3.0)

    # create and add buyer and sell
    #session.add(BetterRandomNegotiator(name="buyer2"), preferences=buyer_utility)
    session.add(BetterRandomNegotiator(name="buyer"), preferences=buyer_utility)
    session.add(TimeBasedConcedingNegotiator(name="seller"), ufun=seller_utility)
    rr=session.run()
    ll=[0,0,0]
    if rr["agreement"]:
        ll=list(rr["agreement"])
    #b=LinearFun(0.2)
    #print(buyer_utility(rr["current_offer"]))
    if buyer_utility(rr["current_offer"])==float("-inf"):
        result3.append([x + 1, 0])
    else:
        result3.append([x+1,buyer_utility(rr["current_offer"])])
    result1.append([x+1,rr["time"]])
    if seller_utility(rr["current_offer"])==float("-inf"):
        result2.append([x + 1, 0])
    else:
        result2.append([x+1, seller_utility(rr["current_offer"])])
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
plt.plot(x4,y4,label="social welfare/issues",marker="o")
plt.plot(x2,y2,label="utility of agent 1/issues",marker="x")
plt.plot(x3,y3,label="utility of agent 2/issues",marker="o")
plt.xlabel("number of issues")
plt.ylabel("utilities")
#plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f s'))
#print(result1,result2)
plt.legend()
#print(session.run())
#session.plot(show_reserved=False)
plt.show()