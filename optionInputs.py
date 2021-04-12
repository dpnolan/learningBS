import numpy as np
import pandas as pd
from math import log, sqrt, exp

import scipy as scipy
from scipy import stats

'''
Black-Scholes vanilla European call price function, with the default values as shown in the example in Hull (2009)
'''

def bs(S=42,K=40,R=0.1,Sigma=0.2,T=0.5):
    if ( np.any(S)==0 ):
        return 0
    elif (np.any(K)==0):
        return S
    else:
        d1 = (np.log(S/K)  +   (R+(np.square(Sigma)/2) )* T )  / (Sigma*np.sqrt(T))
        d2 = d1 - (Sigma * np.sqrt(T) )
        c = S*stats.norm.cdf(d1, 0.0, 1.0)-  K*np.exp(-R*T)*stats.norm.cdf(d2, 0.0, 1.0)
        return c

# size constants
NUMBER_S = 10
INCREMENT_S = 1
optionS = np.array(np.arange(35, 45, INCREMENT_S))

NUMBER_K = 10
INCREMENT_K = 1
optionK = np.array(np.arange(35, 45, INCREMENT_K))

NUMBER_SIGMA = 10
INCREMENT_SIGMA = 0.01
optionSIGMA = np.array(np.arange(0.05, 0.15, INCREMENT_SIGMA))

BUSINESS_DAYS = 252
NUMBER_T = BUSINESS_DAYS
INCREMENT_T = -int(BUSINESS_DAYS / 12)
optionT = (np.arange(-INCREMENT_T, NUMBER_T - INCREMENT_T, -INCREMENT_T)) / BUSINESS_DAYS

NUMBER_R = 5
INCREMENT_R = 0.01
optionR = np.array(np.arange(0, (NUMBER_R / 100) + 0.01, INCREMENT_R))

NUMBER_OPTIONS = NUMBER_S * NUMBER_K * NUMBER_SIGMA * int(-NUMBER_T / INCREMENT_T) * (NUMBER_R + 1)
print("NUMBER_OPTIONS=",NUMBER_OPTIONS)
print(optionK.size)

#s = np.zeros(NUMBER_OPTIONS)
#k = np.zeros(NUMBER_OPTIONS)
#r = np.zeros(NUMBER_OPTIONS)
#sigma = np.zeros(NUMBER_OPTIONS)
#t = np.zeros(NUMBER_OPTIONS)
#v = np.zeros(NUMBER_OPTIONS)

t=np.tile(optionT, NUMBER_OPTIONS // optionT.size)

sigma=np.tile(np.repeat(optionSIGMA, optionT.size),NUMBER_OPTIONS//(optionT.size*optionSIGMA.size))

r=np.tile(np.repeat(optionR, optionT.size*optionSIGMA.size),NUMBER_OPTIONS//(optionT.size*optionSIGMA.size*optionR.size))

k=np.tile(np.repeat(optionK, optionT.size*optionSIGMA.size*optionR.size), \
          NUMBER_OPTIONS//(optionT.size*optionSIGMA.size*optionR.size*optionK.size))

s=np.tile(np.repeat(optionS, optionT.size*optionSIGMA.size*optionR.size*optionK.size), \
          NUMBER_OPTIONS//(optionT.size*optionSIGMA.size*optionR.size*optionK.size*optionS.size))

v=bs(s,k,r,sigma,t)

print(bs())

optionDF = pd.DataFrame(data={"S": s, "K": k, "R": r, "Sigma": sigma, "T": t, "Value": v},
                        index=np.arange(0, NUMBER_OPTIONS, 1))

'''
export csv

with open('optionInputs.csv', 'w') as fout:
    csvout = csv.writer(fout, delimiter='\n')
    csvout.writerows([r])

fout.close
'''

f = open('optionDF.csv', 'w+')
optionDF.to_csv('optionDF.csv')
f.close()

