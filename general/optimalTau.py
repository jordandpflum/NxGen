import numpy as np
from scipy.signal import argrelextrema

from general.mutualInformation.mutualInformationWTS import mutualInformationWTS

def generateOptimalTau(signalData, dimension, method, maxTimeDelay=100):
    # Calculate Mutual Information for every time delay
    signalData = mutualInformationWTS(signalData, maxTimeDelay)
    #signalData = signalData[0:dimension, 1]

    # Calculate Optimal Tau
    if method=="Minimizing MI over first (d-1) Tau":
        optimalTau = int(round(sum(signalData[0:(dimension-1), 1])))
    elif method=="First Local Minimum":
        localMinimums = argrelextrema(signalData[:, 1], np.greater)
        optimalTau = localMinimums[0][0]

    return optimalTau





