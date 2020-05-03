import math
import numpy as np


def trajectory(signalData, maxEmDim, timeDelay):
    """
    reproduce unknown dynamic system from signal data by creates a matrix containing the MaxEmDim trajectories
    generated for a specified time delay.

    Parameters:
    signalData: a numpy array (nx1), representing the signal data to be analyzed
    maxEmDim: a float, representing the maximum embedding dimension for which the trajectory (portrait)is to be
              constructed
    timeDelay: an integer, representing the time delay to apply to the signal data

    Returns: portrait: matrix in which each row is a point in the reconstructed trajectory. Each point in the row is
                       the corrdinate of that point.
             noPoints: Number of points for each dimension. For any dimension EmDim, NoPoints=length(SigData)-(EmDim-1)
                       *TimeDelay
             maxPosEmDim: Maximum possible embedding dimension for the number of points in SigData
    """
    len_data = len(signalData)
    maxPosEmDim = math.floor(2 * np.log10(len_data))

    noPoints = []
    for i in range(maxEmDim):
        noPoints.append(len_data - (i * timeDelay))

    portrait = np.zeros((noPoints[0], maxEmDim))

    for i in range(maxEmDim):
        portrait[0:len_data - (i * timeDelay), i] = signalData[(i * timeDelay): len_data]

    return portrait, noPoints, maxPosEmDim