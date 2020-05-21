import math
import numpy as np




def trajectory(signalData, maxEmDim, timeDelay):
    """
    Time delay embedding of timeseries of scalars
    Reproduce unknown dynamic system from time series signal data by creating a matrix containing the MaxEmDim
    (maximum embedding dimensions) trajectories generated for a specified time delay (tau).

    Parameters:
    signalData: a numpy array (nx1) of scalars, representing the signal data to be analyzed
    maxEmDim: a float, representing the maximum embedding dimension for which the trajectory (portrait)is to be
              constructed
    timeDelay: an integer, representing the time delay (tau) to apply to the signal data

    Returns: portrait: matrix in which each row is a point in the reconstructed trajectory. Each point in the row is
                       the corrdinate of that point.
    """
    indexes = np.arange(0, maxEmDim, 1) * timeDelay
    print(indexes)
    portrait = np.array([signalData[indexes + i] for i in range(len(signalData) - (maxEmDim - 1) * timeDelay)])
    return portrait


def trajectory2(signalData, maxEmDim, timeDelay):
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

    noPoints = np.arange(0, maxEmDim, 1) * timeDelay

    portrait = np.zeros((noPoints[0], maxEmDim))

    for i in range(maxEmDim):
        portrait[0:len_data - (i * timeDelay), i] = signalData[(i * timeDelay): len_data]

    return portrait, noPoints, maxPosEmDim