import numpy as np
from general.mutualInformation.mutual_info import mutual_information


def mutualInformationWTS(signalData,maxTimeDelay):
    """
    Calculates the Mutual Information of signalData with respect to signalData with a time delay up to maxTimeDelay.

    Creates a matrix containing the MaxEmDim trajectories generated for a specified time delay and a given set of data
    (sigalData).

    Parameters:
    signalData: a numpy array (nx1), representing the signal data to be analyzed.
    maxPosEmDim: an integer, representing the maximum possible embedding dimension for the number of points in
                 signalData.

    Returns: MI, a numpy array (maxTimeDelay,2), representing the solved MI
    """
    len_signalData = len(signalData)

    # Crete Storage for mutual information
    MI = np.zeros((maxTimeDelay, 2))

    # Calculate and store mutual information for every timeDelay value
    for timeDelay in range(maxTimeDelay):
        shifted_data = signalData[timeDelay:len_signalData]

        # Create Vectors to Calculate Mutual Information
        vec1 = np.array(signalData[0:len_signalData-timeDelay])
        vec2 = np.array(shifted_data)

        # Reshape vectors for input into func:mutual_information
        vec1 = vec1.reshape(-1, 1)
        vec2 = vec2.reshape(-1, 1)

        # Calculate and store mutual information for given timeDelay
        MI[timeDelay,0] = timeDelay
        MI[timeDelay, 1] = mutual_information((vec1, vec2))

    return MI


'''
# Does not work
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing

def mutualInformationWTS2(signalData,maxTimeDelay):
    """
    Calculates the Mutual Information of signalData with respect to signalData with a time delay up to maxTimeDelay.

    Creates a matrix containing the MaxEmDim trajectories generated for a specified time delay and a given set of data
    (sigalData).

    Parameters:
    signalData: a numpy array (nx1), representing the signal data to be analyzed.
    maxPosEmDim: an integer, representing the maximum possible embedding dimension for the number of points in
                 signalData.

    Returns: MI, a numpy array (maxTimeDelay,2), representing the solved MI
    """
    len_signalData = len(signalData)

    # Crete Storage for mutual information
    MI = np.zeros((maxTimeDelay, 2))

    # Calculate and store mutual information for every timeDelay value
    for timeDelay in range(maxTimeDelay):
        shifted_data = signalData[timeDelay:len_signalData]

        # Create Vectors to Calculate Mutual Information
        vec1 = np.array(signalData[0:len_signalData-timeDelay])
        vec2 = np.array(shifted_data)

        # Reshape vectors for input into func:mutual_information

        lab_enc = preprocessing.LabelEncoder()
        vec2 = lab_enc.fit_transform(vec2)
        vec1 = vec1.reshape(-1, 1)
        #vec2 = vec2.reshape(-1, 1)

        # Calculate and store mutual information for given timeDelay
        MI[timeDelay,0] = timeDelay
        mutual_information_bits = mutual_info_classif(X=vec1, y=vec2, discrete_features=True)
        MI[timeDelay, 1] = mutual_information_bits

    return MI
    
'''


