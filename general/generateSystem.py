from general.rk4singleStep import rk4_singleStep
import numpy as np


def generate_system(num_points, odes, initial_state, parameters, dt):
    """
    Generate the points of the given ordinary system of equations (ODES) using Runge-Kutta 4th order method
    given an intial starting point (state), parameters for the odes, step size (dt) and number of points to generate.

    Parameters:
    num_points: an integer, representing the number of points to generate for the given dynamic system
    odes: a user function, representing the dynamic system.
    initial_state: a numpy array, representing the starting state of the dynamic system
    parameters: a numpy array, representing the parameters of the dynamic system
    dt: a float, representing the step size to take between points

    Returns: results, a numpy array, representing the solved points of the ODES
    """
    # Create matrix to store results [n_states x data_length)
    results = np.zeros([num_points, initial_state.shape[0]])

    # Store Initial State
    results[0, :] = initial_state

    # Initialize state as initial_state
    state = initial_state

    # Calculate Remaining States
    for point in range(num_points - 1):
        # Calculate next State from current State using Runge-Kutta 4th order method
        state = rk4_singleStep(odes, state, parameters, dt)

        # Append state to results
        results[point + 1, :] = state

    return results
