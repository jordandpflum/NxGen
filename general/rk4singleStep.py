


def rk4_singleStep(odes, state, parameters, dt):
    """
    Numerically integrates a single step of a dynamic system using a fourth order Runge-Kutta algorithm

    Parameters:
    odes: a user function, representing the dynamic system.
    state: a numpy array, representing the current state of the dynamic system
    parameters: a numpy array, representing the parameters of the dynamic system
    dt: a float, representing the step size to take between points

    Returns: a numpy array, representing the next state of the dynamic system
    """
    k1 = dt * odes(state, parameters)
    k2 = dt * odes(state + 0.5 * k1, parameters)
    k3 = dt * odes(state + 0.5 * k2, parameters)
    k4 = dt * odes(state + k3, parameters)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
