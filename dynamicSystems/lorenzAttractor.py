import numpy as np
from general.generateSystem import generate_system

def lorenz_odes(Y, parameters):
    """
    Models the lorenz attractor

    Parameters:
    Y: a numpy array, representing the current state of the dynamic system Y = [x, y, z]
    parameters: a numpy array, representing the parameters of the dynamic system
    dt: a float, representing the step size to take between points

    Returns: a numpy array, representing dY, a system of three non-linear ordinary differential equations modeling
    the Lorenz attractor
    """
    # State
    x, y, z = Y

    # Parameters
    sigma = 10
    b = 8 / 3
    r = 28
    sigma, b, c = parameters

    # Returns dY
    return np.array([sigma * (y - x),
                     (r * x) - y - (x * z),
                     (x * y) - (b * z)
                     ]
                    )


def lorenz_generate(num_points, initial_state, parameters, dt):
    """
    Generate the points of the given ordinary system of equations (ODES) using Runge-Kutta 4th order method
    given an initial starting point (state), parameters for the odes, step size (dt) and number of points to generate.

    Parameters:
    num_points: an integer, representing the number of points to generate for the given dynamic system
    initial_state: a numpy array, representing the starting state of the dynamic system
    parameters: a numpy array, representing the parameters of the dynamic system
    dt: a float, representing the step size to take between points

    Returns: generateSystem, a user function, which generate the points of the given ordinary system of equations
    (ODES) using Runge-Kutta 4th order method given an initial starting point (state), parameters for the odes,
    step size (dt) and number of points to generate.
    """
    return generate_system(num_points=num_points, odes=lorenz_odes, initial_state=initial_state,
                           parameters=parameters, dt=dt)