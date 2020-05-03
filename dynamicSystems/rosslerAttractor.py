from general.generateSystem import generate_system
import numpy as np


def rossler_odes(Y, parameters):
    """
    Models the rossler attractor

    Parameters:
    Y: a numpy array, representing the current state of the dynamic system Y = [x, y, z]
    parameters: a numpy array, representing the parameters of the dynamic system
    dt: a float, representing the step size to take between points

    Returns: a numpy array, representing dY, a system of three non-linear ordinary differential equations modeling
    the Rossler attractor
    """
    # State
    x, y, z = Y

    # Parameters
    a, b, c = parameters

    # Returns dY
    return np.array([-y - z,
                     x + a * y,
                     b + z * (x - c)
                     ]
                    )


def rossler_generate(num_points, initial_state, parameters, dt):
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
    return generate_system(num_points=num_points, odes=rossler_odes, initial_state=initial_state,
                           parameters=parameters, dt=dt)
