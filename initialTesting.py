import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

import matplotlib.pyplot as plt


def generateSystem(num_points, odes, initial_state, parameters, dt):
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
    results = np.zeros([initial_state.shape[0], num_points])

    # Store Initial State
    results[:, 0] = initial_state

    # Initalize state as initial_state
    state = initial_state

    # Calculate Remaining States
    for point in range(num_points-1):
        # Calculate next State from current State using Runge-Kutta 4th order method
        state = rk4_singleStep(odes, state, parameters, dt)

        # Append state to results
        results[:, point+1] = state

    return results



def rk4_singleStep(odes, state, parameters, dt):
    """
    Numeriacally integrates a single step of a dynamic system using a fourth order Runge-Kutta algorithm

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
    given an intial starting point (state), parameters for the odes, step size (dt) and number of points to generate.

    Parameters:
    num_points: an integer, representing the number of points to generate for the given dynamic system
    initial_state: a numpy array, representing the starting state of the dynamic system
    parameters: a numpy array, representing the parameters of the dynamic system
    dt: a float, representing the step size to take between points

    Returns: generateSystem, a user function, which generate the points of the given ordinary system of equations
    (ODES) using Runge-Kutta 4th order method given an intial starting point (state), parameters for the odes,
    step size (dt) and number of points to generate.
    """
    return generateSystem(num_points=num_points, odes=rossler_odes, initial_state=initial_state,
                          parameters=parameters, dt=dt)

# Rossler parameters: Paper
a = 0.2
b = 0.2
c = 5.7

# Initial condition (random, just intersting starting points)
initial_condition = np.array([0, 0, 0])

# Set Parameters
# Paper Parameters
dt = 0.05
num_points = 4096
# tspan = np.arange(start=dt, stop=dt*num_points, step=dt)

data = rossler_generate(num_points=num_points, initial_state=initial_condition, parameters=np.array([a, b, c]), dt=dt)


fig = plt.figure(figsize=(13,9))
ax = fig.gca(projection='3d')
ax.set_ylim(-15, 10)
ax.set_xlim(-10, 15)
ax.set_zlim(0, 25)
#ax.view_init(20, 160)
ax.plot(data[0],data[1],data[2],'blue')
ax.set_xlabel('X(t)', rotation=0)
ax.set_ylabel('Y(t)', rotation=0)
ax.set_zlabel('Z(t)')
ax.set_title('The Rossler Attractor', loc='center')
plt.show()



def lorenz_odes(x, y, z, sigma, beta, rho):
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def lorenz_generate(data_length):
    return generateSystem(data_length, lorenz_odes, np.array([-8.0, 8.0, 27.0]), np.array([10.0, 8/3.0, 28.0]))








