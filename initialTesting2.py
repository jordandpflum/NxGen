import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from matplotlib import pyplot as plt

from scipy.integrate import odeint
from scipy.integrate import ode


# Rossler parameters: Paper
a = 0.2
b = 0.2
c = 5.7

# Rössler system
def rossler(X,t):
    x, y, z = X
    dx = -y - z
    dy = x + a*y
    dz = b*x - c*z + x*z
    return [dx, dy, dz]

# Numerical integration

# Initial condition (random, just intersting starting points)
X0 = [0.1, 0.1, 0.1]
X0 = [2, 2, 2]

# Set Time Parameters
# Paper Parameters
dt = 0.05
points = 4096
points=10000
tspan = np.arange(start=dt, stop=dt*points, step=dt)
print(tspan)


time = np.arange(0, 300, 0.01)
#result = odeint(rossler, X0, time)
r = ode().set_integrator('dopri5')

x, y, z = result.T
print(x)

# figure
fig = plt.figure(figsize=(13,9))
ax = fig.gca(projection='3d')
ax.set_ylim(-15, 10)
ax.set_xlim(-10, 15)
ax.set_zlim(0, 25)
ax.view_init(20, 160)
ax.plot(x,y,z,'blue')
plt.show()