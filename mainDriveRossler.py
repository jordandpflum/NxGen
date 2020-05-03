from dynamicSystems.rosslerAttractor import rossler_generate
from general.trajectory import trajectory
import numpy as np
from mpl_toolkits import mplot3d  # needed to plot 3D
import matplotlib.pyplot as plt

from general.mutualInformation.mutualInformationWTS import mutualInformationWTS

# Rossler parameters: Paper
a = 0.2
b = 0.2
c = 5.7

# Initial condition (random, just interesting starting points)
initial_condition = np.array([0, 0, 0])

# Set Time Parameters
# Paper Parameters
dt = 0.05
num_points = 4096
# tspan = np.arange(start=dt, stop=dt*num_points, step=dt)

# Generate Rossler Attractor
rossler_data = rossler_generate(num_points=num_points, initial_state=initial_condition, parameters=np.array([a, b, c]),
                                dt=dt)

'''
# Plot Rossler Attractor
fig = plt.figure(figsize=(13, 9))
ax = fig.gca(projection='3d')
ax.set_ylim(-15, 10)
ax.set_xlim(-10, 15)
ax.set_zlim(0, 25)
ax.view_init(25, -130)
ax.plot(rossler_data[0], rossler_data[1], rossler_data[2], 'blue')
ax.set_xlabel('X(t)', rotation=0)
ax.set_ylabel('Y(t)', rotation=0)
ax.set_zlabel('Z(t)', rotation=0)
ax.set_title('The Rossler Attractor', loc='center')
plt.show()

# Trajectory: Paper Replications

# Tau: 32
# Compute Trajectory
signalData = rossler_data[0, :]  # First factor (X component) of Rossler Attractor
maxEmDim = 3
timeDelay = 32
portrait, noPoints, maxPosEmDim = trajectory(signalData, maxEmDim, timeDelay)
# Plot
fig = plt.figure(figsize=(13, 9))
ax = fig.gca(projection='3d')
ax.set_xlim(-10, 15)
ax.set_ylim(-10, 15)
ax.set_zlim(-10, 15)
ax.view_init(25, -130)
ax.plot(portrait[:, 0], portrait[:, 1], portrait[:, 2], 'blue')
ax.set_xlabel('X(t)', rotation=0)
ax.set_ylabel('X(t + tau)', rotation=0)
ax.set_zlabel('X(t + 2tau)', rotation=180)
ax.set_title('Rossler Reconstructed with tau=32, steps=1.6', loc='center')
plt.show()

# Tau: 63
# Compute Trajectory
signalData = rossler_data[0, :]  # First factor (X component) of Rossler Attractor
maxEmDim = 3
timeDelay = 63
portrait, noPoints, maxPosEmDim = trajectory(signalData, maxEmDim, timeDelay)
# Plot
fig = plt.figure(figsize=(13, 9))
ax = fig.gca(projection='3d')
ax.set_xlim(-10, 15)
ax.set_ylim(-10, 15)
ax.set_zlim(-10, 15)
ax.view_init(25, -130)
ax.plot(portrait[:, 0], portrait[:, 1], portrait[:, 2], 'blue')
ax.set_xlabel('X(t)', rotation=0)
ax.set_ylabel('X(t + tau)', rotation=0)
ax.set_zlabel('X(t + 2tau)', rotation=270)
ax.set_title('Rossler Reconstructed with tau=63, steps=3.15', loc='center')
plt.show()

# Tau: 17
# Compute Trajectory
signalData = rossler_data[0, :]  # First factor (X component) of Rossler Attractor
maxEmDim = 3
timeDelay = 17
portrait, noPoints, maxPosEmDim = trajectory(signalData, maxEmDim, timeDelay)
# Plot
fig = plt.figure(figsize=(13, 9))
ax = fig.gca(projection='3d')
ax.set_xlim(-10, 15)
ax.set_ylim(-10, 15)
ax.set_zlim(-10, 15)
ax.view_init(25, -130)
ax.plot(portrait[:, 0], portrait[:, 1], portrait[:, 2], 'blue')
ax.set_xlabel('X(t)', rotation=0)
ax.set_ylabel('X(t + tau)', rotation=0)
ax.set_zlabel('X(t + 2tau)', rotation=90)
ax.set_title('Rossler Reconstructed with tau=17, steps=0.85', loc='center')
plt.show()

# Tau: 8
# Compute Trajectory
signalData = rossler_data[0, :]  # First factor (X component) of Rossler Attractor
maxEmDim = 3
timeDelay = 8
portrait, noPoints, maxPosEmDim = trajectory(signalData, maxEmDim, timeDelay)
# Plot
fig = plt.figure(figsize=(13, 9))
ax = fig.gca(projection='3d')
ax.set_xlim(-10, 15)
ax.set_ylim(-10, 15)
ax.set_zlim(-10, 15)
ax.view_init(25, -130)
ax.plot(portrait[:, 0], portrait[:, 1], portrait[:, 2], 'blue')
ax.set_xlabel('X(t)', rotation=0)
ax.set_ylabel('X(t + tau)', rotation=0)
ax.set_zlabel('X(t + 2tau)', rotation=180)
ax.set_title('Rossler Reconstructed with tau=8, steps=0.40', loc='center')
plt.show()

'''

signalData = rossler_data[0, :]  # First factor (X component) of Rossler Attractor
maxTimeDelay = 100
mutualInformation = mutualInformationWTS(signalData,maxTimeDelay)

fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot()
ax.plot(mutualInformation[:, 0], mutualInformation[:, 1], 'blue')
ax.set_xlabel('Time Shift Tau', rotation=0)
ax.set_ylabel('Mutual Information (bits)', rotation=90)
ax.set_title('Mutual Information for Rossier Attractor', loc='center')
plt.show()



