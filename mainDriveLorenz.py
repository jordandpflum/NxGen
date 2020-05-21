from dynamicSystems.lorenzAttractor import lorenz_generate
from general.trajectory import trajectory
from general.trajectory import trajectory2
import numpy as np

# Plotting
from general.Plotting.graphicalPlotting import generate3Dplot
from general.Plotting.graphicalPlotting import generate2Dplot

# Optimal Tau
from general.mutualInformation.mutualInformationWTS import mutualInformationWTS
from general.optimalTau import generateOptimalTau

# Embedding Dimension
from general.optimalEmbeddingDimension import grassberg_procaccia

# Rossler parameters: Paper
sigma = 10
b = 8/3
r = 28

#a = 0.15
#b = 0.2
#c = 10

# Initial condition (random, just interesting starting points)
initial_condition = np.array([0, 1, 0])

#initial_condition = np.array([0.5, 0, 0])

# Set Time Parameters
# Paper Parameters
dt = 0.05
num_points = 4096
# tspan = np.arange(start=dt, stop=dt*num_points, step=dt)

# Generate Rossler Attractor
lorenz_data = lorenz_generate(num_points=num_points, initial_state=initial_condition, parameters=np.array([sigma, b, r]),
                                dt=dt)

# Plot Rossler Attractor
generate3Dplot(lorenz_data,
               title='The Lorenz Attractor',
               axis_labels=('X(t)', 'Y(t)', 'Z(t)'),
               axis_lim=([-10, 15], [-10, 15], [0, 25]))


# Calculate Optimal Choice of Time Shift Tau
# Create Signal Data
signalData = lorenz_data[:, 0]  # First factor (X component) of Rossler Attractor

'''
# Set Max Time Delay when calculating Mutual Information
maxTimeDelay = 100

# Calculate Mutual Information for every time delay
mutualInformation = mutualInformationWTS(signalData, maxTimeDelay)

# Plot Mutual Information
generate2Dplot(data=mutualInformation,
               title='Mutual Information for Rossier Attractor',
               axis_labels=('Time Shift Tau', 'Mutual Information (bits)'),
               axis_lim_inc=False
               )
'''

# Get Optimal Choice of Time Shift Tau

optimalTau = generateOptimalTau(signalData, dimension=3, method="Minimizing MI over first (d-1) Tau")
#optimalTau = generateOptimalTau(signalData, dimension=3, method="First Local Minimum")

# Compute Trajectory
maxEmDim = 3
timeDelay = optimalTau
portrait = trajectory(signalData, maxEmDim, timeDelay)

# Plot
'''
generate3Dplot(data=portrait,
               title='Rossler Reconstructed with tau=' + str(timeDelay),
               axis_labels=('X(t)', 'X(t + tau)', 'X(t + 2tau)'),
               axis_lim=([-10, 15], [-10, 15], [-10, 15])
               )
'''

# corr_dim = grassberg_procaccia(signalData, 3, timeDelay, plot=True)
from general.optimalEmbeddingDimension import grassberg_procaccia_test
corr_dim = grassberg_procaccia_test(lorenz_data, True)
print('Estimated Fractal Dimension: ' + str(corr_dim))
print('Know Fractal Dimension: ~2.05')