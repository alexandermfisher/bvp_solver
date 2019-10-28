import Simulation as sim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
import math
import sys

t0 = 100
t = np.linspace(0,t0,1000)
b_values = np.linspace(0.1,0.5,1)

for b in b_values:
	args = (1,0.1,b)
	solutions = sim.solve_ODE(sim.predator_prey_ODE,t,t0,[0.4,0.4],args)
	xs = solutions[0]
	ys = solutions[1]
	plt.plot(xs,ys) 

plt.show()





'''
x = np.arange(0,1,0.01)
y = np.arange(0,1,0.01)


X, Y = np.meshgrid(x, y)

u = X*(1-X)-(X*Y)/(0.1+X) 
v = 0.2*Y*(1-(Y/X))
fig, ax = plt.subplots(figsize=(7,7))
ax.quiver(X,Y,u,v)

plt.show()
'''
