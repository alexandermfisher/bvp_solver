# Standard imports as well as addition functions from Simulation.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as md
from scipy.integrate import odeint
from simulation_prey_predator import predator_prey_ODE as myode
from scipy.optimize import fsolve
from math import pi, sqrt
from numpy import append


def shooting(fun,phase,u0,T,args,plot=0):


	def G(u):
		x = u[:-1]		
		t = u[-1]
		solve_ode = odeint(fun, x, np.linspace(0,t,1000), args=(args,))

		residues = np.zeros(len(u)) 
		for i in range(len(x)): residues[i] = x[i] - solve_ode[-1,i]
		residues[-1] = phase(x,args) 

		return residues
		
	return fsolve(G, np.concatenate((u0,T),axis=None)) 


def continuation(fun,phase,u0,T,args,var_par = 0, max_steps = 100, step_size = 0.01):

	solutions = np.zeros((max_steps,3))
	params = np.zeros((max_steps,1))
	roots = shooting(fun,phase,u0,T,args)
	

	for i in range(max_steps):
		solutions[i,:] = roots
		params[i] = args[var_par]
		args[var_par] = (float(args[var_par]) - step_size)
		roots = shooting(fun,phase,roots[:-1],T,args)
	

	return solutions, params




def phase(u, args):
	x,y = u
	a,b = args 
	return x-sqrt(b)

u0 = np.array([sqrt(2),0])
T0 = np.array([2*pi])
args = np.array([-1,float(2)])

def hopf_ode(u,t,args):
			x,y = u
			a,b = args
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
			return [dxdt,dydt]



solutions, params = continuation(hopf_ode,phase,u0,T0,args, var_par = 1, max_steps = 1999, step_size = 0.001)







fig = plt.figure()
ax = plt.axes(projection="3d")

x_line = params[:]
z_line = solutions[:,0]
y_line = solutions[:,1]
ax.plot3D(x_line, y_line, z_line)


plt.show()



















'''


#Test 

def hopf_ode(u,t,args):
			x,y = u
			a,b = args
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
	
			return [dxdt,dydt]

def phase(u, args):
	x,y = u
	a,b = args 
	return x-sqrt(b)

u0 = np.array([sqrt(2),0])
T0 = np.array([2*pi])
args = np.array([-1,2])

def phase(u,args):
	x,y = u
	a,d,b = args
	
	return x*(1-x)-(a*x*y)/(d+x)

u0 = np.array([0.4,0.4])
T0 = np.array([20])
args = np.array([1,0.1,0.2])


solution = shooting(hopf_ode,phase,u0,T0,args)
print(np.shape(solution)) 

'''






