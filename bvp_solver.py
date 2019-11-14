# Standard imports as well as addition functions from Simulation.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as md
from scipy.integrate import odeint
from simulation_prey_predator import predator_prey_ODE as myode
from scipy.optimize import fsolve
from math import pi, sqrt
from numpy import append
import scipy



def shooting(fun,u0,args,phase = None,solver=scipy.optimize.fsolve):

	def _G(u,args):
		x = u[:-1]		
		t = u[-1]
		solve_ode = odeint(fun, x, np.linspace(0,t,1000), args=(args,))

		residues = np.zeros(len(u)) 
		for i in range(len(x)): residues[i] = x[i] - solve_ode[-1,i]
		residues[-1] = phase(x,args) 
		
		return residues

	if phase == None: 

			G = fun
	else:
			G = _G

		 
	return fsolve(G, u0, args=(args,)) 


def continuation(fun,u0,args, phase = None, var_par = 0, max_steps = 100, step_size = 0.01):

	solutions = np.zeros((max_steps,int(len(u0))))
	params = np.zeros((max_steps,1))
	roots = shooting(fun,u0,args,phase)
	

	for i in range(max_steps):
		solutions[i,:] = roots
		params[i] = args[var_par]
		args[var_par] = (float(args[var_par]) - step_size)
		roots = shooting(fun,roots, args, phase)
	

	return solutions, params




