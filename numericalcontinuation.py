"""

Natural Paramater Continuation 


"""
import shootingmethod as sm
from math import pi, sqrt
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
from simulation_prey_predator import predator_prey_ODE as fun
from simulation_prey_predator import plot
import sys
from scipy.optimize import fsolve







def continuation(myode, u0,T, phase, par0, var_par_index = 0, step_size=0.1, max_steps=100, discretisation = 'shooting'): 
	
	if discretisation == 'shooting':

		solutions, parmas = contin_shooting(myode,u0,T,phase,par0,var_par_index,max_steps,step_size)

		print(solutions)
		print(params)

	else:

		return

	return

phase = lambda u0: u0[0]*(1-u0[0])-(u0[0]*u0[1])/(0.1+u0[0])









def contin_shooting(fun,u0,T,phase,par0,var_par_index,max_steps,step_size):

	solutions = []
	params = []
	roots = sm.shooting(fun,u0,phase,T,par0)


	for i in range(max_steps):
		solutions.append(roots)
		params.append(par0[var_par_index])
		par0[var_par_index] = par0[var_par_index]-step_size
		roots = sm.shooting(fun,u0,phase,T,par0)
	return

	return [solutions, params]



	


continuation(fun,[0.4,0.4],25,phase,[1,0.1,1],var_par_index = 2)





























