"""
#####Numerical Shooting Methdod to solve Boundary Value Probems#####

Implemation of the 'Numerical Shooting Method' to solve boundary 
value problems. This module consisits of several modular functions
that may be imported indivually or imported and used in conjuction with each other. 
This module takes in as input a n-dim. system of first order equations. I.e. higher ordewr ODEs 
have been reduced to state space and are in first order coupled form.   

"""
# Standard imports as well as addition functions from Simulation.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
from simulation_prey_predator import predator_prey_ODE as fun
from simulation_prey_predator import plot
import sys
from scipy.optimize import fsolve

def shooting(fun,u0,phase,T,args):

	"""Calculates the residues for the given initial conditions and BVP problem.

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`. Note the function utilises scipy.intergrate.ode for integrations purposes. 

    Parameters
    ----------

				fun:	callable
						Right-hand side of the system. Dimentionality of (n)

    			u0:		array_like, shape
    					initial guess at starting boundary conditions for the given BVP.
        		
				phase: 	callable 
						A given phase given condition with inputs same as u0 	
	
				T:		int
						guess at period value
    Returns
    -------		
				sol:	

				t:		ndarray,shape(n_points,)
						time points

				y:		ndarray,shape(n,n_points)
						values of the solution at t			

    """

    ### Unpack variables and find dimention (n) of given ode system.
	args = args 
	n = len(u0)

	### Solve `fun` with initail conditions `u0` and period `T` and store final values. 
	def res_fun(u):
		x = u[range(len(u)- 1)]		# Solving ode system fun integrating with odeint for given input x, t. 
		t = u[-1]
		sol_ode = odeint(fun, x, np.linspace(0,t,1000), args=args)
		
		sol = np.zeros(n+1) 		# Solving and storing the residuals of the difference and phase condition. 
		for i in range(n): sol[i] = abs(x[i] - sol_ode[-1,i])
		sol[-1] = phase(x)  

		return sol

	
	return fsolve(res_fun, u0+[T]) 	### return result of Root finding using fsolve to solve boundary value problem.
	



#Simulation of shooting.() fynction

def simulation():
	u0 = [0.5,0.3]
	T = 25
	sol = shooting(fun, u0, lambda u0: u0[0]*(1-u0[0])-(u0[0]*u0[1])/(0.1+u0[0]), T, (1,0.1,0.2))
	print(sol)
	plot([sol[0],sol[1]],sol[2])
	return 



simulation()





















