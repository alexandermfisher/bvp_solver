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
import sys
from scipy.optimize import fsolve

def shooting(fun,u0,phase,T,args,plot=0):

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




	Example
	--------		def simulation():
						u0 = [0.5,0.3]
						T = 25
						sol = shooting(fun, u0, lambda u0: u0[0]*(1-u0[0])-(u0[0]*u0[1])/(0.1+u0[0]), T, (1,0.1,0.2), plot = 0)
						print(sol)
						plot([sol[0],sol[1]],sol[2])
						return 

					simulation()
    """



    ###	Shooting Code
    
	### Solve `fun` with initail conditions `u0` and period `T` and store final values. 
	def res_fun(u):
		x = u[range(len(u)- 1)]		# Solving ode system fun integrating with odeint for given input x, t. 
		t = u[-1]
		sol_ode = odeint(fun, x, np.linspace(0,t,1000), args=(args,))
		
		sol = np.zeros(len(x)+1) 		# Solving and storing the residuals of the difference and phase condition. 
		for i in range(len(x)): sol[i] = abs(x[i] - sol_ode[-1,i])
		sol[-1] = phase(x)  

		return sol

	# Optional Plotting
	def plotoption(fun,u0,args,T):
		t = np.linspace(0, T,1000)
		output = odeint(fun,u0,t,args=(args,))
		plt.plot(t,output[:,0])
		plt.show()




	### result of Root finding using fsolve to solve boundary value problem.
	bvp_sol = fsolve(res_fun, np.concatenate((u0,T), axis=None))

	### if selected plotoption() will plot bvp_sol results.
	if plot == 1:
		plotoption(fun,[bvp_sol[0],bvp_sol[1]],args,bvp_sol[-1])

	
	
	return bvp_sol





'''
u0 = np.array([0.5,0.3])
T = np.array([25])
phase = lambda u0: u0[0]*(1-u0[0])-(u0[0]*u0[1])/(0.1+u0[0])
print(shooting(fun, u0, phase, T, args = [1,0.1,0.2], plot = 0))
'''





