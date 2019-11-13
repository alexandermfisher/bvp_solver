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
from math import pi, sqrt



def shooting(fun,u0,phase,T,args,plot=0):

	"""

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



    ###	Shooting Code
    
	
	def G(u):
		x = u[:-1]		
		t = u[-1]
		sol_ode = odeint(fun, x, np.linspace(0,t,1000), args=(args,))
		
		sol = np.zeros(len(x)+1) 		# Solving and storing the residuals of the difference and phase condition. 
		for i in range(len(x)): sol[i] = x[i] - sol_ode[-1,i]
		sol[-1] = phase(x,args)  

		return sol





	# Optional Plotting
	def plotoption(fun,u0,args,T):
		t = np.linspace(0, T,1000)
		output = odeint(fun,u0,t, args=(args,))
		plt.plot(t,output[:,0])
		plt.show()




	### result of Root finding using fsolve to solve boundary value problem.
	bvp_sol = fsolve(G, np.concatenate((u0,T), axis = None))

	### if selected plotoption() will plot bvp_sol results.
	if plot == 1:
		plotoption(fun,[bvp_sol[0],bvp_sol[1]],args,bvp_sol[-1])

	
	
	return bvp_sol




u0 = np.array([0.5,0.4])
T = np.array([20])
args = [1,0.1,0.2]
def phase(u,args):
	a,d,b = args
	phi = u[0]*(1-u[0])-(a*u[0]*u[1])/(d+u[0])

	return phi


print(shooting(fun, u0, phase, T,args, plot = 0))








'''

def continuation(fun,u0,T,phase,par0,var_par_index,max_steps = 100,step_size = 0.1):

	solutions = []
	params = []
	roots = shooting(fun,u0,phase,T,par0)


	for i in range(max_steps):
		solutions.append(roots)
		params.append(par0[var_par_index])
		par0[var_par_index] = par0[var_par_index]-step_size
		roots = shooting(fun,u0,phase,T,par0)
	return

	return [solutions, params]


def hopf_ode(u0,t,args):
			x,y = u0
			a,b = args
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
			return [dxdt,dydt]
			
	
phase = lambda u0: u0[0]*(1-u0[0])-(u0[0]*u0[1])/(0.1+u0[0])

def phase(u0, args):
	a,b = args 
	return int(u0[0]-sqrt(abs(b)))

sol = continuation(hopf_ode,[0.8*sqrt(2),0.01],2*pi,phase,[-1,2],var_par_index = 1)

print(sol)
'''

