




### Description of module 




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
import sys



def shooting(fun,u0,args,phase = None,solver=scipy.optimize.fsolve):

"""This function perfomrs numerical shooting on an n-dimentional system of first-order ordinary differnetial equations.

   

    Parameters
    ----------

				fun:	callable
						Right-hand side of the system. Dimentionality of (n)

    			u0:		array_like, shape
    					initial guess at starting boundary conditions for the given BVP with the guess augemented on the end.
    					E.G for a 2d system the general input will look like [x1,x2,t], where x1,x2 are initial guess of state space vals, and t is guess at period.
        		
				phase: 	callable 
						A given phase given condition with inputs same as u0 	
	
    Returns
    -------		
				solution:	ndarray,shape(n_points,)
							corrected inital guess, and period. (Same format as input 'u0')




	Example
	--------
						def hopf_ode(u,t,args):
								x,y = u
								a,b = args
								dxdt = b*x-y+a*x*(x**2+y**2)
								dydt = x+b*y+a*y*(x**2+y**2)
								return [dxdt,dydt]

						u0 = [2, 0, 2*pi]
						args = [2,1]

						def phase(u, args):
								x,y = u
								a,b = args 
								return x-sqrt(b)


						solution = shooting(hopf_ode,u0,args,phase)
						

---------------------------------------------------------------------------------------------------
"""   #####   Shooting Code  #######


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

		
		try:
			solver(G,u0,args=(args,))
			

		except ValueError:
			print("Oops Value Error: Failed to converge. Please try another value or equation.")
			sys.exit()

		except TypeError:
			print("Oops TypeError: check that the inputs are of the correct type.")
			sys.exit()
		
		except IndexError:
			print("Oops IndexError: Check that dimentions of inputs makes sense.")
			sys.exit()
			



		solution = solver(G,u0,args=(args,))

		return	solution








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



