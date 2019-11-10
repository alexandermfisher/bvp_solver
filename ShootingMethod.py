"""
#####Numerical Shooting Methdod to solve Boundary Value Probems#####

An implenation of the 'Numerical Shooting Method' to solve boundary 
value problems. This module consisits of several modular functions
that may be imported indicually or imported and used in conjuction with each other. 
This module takes in as input a n-dim. system of first order equations. I.e. higher ordewr ODEs 
have been reduced to state space and are in first order coupoled form.   



"""


# Standard imports as well as addition functions from Simulation.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
from Simulation import predator_prey_ODE as F
from Simulation import plot
import sys
from scipy.optimize import fsolve







def shooting(fun,u0,phase,T,n):

	"""Calculates the residues for the given initial conditions and BVP problem.

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`. Note the function utilises scipy.intergrate.ode for integrations purposes. 

    
    Parameters
    ----------

				fun:	callable
						Right-hand side of the system. Dimentionality of (n)



    			u0:		array_like, shape
    					initial guess at starting boundary conditions for the given BVP.
        		

				phase: 	
	


				T:		int
						guess at period value


				n:		int
						dimention of `fun`. That is to say the the dimentionality of the right hand side system (n).
    Returns
    -------
    		
				sol:	


				t:		ndarray,shape(n_points,)
						Time points

				y:		ndarray,shape(n,n_points)
						Values of the solution at t			
        
   

    """

	### Solve `fun` with initail conditions `u0` and period `T` and store final values. 


	def res_fun(u):
		x = u[range(len(u)- 1)]
		t = u[-1]
		sol_ode = odeint(fun, x, np.linspace(0,t,1000), args=(1,0.1,0.2))
		sol = np.zeros(n+1) 
		for i in range(n): sol[i] = abs(x[i] - sol_ode[-1,i])
		sol[-1] = (phase)(x) 

		return sol


	roots = fsolve(res_fun, u0+[T])


	return roots 
	


#Call/run  red_fun()  example:


u0 = [0.4,0.4]
T = 18

sol = shooting(F, u0, lambda u0: u0[0]*(1-u0[0])-(u0[0]*u0[1])/(0.1+u0[0]), T, 2)
print(sol)

plot([sol[0],sol[1]],sol[2])





















