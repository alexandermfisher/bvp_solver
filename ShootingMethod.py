import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
from Simulation import predator_prey_ODE as F
from Simulation import plot
import sys
from scipy.optimize import fsolve


##### Numerical Shooting Methdod to solve Boundary Value Probems. 


# residue function to be fsolved. 
def res_fun(u):
	x0 = u[0]
	y0 = u[1]
	T = u[2]
	
	t = np.linspace(0, T,1000)
	sol_ode = odeint(F,[x0,y0],t,args=(1,0.1,0.2))
	
	sol = np.zeros(3)
	sol[0] = x0 - sol_ode[-1,0]
	sol[1] = y0 - sol_ode[-1,1]
	sol[2] = x0*(1-x0)-(x0*y0)/(0.1+x0) 


	return sol


solution = fsolve(res_fun, [0.4,0.4,20])

print(solution)
print(res_fun(solution))

x = [solution[0],solution[1]]
plot(x)

