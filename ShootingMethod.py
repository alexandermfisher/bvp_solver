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
	x0, y0, T = u
	
	t = np.linspace(0, T,1000)
	sol_ode = odeint(F,[x0,y0],t,args=(1,0.1,0.2))
	
	sol = np.zeros(3) 
	sol[0] = abs(x0 - sol_ode[-1,0])
	sol[1] = abs(y0 - sol_ode[-1,1])
	sol[2] = x0*(1-x0)-(x0*y0)/(0.1+x0)

	return sol



def solve_bvp(u):
	x0, y0, T= u
	xs0, ys0, Ts = fsolve(res_fun, [x0, y0, T])

	t = np.linspace(0,Ts,1000)
	sol_ode = odeint(F,[xs0,ys0],t,args=(1,0.1,0.2))
	xf = sol_ode[-1,0]
	yf = sol_ode[-1,1]

	return xs0, ys0, xf, yf, Ts

xs0, ys0, xf, yf, Ts = solve_bvp([0.5,0.2,25])
print(xs0, ys0, xf, yf, Ts)
















