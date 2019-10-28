import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
import math
import sys



# coupled ODE function
def predator_prey_ODE(x,t,a,d,b):
	xs = x[0]
	ys = x[1]
	
	dxdt = xs*(1-xs)-(a*xs*ys)/(d+xs) 
	dydt = b*ys*(1-(ys/xs))

	return [dxdt,dydt]



def solve_ODE(f_ode,t,t0,x0,args):
	t = np.linspace(0, t0,1000)
	xs = odeint(f_ode,x0,t,args=args)
	x,y = xs[:,0], xs[:,1]

	return [x,y]






