import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
import math
import sys



# coupled ODE function
def predator_prey_ODE(x,t,args): 
	xs = x[0]
	ys = x[1]
	a,d,b = args
	dxdt = xs*(1-xs)-(a*xs*ys)/(d+xs) 
	dydt = b*ys*(1-(ys/xs))

	return [dxdt,dydt]

def plot(x,T):
	x0 = x
	t = np.linspace(0, T,1000)
	args = (1,0.1,0.2)
	output = odeint(predator_prey_ODE,x0,t,args=(args,))

	plt.plot(t,output[:,0])
	plt.show()



