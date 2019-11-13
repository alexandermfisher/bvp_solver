# Standard imports as well as addition functions from Simulation.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as md
from scipy.integrate import odeint
from simulation_prey_predator import predator_prey_ODE as myode
from scipy.optimize import fsolve
from math import pi, sqrt
from numpy import append



def shooting(fun,u0,args,phase = None, arclengthcon = None):

	def _G(u,args):
		x = u[:-1]		
		t = u[-1]
		solve_ode = odeint(fun, x, np.linspace(0,t,1000), args=(args,))

		residues = np.zeros(len(u)) 
		for i in range(len(x)): residues[i] = x[i] - solve_ode[-1,i]
		residues[-1] = phase(x,args) 
		
		return residues

	def __G(v,args):
		
		
		x = v[:-2]		
		t = v[-2]
		p = v[-1]
		argss = np.concatenate((args,p),axis=None)
		solve_ode = odeint(fun, x, np.linspace(0,t,1000), args=(argss,))

		residues = np.zeros(len(v)) 
		for i in range(len(x)): residues[i] = x[i] - solve_ode[-1,i]
		residues[-2] = phase(x,argss) 
		residues[-1] = arclengthcon(v,argss) 
		
		return residues

	if phase == None: 
		G = fun
	else:
		if arclength == None:
			G = _G 
		else:
			G = __G 

		 
	return fsolve(G, u0, args=(args,)) 

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


def arclength(fun,u0,args,phase=None, var_par=0,max_steps=100,step_size=0.01,step_length = 0.01):

	## Generate first two points using continuation max_steps= 2

	sol,params = continuation(fun,u0,args,phase,var_par,2,step_size)
	v = np.concatenate((sol,params), axis=1)

	secant = v[1,:]-v[0,:]
	v2 = v[1,:]+secant*step_length

	arc = lambda v: np.dot((v-v2),secant)
	print(arc)

	sol = shooting(fun,v2,args,phase,arclengthcon=arc)

	

	return sol 





def phase(u, args):
	x,y = u
	a,b = args 
	return x-sqrt(b)

u0 = np.array([sqrt(2),0, 2*pi,2])
#T0 = np.array([2*pi])
args = np.array([-1,float(2)])
argss = np.array([-1])

def hopf_ode(u,t,args):
			x,y = u
			a,b = args
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
			return [dxdt,dydt]




sol = arclength(hopf_ode,u0,argss,phase, var_par=0,max_steps=100,step_size=0.01,step_length = 0.01)

"""

sol = shooting(hopf_ode,u0,argss,phase,arclength)
print(sol)

"""





















#arclength(hopf_ode,u0,args,phase,var_par=1,max_steps=2,step_size=0.01)
'''
solutions, params = continuation(hopf_ode,u0,args, phase, var_par = 1, max_steps = 2, step_size = 0.01)
sol = np.concatenate((solutions,params), axis=1)
print(sol)
print(np.shape(sol))
'''





"""
def phase(u, args):
	x,y = u
	a,b = args 
	return x-sqrt(b)

u0 = np.array([sqrt(2),0])
T0 = np.array([2*pi])
args = np.array([-1,float(2)])

def hopf_ode(u,t,args):
			x,y = u
			a,b = args
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
			return [dxdt,dydt]

"""
'''
def fun(x,c): 
	return x**3-x+c 

u0 = np.array([1.5])
c = np.array([float(2)])
solutions, params = continuation(fun,u0,c, phase = None, var_par = 0, max_steps = 399, step_size = 0.01)
plt.plot(params,solutions)
plt.show()

'''



"""
fig = plt.figure()
ax = plt.axes(projection="3d")

x_line = params[:]
z_line = solutions[:,0]
y_line = solutions[:,1]
ax.plot3D(x_line, y_line, z_line)


plt.show()
"""
'''


#Test 

def hopf_ode(u,t,args):
			x,y = u
			a,b = args
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
	
			return [dxdt,dydt]

def phase(u, args):
	x,y = u
	a,b = args 
	return x-sqrt(b)

u0 = np.array([sqrt(2),0])
T0 = np.array([2*pi])
args = np.array([-1,2])

def phase(u,args):
	x,y = u
	a,d,b = args
	
	return x*(1-x)-(a*x*y)/(d+x)

u0 = np.array([0.4,0.4])
T0 = np.array([20])
args = np.array([1,0.1,0.2])


solution = shooting(hopf_ode,phase,u0,T0,args)
print(np.shape(solution)) 

'''




