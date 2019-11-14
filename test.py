"""
Test Script:

This is a script designed to test the shooting code in bvp_solver.py
for diffent types of inputs and to see accuacry and effectivess as well. This will include testing against ode systems with known analytic solutions. 

"""




from bvp_solver import shooting, continuation
from math import pi, sqrt
import numpy as np
from scipy.optimize import fsolve
import scipy
from matplotlib import pyplot as plt
import argparse
import sys

# Shooting tests given analytical solutions:

def test_hopf_bifurcation(u0,args,phase):

		def hopf_ode(u,t,args):
			x,y = u
			a,b = args
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
			return [dxdt,dydt]

		shooting_sol = shooting(hopf_ode,u0,args,phase)
		print(shooting_sol)
		numerical_sol = []
		analytical_sol = []	
		numerical_sol.append(shooting_sol)
		analytical_sol.append([sqrt(args[1]),0,2*pi])

		
		test1 = np.isclose(numerical_sol, analytical_sol, atol=1e-04)
		test2 = np.allclose(numerical_sol,analytical_sol, rtol=1e-05, atol=1e-04)


		"""
		print("test_hopf_bifurcation using np.isclose()")
		print(test1)
		print("test_hopf_bifurcation using np.allclose")
		print(test2)
		"""
		return  [test1[0],test2]


def test_hopf_bifurcation_3D(u0,args,phase):


		def hopf_ode(u,t,args):
			x,y,z = u
			a,b = args
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
			dzdt = -z

			return [dxdt,dydt,dzdt]


		shooting_sol = shooting(hopf_ode,u0,args,phase)
		numerical_sol = []
		analytical_sol = []	
		numerical_sol.append(shooting_sol)
		analytical_sol.append([sqrt(args[1]),0,0,2*pi])

		
		test1 = np.isclose(numerical_sol, analytical_sol, atol=1e-04)
		test2 = np.allclose(numerical_sol,analytical_sol, rtol=1e-05, atol=1e-04)

		print("test_hopf_bifurcation using np.isclose()")
		print(test1[0])
		print("test_hopf_bifurcation using np.allclose")
		print(test2)

		return [test1[0],test2]


# Continuation tests: given exmaple exquations a plot is returned: 

def test_cubic_equation():

	def fun(x,c): 
		return x**3-x+c 
	
	u0 = np.array([float(0)])
	c  = np.array([float(2)])
	solutions, params = continuation(fun,u0,c, phase = None, var_par = 0, max_steps = 399, step_size = 0.01)
	
	plt.scatter(params,solutions)


	return


def test_hopf_bifurcation_normal(u0,args,phase):

	def hopf_ode(u,t,args):
			x,y = u
			a,b = args
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
			return [dxdt,dydt]

	solutions, params = continuation(hopf_ode,u0,args, phase, var_par = 1, max_steps = 199, step_size = 0.01)
	
	plt.scatter(params,solutions[:,0])
	plt.show()

	#### Add numerical solutions using params, and anaytical solutions to solve u1(0) for each case hrapgh should be same, and vals as well.  


	return


def test_hopf_bifurcation_modified(u0,args,phase):

	def hopf_ode_mod(u,t,args):
			x,y = u
			a,b = args
			dxdt = b*x-y-a*x*(x**2+y**2)+a*x*(x**2+y**2)**2
			dydt = x+b*y+a*y*(x**2+y**2)+a*y*(x**2+y**2)**2
			return [dxdt,dydt]

	solutions, params = continuation(hopf_ode_mod,u0,args, phase, var_par = 1, max_steps = 40, step_size = 0.075)
	
	plt.scatter(params,solutions[:,0])
	plt.show()


	return


# Error and Raise checks:

def incorrect_type_test():
	fun = ["string"] 
	return continuation(fun,[1],[1])












###--------------------------------------------------------------------------###
### Testing section for shooting using hopf bifurcation equations 2d, and 3d ###
###--------------------------------------------------------------------------###


#----------------------------------------------3d Hopf Bifurcation:


u0 = np.array([sqrt(2),0,0,2*pi])

args = [-1,2]

def phase(u, args):
	x,y,z = u
	a,b = args 
	return x-sqrt(b)

test_hopf_bifurcation_3D(u0,[-1,2],phase)


#----------------------------------------------2d Hopf Bifurcation:



###--------------------------------------------------------------------------------###
### Testing section for Continuation using a cubic equation and hopf bifurcations  ###
###--------------------------------------------------------------------------------###



#----------------------------------------------Cubic Equation:


test_cubic_equation()

#----------------------------------------------Normal Hopf Bifurcation:


u0 = np.array([sqrt(2),0,2*pi])

args = [-1,2]

def phase(u, args):
		x,y = u
		a,b = args 
		return x-sqrt(b)

test_hopf_bifurcation_normal(u0,args,phase)

#----------------------------------------------Modified Hopf Bifurcation:


u0 = np.array([sqrt(2),0,2*pi])

args = [-1,2]

def phase(u, args):
	x,y = u
	a,b = args 
	return  b*x-y-a*x*(x**2+y**2)+a*x*(x**2+y**2)**2

test_hopf_bifurcation_modified(u0,args,phase)

#----------------------------------------------Incorrect Data Type:


incorrect_type_test()














