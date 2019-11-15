"""
-----------------------------------------------------------------------------------------------------------

Test Script:

This is a script designed to test the shooting and natural continuation code in bvp_solver.py
for diffent types of inputs.  This will include testing against ode systems with known analytic solutions.
And producing graphs of the resulting continuation scheme. 

___________________________________________________________________________________________________________


How to USE:

This script takes a command line argument in the form of a single digit in domain [0,5].
Consequently either test 1,2,3,4,5 or 6 is run. For more information and detail on individual tests see code below.

"""


### Standard Imorts as as well as code for testing from bvp_solver.py

from bvp_solver import shooting, continuation
from math import pi, sqrt
import numpy as np
from scipy.optimize import fsolve
import scipy
from matplotlib import pyplot as plt
import sys


# 1,2: Shooting tests given analytical solutions:

def test_hopf_bifurcation(u0,args,phase):

		def hopf_ode(u,t,args):
			x,y = u
			a,b = args
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
			return [dxdt,dydt]

		shooting_sol = shooting(hopf_ode,u0,args,phase)
		numerical_sol = []
		analytical_sol = []	
		numerical_sol.append(shooting_sol)
		analytical_sol.append([sqrt(args[1]),0,2*pi])

		
		test1 = np.isclose(numerical_sol, analytical_sol, atol=1e-04)
		test2 = np.allclose(numerical_sol,analytical_sol, rtol=1e-05, atol=1e-04)


		print("test_hopf_bifurcation using np.isclose()")
		print(test1[0])
		print("test_hopf_bifurcation using np.allclose")
		print(test2)

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

		print("test_hopf_bifurcation_3D using np.isclose()")
		print(test1[0])
		print("test_hopf_bifurcation_3D using np.allclose")
		print(test2)

		return [test1[0],test2]


# 3,4,5: Continuation tests: given exmaple exquations a plot is returned: 

def test_cubic_equation():

	def fun(x,c): 
		return x**3-x+c 
	
	u0 = np.array([float(0)])
	c  = np.array([float(2)])
	solutions, params = continuation(fun,u0,c, phase = None, var_par = 0, max_steps = 399, step_size = 0.01)
	
	plt.scatter(params,solutions)
	plt.show()


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

	solutions, params = continuation(hopf_ode_mod,u0,args, phase, var_par = 1, max_steps = 20, step_size = 0.15)
	
	plt.scatter(params,solutions[:,0])
	plt.show()


	return


# 6: Error and Raise checks:

def incorrect_type_test():
	fun = ["string"] 
	return continuation(fun,[1],[1])



###----------------Run Tests----------------###

def runtest(i=0):


	###--------------------------------------------------------------------------###
	### Testing section for shooting using hopf bifurcation equations 2d, and 3d ###
	###--------------------------------------------------------------------------###


	#----------------------------------------------2d Hopf Bifurcation:

	if i ==0:


		u0 = np.array([sqrt(2),0,2*pi])

		args = [-1,2]

		def phase(u, args):
			x,y = u
			a,b = args 
			return x-sqrt(b)

		test_hopf_bifurcation(u0,args,phase)


	#----------------------------------------------3d Hopf Bifurcation:
	
	elif i ==1:

		u0 = np.array([sqrt(2),0,0,2*pi])

		args = [-1,2]

		def phase(u, args):
			x,y,z = u
			a,b = args 
			return x-sqrt(b)

		test_hopf_bifurcation_3D(u0,args,phase)



	###--------------------------------------------------------------------------------###
	### Testing section for Continuation using a cubic equation and hopf bifurcations  ###
	###--------------------------------------------------------------------------------###



	#----------------------------------------------Cubic Equation:
	elif i == 2:

		test_cubic_equation()
	
	#----------------------------------------------Normal Hopf Bifurcation:
	elif i ==3:

		u0 = np.array([sqrt(2),0,2*pi])

		args = [-1,2]

		def phase(u, args):
			x,y = u
			a,b = args 
			return x-sqrt(b)

		test_hopf_bifurcation_normal(u0,args,phase)

	#----------------------------------------------Modified Hopf Bifurcation:
	elif i == 4:

		u0 = np.array([sqrt(2),0,2*pi])

		args = [-1,2]

		def phase(u, args):
			x,y = u
			a,b = args 
			return  b*x-y-a*x*(x**2+y**2)+a*x*(x**2+y**2)**2

		test_hopf_bifurcation_modified(u0,args,phase)

	#----------------------------------------------Incorrect Data Type:
	elif i == 5:

		incorrect_type_test()

	else:
		print("woops wrong input try again: choose either 0,1,2,3,4, or 5")


	return


def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


try:
	for arg in sys.argv[1]:
		if not is_intstring(arg):
			sys.exit("Arguments must be a single integer in domain [0,5]. Exit.")

except IndexError:
	print("Please enter an integer in domain [0,5]")


i = int(sys.argv[1])
runtest(i)










