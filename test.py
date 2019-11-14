"""
Test Script:

This is a script designed to test the shooting code in shooting.py
for diffent types of inputs and to see accuacry and effectivess as well. This will include testing against ode systems with known analytic solutions.
In addition erros messages and raises will also be tested.  

"""
from bvp_solver import shooting, continuation
from math import pi, sqrt
import numpy as np
from scipy.optimize import fsolve
import scipy

# Shooting test given analytical solutions:

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
		print(shooting_sol)
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


















































###--------------------------------------------------------------------------###
### Testing section for shooting using hopf bifurcation equations 2d, and 3d ###
###--------------------------------------------------------------------------###

def phase(u, args):
	x,y,z = u
	a,b = args 
	return x-sqrt(b)

u0 = np.array([sqrt(2),0,0,2*pi])

test_hopf_bifurcation_3D(u0,[-1,2],phase)

















