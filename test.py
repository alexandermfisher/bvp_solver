"""
Test Script:

This is a script designed to test the shooting code in shooting.py
for diffent types of inputs and to see accuacry and effectivess as well. This will include testing against ode systems with known analytic solutions.
In addition erros messages and raises will also be tested.  

"""

import shootingmethod as sm
from math import pi, sqrt
import numpy as np


# Shooting test given aanalytical solutions:

def test_hopf_bifurcation(b0,bf):

		def hopf_ode(u0,t,b,a):
			x,y = u0
			dxdt = b*x-y+a*x*(x**2+y**2)
			dydt = x+b*y+a*y*(x**2+y**2)
			return [dxdt,dydt]

		def simulation(b,a):
			sol = sm.shooting(hopf_ode, [0.8*sqrt(b),0.01], lambda u0: u0[0]-sqrt(b), 2*pi, (b,a))
			return sol[0]

		numerical_sol = []
		analytical_sol =[]
		for i in [int(x) for x in np.linspace(b0,bf,50)]:	
			numerical_sol.append(simulation(i,-1))
			analytical_sol.append(sqrt(i))

		sol1 = np.isclose(numerical_sol, analytical_sol, atol=1e-08)
		sol2 = np.allclose(numerical_sol,analytical_sol, rtol=1e-05, atol=1e-08)

		print("Results using np.isclose()")
		print(sol1)
		print("Results using np.allclose")
		print(sol2)

		return 

# Run Test:
test_hopf_bifurcation(0,2)


def test_hopf_bifurcation_3D():















