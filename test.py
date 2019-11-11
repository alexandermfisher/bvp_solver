"""
Test Script:

This is a script designed to test and run the shooting code in shooting.py
through multiple differnt tests. They include testing against ode systems with known analytic solutions. 

"""

import shootingmethod
import unittest


"""
Create inputs 

Execute the code being tested, capturing the output

Compare the output with an expected result

first set:

"""



class test_shootingmethod_py(unittest.TestCase):

    def test_hopf_bifurcation(self):

    	def hopf_ode(u0,t,b,a):
    		x,y = u0
    		dxdt = bx-y+a*x*(x**2+y**2)
    		dydt = x+b*y+a*y*(x**2+y**2)
    		return [dxdt,dydt]
    		











        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")














    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()




