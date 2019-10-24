import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import sys


def predator_prey_ODE(x,t,a,b,d):
	xs = x[0]
	ys = x[1]
	
	dxdt = xs*(1-xs)-(a*xs*ys)/(d+xs) 
	dydt = b*ys*(1-(ys/xs))

	return [dxdt,dydt]



# b: [0.1,0.5] test1 parameters with b at lower bound and upper bound. 
paramstest1 = (1,0.1,0.1)	
paramstest2 = (1,0.1, 0.5)
param_vec = [paramstest1,paramstest2]


def solve_ODE(f_ode,t,t0,x0,a,b,d):
	t = np.linspace(0, 100, t0)
	xs = odeint(f_ode,x0,t,args=(a,b,d))
	x,y = xs[:,0], xs[:,1]

	return [x,y]

#plotting results
t = np.linspace(0,1000,1000)
output1 = solve_ODE(predator_prey_ODE,t,1000,[1,1],paramstest1[0],paramstest1[1],paramstest1[2])
output2 = solve_ODE(predator_prey_ODE,t,1000,[1,1],paramstest1[0],paramstest2[0],paramstest2[2])


solutions = np.stack((output1, output2), axis = 0) 




x1 = solutions[0][0]
y1 = solutions[0][1]
x2 = solutions[1][0]
y2 = solutions[1][1]


plt.figure(1)
plt.subplot(221)
plt.semilogy(t,x1)

plt.subplot(222)
plt.semilogy(t,y1)

plt.subplot(223)
plt.semilogy(t,x2)

plt.subplot(224)
plt.semilogy(t,y2)







plt.figure(2)
plt.subplot(2, 2, 1)
plt.plot(t,x1)
plt.title('predator  [d = 0.1]')
plt.ylabel('Population')

plt.subplot(2, 2, 2)
plt.plot(t,y1)
plt.title('prey  [d = 0.1]')
plt.ylabel('Population')

plt.subplot(2, 2, 3)
plt.plot(t,x2)
plt.title('predator  [d = 0.5]')
plt.ylabel('Population')

plt.subplot(2, 2, 4)
plt.plot(t,y2)
plt.title('prey  [d = 0.5]')
plt.ylabel('Population')



plt.show()


		




