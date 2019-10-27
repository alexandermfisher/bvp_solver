import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
import math
import sys




def predator_prey_ODE(x,t,a,d,b):
	xs = x[0]
	ys = x[1]
	
	dxdt = xs*(1-xs)-(a*xs*ys)/(d+xs) 
	dydt = b*ys*(1-(ys/xs))

	return [dxdt,dydt]



# b: [0.1,0.5] test1 parameters with b at lower bound and upper bound. 
paramstest1 = (1,0.1,0.10)	
paramstest2 = (1,0.1,0.8)
param_vec = [paramstest1,paramstest2]


def solve_ODE(f_ode,t,t0,x0,a,d,b):
	t = np.linspace(0, t0,1000)
	xs = odeint(f_ode,x0,t,args=(a,d,b))
	x,y = xs[:,0], xs[:,1]

	return [x,y]

#plotting results
t0 = 1000
t = np.linspace(0,t0,1000)
output1 = solve_ODE(predator_prey_ODE,t,t0,[0.4,0.4],paramstest1[0],paramstest1[1],paramstest1[2])
output2 = solve_ODE(predator_prey_ODE,t,t0,[10,10],paramstest2[0],paramstest2[1],paramstest2[2])


solutions = np.stack((output1, output2), axis = 0) 



x1 = solutions[0][0]
y1 = solutions[0][1]
x2 = solutions[1][0]
y2 = solutions[1][1]


plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(t,x1)
plt.title('predator  [b = 0.15]')
plt.ylabel('Population')

plt.subplot(2, 2, 2)
plt.plot(t,y1)
plt.title('prey  [b = 0.15]')
plt.ylabel('Population')

plt.subplot(2, 2, 3)
plt.plot(t,x2)
plt.title('predator  [b = 0.50]')
plt.ylabel('Population')

plt.subplot(2, 2, 4)
plt.plot(t,y2)
plt.title('prey  [b = 0.50]')
plt.ylabel('Population')


plt.figure(2)
plt.subplot(2,1,1)
plt.plot(x1,y1)
plt.title('predator  [b = 0.15]')
plt.ylabel('y')
plt.xlabel('x')

plt.subplot(2,1,2)
plt.plot(x2,y2)
plt.title('predator  [b = 0.50]')
plt.ylabel('y')
plt.xlabel('x')


plt.figure(3)

ax = plt.axes(projection='3d')
ax.plot3D(t, x1, y1)
plt.title('Stable Periodic Orbit')
ax.set_xlabel('time')
ax.set_ylabel('x1')
ax.set_zlabel('y1')

plt.figure(4)
ax = plt.axes(projection='3d')
ax.plot3D(t, x2, y2)
plt.title('Sink')
ax.set_xlabel('time')
ax.set_ylabel('x2')
ax.set_zlabel('y2')



plt.show()


		




