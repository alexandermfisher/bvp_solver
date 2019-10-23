import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math
import sys

def euler_step(f,x0,t0,h):
    x1 = x0 +h*f(x0,t0)
    return x1

def solve_to(f,x0,t0,tf,hmax):
	t_current = t0
	x_current = x0
	while t_current < tf:
		h = min(hmax,tf - t_current)
		x_current = euler_step(f,x_current,t_current,h)
		t_current = t_current + h

	return x_current


def rk4(f, t0, x0, tf, n):
    vec_t = [0] * (n + 1)
    vec_x = [0] * (n + 1)
    h = (tf - t0) / float(n)
    vec_t[0] = t = t0
    vec_x[0] = x = x0
    for i in range(1, n + 1):
        k1 = h * f(t, x)
        k2 = h * f(t + 0.5 * h, x + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, x + 0.5 * k2)
        k4 = h * f(t + h, x + k3)
        vec_t[i] = t = t0 + i * h
        vec_x[i] = x = x + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return vec_t, vec_x	



def solve_ode_euler(f,x0,t0,tf):
    h_vec = []
    err_vec = []
    for n in range(1,8):
        h = 10**-(n+1)
        e_numerical =  solve_to(f,x0,t0,tf,h)
        e = math.exp(tf)
        err = abs(e_numerical - e)
        print(f"step size: {h} error: {err}")
        h_vec.append(h)
        err_vec.append(err) 

    plt.figure(figsize=(5, 5), dpi=70)
    xvals = h_vec
    yvals = err_vec
    plt.loglog(xvals,yvals)
    plt.show() 
    return 	


'''
def solve_ode_runge(f, t0, x0, tf, n)


vx, vy = rk4(f, 0, 1, 10, 100)
for x, y in list(zip(vx, vy))[::10]:
    print("%4.1f %10.5f %+12.4e" % (x, y, y - (4 + x * x)**2 / 16))
'''


def f(x,t):
    return x
t0 = 0
x0 = 1
tf = 1

solve_ode_euler(f,x0,t0,tf)






    
