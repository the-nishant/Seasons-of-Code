import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import math
from scipy import misc

def fn_plot1d(fn, x_min, x_max, filename):
    step = (x_max-x_min)/1000 ;
    x = np.arange(x_min,x_max,step)
    y = np.vectorize(fn)
    plt.plot(x,y(x))
    plt.xlabel('x')
    plt.ylabel(f'{fn.__name__}(x)')
    plt.title(f'{fn.__name__}(x) vs x')
    plt.savefig(filename)
    plt.close()

def nth_derivative_plotter(fn, n, x_min, x_max, filename):
    step = (x_max-x_min)/1000 ;
    x = np.arange(x_min,x_max,step)
    y = np.vectorize(lambda x: misc.derivative(fn,x,dx=step,n=n,order=2*n+1))
    plt.plot(x,y(x))
    plt.xlabel('x')
    plt.ylabel(f'{n}th derivative of {fn.__name__}(x)')
    plt.title(f'{n}th derivative of {fn.__name__}(x) vs x')
    plt.savefig(filename)
    plt.close()



def fn_plot2d(fn, x_min, x_max, y_min, y_max, filename):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xstep = (x_max - x_min)/1000 ;
    ystep = (y_max - y_min)/1000 ;
    x = np.arange(x_min,x_max,xstep)
    y = np.arange(y_min,y_max,ystep)
    x,y = np.meshgrid(x,y)
    z = np.vectorize(fn)
    ax.plot_surface(x,y,z(x,y))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(f'{fn.__name__}(x,y)')
    plt.title(f'{fn.__name__}(x,y) vs x,y')
    plt.savefig(filename)
    plt.close()


def b(x):
    return g(abs(x))

def g(x):
    return h(2-x)/(h(2-x)+h(x-1))

def h(x):
    if(x>0):
        return math.exp(-1/(x*x))
    return 0

def twodsinc(x,y):
    arg = math.sqrt(x*x+y*y)
    if(arg>0):
        return math.sin(arg)/arg 
    return 1

fn_plot1d(b,-2,2,'fn1plot.png')
fn_plot2d(twodsinc,-1.5*math.pi,1.5*math.pi,-1.5*math.pi,1.5*math.pi,'fn2plot.png')
for i in range(1,11):
    nth_derivative_plotter(b,i,-2,2,f'bd_{i}.png')

