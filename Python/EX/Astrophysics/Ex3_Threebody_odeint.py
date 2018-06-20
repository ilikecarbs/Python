from numpy import sqrt, linspace
from scipy.integrate import odeint
from pylab import plot, axis, show

# Define the initial conditions for each of the four ODEs
inic = [0,0,0,2]

# Times to evaluate the ODEs. 800 times from 0 to 100 (inclusive).
t = linspace(0, 10, 1000)
#parameters
mu  =   0.2

# The derivative function.
def rhs(z,t):
    x,y,px,py   =   z[0], z[1], z[2], z[3]
    dx1, dx2    =   x + mu, x - 1 + mu
    r1          =   pow(dx1 * dx1 + y * y, 1.5)
    r2          =   pow(dx2 * dx2 + y * y, 1.5)
    dx, dy      =   px + y, py - x
    dpx         =   py - (1-mu) * dx1/r1 - mu * dx2/r2
    dpy         =   -px - (1-mu) * y/r1 - mu * y/r2
    return [ dx, dy, dpx, dpy ]

# Compute the ODE
res = odeint(rhs, inic, t)

# Plot the results
plot(res[:,0], res[:,1])
axis('equal')
show()