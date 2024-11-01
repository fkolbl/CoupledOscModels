from jitcdde import jitcdde, y, t
from numpy import pi, arange, random, max
import numpy as np
from symengine import sin

import matplotlib.pyplot as plt
from matplotlib import colormaps

n = 100
ω = 1
c = 42
q = 0.05
A = random.choice( [1,0], size=(n,n), p=[q,1-q] )
#A = np.ones((n,n))
τ = random.uniform( pi/5, 2*pi, size=(n,n) )

tsim = 400
dt = 0.2

def model(n, ω, c, A, τ, tsim, dt):
    def kuramotos():
        for i in range(n):
            yield ω + c/(n-1)*sum(
                        sin(y(j,t-τ[i,j])-y(i))
                        for j in range(n)
                        if A[j,i]
                        )

    I = jitcdde(kuramotos,n=n,verbose=False,delays=τ.flatten())
    I.set_integration_parameters(rtol=0,atol=1e-5)

    I.constant_past( random.uniform(0,2*pi,n), time=0.0 )
    I.integrate_blindly( max(τ) , 0.1 )

    t_vector = I.t + arange(0, tsim, dt)
    solution_vector =  np.full([len(t_vector), n], np.nan)

    for k, time in enumerate(t_vector):
        #print(*I.integrate(time) % (2*pi))
        solution_vector[k, :] = np.remainder(np.asarray(I.integrate(time)), 2*np.pi)

    return solution_vector.T, t_vector

theta, t = model(n, ω, c, A, τ, tsim, dt)

plt.figure()

print(theta.shape)
print(t.shape)
plt.pcolormesh(t, np.arange(n), theta, cmap='twilight')
plt.show()
