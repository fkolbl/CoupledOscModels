from jitcdde import jitcdde, y, t
from numpy import pi, arange, random, max
import numpy as np
from symengine import sin

import matplotlib.pyplot as plt
from matplotlib import colormaps

N = 100
omega = np.ones(N)
c = 90#42
q = 0.05
A = random.choice([1,0], size=(N,N), p=[q,1-q] )
#A = np.ones((n,n))
tau = random.uniform( pi/5, 2*pi, size=(N,N) )

tsim = 400
dt = 0.2

def model(N, omega, c, A, tau, tsim, dt):
    def kuramotos():
        for i in range(N):
            yield omega[i] + c/(N-1)*sum(
                        sin(y(j,t-tau[i,j])-y(i))
                        for j in range(N)
                        if A[j,i]
                        )

    I = jitcdde(kuramotos, n=N, verbose=False, delays=tau.flatten())
    I.set_integration_parameters(rtol=0, atol=1e-5)

    I.constant_past(random.uniform(0, 2*pi, N), time=0.0)
    I.integrate_blindly(max(tau), dt/2)

    t_vector = I.t + arange(0, tsim, dt)
    solution_vector = np.full([len(t_vector), N], np.nan)

    for k, time in enumerate(t_vector):
        solution_vector[k, :] = np.remainder(np.asarray(I.integrate(time)), 2*np.pi)

    return solution_vector.T, t_vector

theta, t = model(N, omega, c, A, tau, tsim, dt)

plt.figure()

print(theta.shape)
print(t.shape)
plt.pcolormesh(t, np.arange(N), theta, cmap='twilight')
plt.show()
