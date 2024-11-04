import com.solveroutines as slv
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colormaps

N = 100
omega = np.ones(N)
c = 42
q = 0.05
A = np.random.choice([1, 0], size=(N,N), p=[q,1-q] )
tau = np.random.uniform(np.pi/5, 2*np.pi, size=(N,N) )
#tau = np.zeros((N,N))

t_sim = 400
dt = 0.2

fig = plt.figure(figsize=(14, 8))  # fig, ax = plt.subplots(layout='constrained', figsize=(10,7))
# create grid for different subplots
spec = gridspec.GridSpec(ncols=2, nrows=1,
                         wspace=0.15,
                         hspace=0.05,
)
ax_adj = fig.add_subplot(spec[0, 0])
ax_delay = fig.add_subplot(spec[0, 1])

ax_adj.pcolormesh(A)
ax_adj.set_title('Adjacency Matrix')
ax_delay.pcolormesh(tau)
ax_delay.set_title('Delay Matrix')

res = slv.__solve_Kuramoto_DDE(
    N,
    omega,
    c,
    A,
    tau,
    t_sim,
    dt
)

plt.figure()

print(res['theta'].shape)
print(res['t'].shape)
plt.pcolormesh(res['t'], np.arange(N), res['theta'], cmap='twilight')
plt.show()
