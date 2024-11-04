import jitcode as code
import scipy.integrate as solvers
from numba import jit


def __solve_Kuramoto_ODE_scipy():
    # define ODE system
    @jit(nopython=True)
    def F(X, t, Omegas, K, N, A):
        dXdt = np.zeros(N)
        for k in range(N):
            # compute coupling term
            #coupling = 0
            #for i in range(N):
            #    if k != i:
            #        coupling += np.sin(X[i] - X[k])
            coupling = np.sum(np.sin(X - X[k]))
            # derivative
            dXdt[k] = Omegas[k] + (K / N) * coupling
        return dXdt

    return True
