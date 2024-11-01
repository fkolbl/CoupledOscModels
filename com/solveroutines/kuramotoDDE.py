from jitcdde import jitcdde, y, t
import numpy as np
from symengine import sin

def __solve_Kuramoto_DDE(
    N,
    omega,
    c,
    A,
    tau,
    t,
    dt
    ):
    """
    Routine to solve the delayed Kuramoto system of differential equation.
    Problem of the form:

    .. math::
     \dot{y}(i,t) = \omega + c/(N-1) \sum_{j=0}^{N-1} \sin(y(j,t-tau_{i,j}) - y(i)) \quad \text{for} \quad i=0,\ldots,N-1,  


    """
    RESULTS = {}
    # check argument consistency

    # Define the delayed Kuramoto equations
    def kuramotos():
        for i in range(N):
            yield omega + c/(N-1)*sum(
                        sin(y(j, t-tau[i,j]) - y(i))
                        for j in range(N)
                        if A[j,i]
                    )

    # integrate
    I = jitcdde(kuramotos, n=N, verbose=False, delays=tau.flatten())
    I.set_integration_parameters(rtol=0, atol=1e-5)

    I.constant_past(np.random.uniform(0, 2 * np.pi, N), time=0.0)
    # we integrate blindly with a maximum time step of 0.1 up to the maximal delay to ensure that initial discontinuities have smoothened out.
    I.integrate_blindly(np.max(tau), dt/2)
    # integrate
    Y = []
    time = I.t + np.arange(0, t, dt)
    for time in time:
        Y.append(I.integrate(time) % (2*np.pi))

    # get results
    RESULTS['t'] = time
    RESULTS['solution'] = np.asarray(Y)
    return RESULTS


