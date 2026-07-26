import numpy as np
import matplotlib.pyplot as plt



def leapfrog(q0: np.ndarray, p0: np.ndarray, q_dot: callable, p_dot: callable, dt=0.01, n_steps=1000):
    """
    Leapfrog integrator for Hamiltonian dynamics with a separable Hamiltonian H(q, p) = U(q) + K(p).
    """

    q = np.array(q0, copy=True)
    p = np.array(p0, copy=True)

    for i in range(n_steps):
        p_half = p + 0.5 * p_dot(q) * dt
        q_new = q + q_dot(p_half) * dt
        p_new = p_half + 0.5 * p_dot(q_new) * dt

        q = q_new
        p = p_new

    return q, p

def leapfrog_history(q0: np.ndarray, p0: np.ndarray, q_dot: callable, p_dot: callable, dt=0.01, n_steps=1000):
    """
    Leapfrog integrator for Hamiltonian dynamics with a separable Hamiltonian H(q, p) = U(q) + K(p).
    Returns the history of positions and momenta.
    """

    q = np.array(q0, copy=True)
    p = np.array(p0, copy=True)

    q_history = [q.copy()]
    p_history = [p.copy()]

    for i in range(n_steps):
        p_half = p + 0.5 * p_dot(q) * dt
        q_new = q + q_dot(p_half) * dt
        p_new = p_half + 0.5 * p_dot(q_new) * dt

        q = q_new
        p = p_new

        q_history.append(q.copy())
        p_history.append(p.copy())

    return np.array(q_history), np.array(p_history)