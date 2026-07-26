import numpy as np
from scipy import stats


class BetaProposal:
    """Importance sampling proposal using independent Beta distributions.

    Each parameter is sampled via Beta(alpha, beta) scaled to [lo, hi].
    Weights are p(theta)/q(theta) where p is the uniform prior.
    """

    def __init__(self, params):
        """
        params: list of dicts, each with keys:
            'name', 'lo', 'hi', 'alpha', 'beta'
        """
        self.params = params

    def sample(self, batch_size):
        """Returns (theta, weights) as numpy arrays."""
        n_params = len(self.params)
        theta = np.zeros((batch_size, n_params))
        weights = np.ones(batch_size)

        for i, p in enumerate(self.params):
            lo, hi = p['lo'], p['hi']
            a, b = p['alpha'], p['beta']

            tilde = np.random.beta(a, b, size=batch_size)
            theta[:, i] = lo + (hi - lo) * tilde

            # w_i = p_i / q_i = (1/(hi-lo)) / (Beta_pdf(tilde) / (hi-lo)) = 1/Beta_pdf(tilde)
            q = stats.beta.pdf(tilde, a, b)
            weights *= 1.0 / q

        return theta, weights
