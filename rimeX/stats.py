import numpy as np
import scipy.stats as stat
from scipy.optimize import minimize

def fit_dist(values, quantiles=None, dist_name=None, threshold=0.55):
    if quantiles is not None:
        assert quantiles == [.5, .05, .95]
    mid, lo, hi = values

    if dist_name == "auto":
        if abs(hi-mid)/abs(hi-lo) > threshold:
            dist_name = "lognorm"
        elif abs(mid-lo)/abs(hi-lo) > threshold:
            dist_name = "lognorm"
        else:
            dist_name = "norm"

    if dist_name == "norm":
        return stat.norm(mid, ((hi-mid)+(mid-lo))/2 / stat.norm.ppf(.95))

    if dist_name == "lognorm":
        import numpy as np

        reverse = hi - mid < mid - lo

        if reverse:
            mid, lo, hi = -mid, -hi, -lo

        # this ensures symmetry in the log-transformed quantiles (I wrote it down and solved the equality)
        loc = (mid ** 2 - hi*lo) / (2*mid - lo - hi)

        assert lo - loc > 0
        # It's not too difficult to prove `lo - loc > 0` since we have hi - mid >= mid - lo, and as a result 2*mid - lo - hi <= 0
        # the equality lo - loc > 0 becomes lo * (2*mid - lo - hi) - mid **2 - hi*lo <= 0
        # and suffices to note that lo * (2*mid - lo - hi) - mid **2 - hi*lo = - (mid - lo)**2 which is always < 0

        normdist = fit_dist([np.log(mid - loc), np.log(lo - loc), np.log(hi - loc)], [50, 5, 95], "norm")
        mu, sigma = normdist.args
        dist = stat.lognorm(sigma, loc, np.exp(mu))

        if reverse:
            dist = ReverseDist(dist)

        return dist

    else:
        raise NotImplementedError(dist_name)


def fit_dist_minimize(data_points, quantiles, dist):
    # Define the objective function
    def objective(params):
        # dist_quantiles = dist.ppf(*params, np.array(quantiles))
        dist_quantiles = dist(*params).ppf(quantiles)
        if np.any(np.isnan(dist_quantiles)):
            return np.inf
        # print("params", params, "dist_quantiles", dist_quantiles)
        return np.sum((dist_quantiles - data_points) ** 2)

    # Initial guess for the parameters
    initial_params = dist.fit(data_points)
    # print("initial_params", initial_params)

    # Optimize the parameters
    # result = minimize(objective, initial_params, method='L-BFGS-B')
    result = minimize(objective, initial_params)

    if not result.success:
        raise RuntimeError("Optimization failed")

    # Return the fitted distribution
    return dist(*result.x)


def repr_dist(dist):
    if isinstance(dist, ReverseDist):
        return f"ReverseDist({repr_dist(dist.dist)})"

    return f"{dist.dist.name}({','.join([str(r) for r in dist.args])})"

class ReverseDist:
    def __init__(self, dist):
        self.dist = dist
        self.args = ('reverse of',) + self.dist.args
        self.name = f"revserse {dist.dist.name}"

    def ppf(self, q):
        return -self.dist.ppf(q)[::-1]
