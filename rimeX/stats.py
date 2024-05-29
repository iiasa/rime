import numpy as np
import scipy.stats as stat

def fit_dist(values, quants=None, dist_name=None, threshold=0.55):
    if quants is not None:
        assert quants == [50, 5, 95]
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

        # this seems to be true, but I haven't proved it
        assert lo - loc > 0

        normdist = fit_dist([np.log(mid - loc), np.log(lo - loc), np.log(hi - loc)], [50, 5, 95], "norm")
        mu, sigma = normdist.args
        dist = stat.lognorm(sigma, loc, np.exp(mu))

        if reverse:
            dist = ReverseDist(dist)

        return dist

    else:
        raise NotImplementedError(dist_name)


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
