import numpy as np
import scipy.stats as stat

def fit_dist(values, quants=None, dist_name=None):
    if quants is not None:
        assert quants == [50, 5, 95]
    mid, lo, hi = values

    if dist_name == "auto":
        if abs(hi-mid)/abs(hi-lo) > 0.55:
            dist_name = "lognorm"
        else:
            dist_name = "norm"

    if dist_name == "norm":
        return stat.norm(mid, ((hi-mid)+(mid-lo))/2 / stat.norm.ppf(.95))

    if dist_name == "lognorm":
        import numpy as np
        if lo < 0:
            loc = lo - (hi-lo)/10  # add a small shift away from 0
        else:
            loc = 0

        normdist = fit_dist([np.log(mid - loc), np.log(lo - loc), np.log(hi - loc)], [50, 5, 95], "norm")
        mu, sigma = normdist.args
        return stat.lognorm(sigma, loc, np.exp(mu))

    else:
        raise NotImplementedError(dist_name)
