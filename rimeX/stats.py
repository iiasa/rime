import numpy as np
import scipy.stats as stat
from scipy.optimize import minimize

from itertools import product

import numpy as np
import pandas as pd
import xarray as xa


def fit_dist(values, quantiles=None, dist_name=None, threshold=0.55):
    if quantiles is not None:
        assert np.array(quantiles).tolist() == [.5, .05, .95], str(np.array(quantiles).tolist())
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

        reverse = hi - mid < mid - lo

        if reverse:
            mid, lo, hi = -mid, -hi, -lo

        # this ensures symmetry in the log-transformed quantiles (I wrote it down and solved the equality)
        loc = (mid ** 2 - hi*lo) / (2*mid - lo - hi)

        assert lo - loc > 0
        # It's not too difficult to prove `lo - loc > 0` since we have hi - mid >= mid - lo, and as a result 2*mid - lo - hi <= 0
        # the equality lo - loc > 0 becomes lo * (2*mid - lo - hi) - mid **2 - hi*lo <= 0
        # and suffices to note that lo * (2*mid - lo - hi) - mid **2 - hi*lo = - (mid - lo)**2 which is always < 0

        normdist = fit_dist([np.log(mid - loc), np.log(lo - loc), np.log(hi - loc)], [.5, .05, .95], "norm")
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



def interp_along_axis(x, xp, fp, axis=-1, **kwargs):
    """Interpolation along a specified axis -- loop over subarrays and apply np.interp.

    Parameters
    ----------
    x : 1D array (numpy)
        New x-coordinates
    xp : 1D array (numpy)
        x-coordinates of the data points along the specified axis
    fp : N-D array (numpy)
        Values at the data points
    axis : int, optional
        Axis along which to interpolate, by default -1
    **kwargs
        Additional keyword arguments for np.interp

    Returns
    -------
    N-D array
    """
    # Shape of the output
    out_shape = list(fp.shape)
    out_shape[axis] = np.size(x)

    # Prepare arrays for interpolation
    out = np.empty(out_shape, dtype=fp.dtype)

    # Loop over subarrays of fp in all dimensions but axis
    if axis < 0:
        axis += fp.ndim
    assert axis >= 0 and axis < fp.ndim

    for idx in product(*[range(s) if i != axis else [slice(None)] for i, s in enumerate(fp.shape)]):
        out[idx] = np.interp(x, xp, fp[idx], **kwargs)

    return out


def fast_quantile(a, quantiles, dim=None):
    """Compute quantiles along a specified dimension of a DataArray.
    """
    quantiles = np.asarray(quantiles)
    if np.isscalar(quantiles):
        a = a.reduce(np.percentile, quantiles*100, dim=dim)
    else:
        # "percentile" is orders of magnitude faster than "quantile"
        a_np = np.percentile(a.values, quantiles*100, axis=a.dims.index(dim))
        a = xa.DataArray(a_np,
                                    coords=[quantiles] + [a.coords[c] for c in a.dims if c != dim],
                                    dims=["quantile"] + [c for c in a.dims if c != dim])
    return a


def weighted_quantiles(values, weights, quantiles=0.5, interpolate=True):
    """
    https://stackoverflow.com/a/75321415/2192272

    NOTE: Perhaps surprisingly, weighted_quantiles([0, 1, 2], [.5, .25, .25], .5) returns 0.666 instead of 0.5
    whereas weighted_quantiles([0, 0, 1, 2], [1, 1, 1, 1], .5) and np.quantile([0, 0, 1, 2], .5) do return 0.5

    This should be fixed in the future.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    i = np.argsort(values)
    sorted_weights = weights[i]
    sorted_values = values[i]
    Sn = np.cumsum(sorted_weights)

    if interpolate:
        Pn = (Sn - sorted_weights/2 ) / Sn[-1]
        return np.interp(quantiles, Pn, sorted_values)
    else:
        return sorted_values[np.searchsorted(Sn, np.asarray(quantiles) * Sn[-1])]

def weighted_quantiles_along_axis(values, weights, quantiles=0.5, axis=-1, **kwargs):

    if np.isscalar(quantiles):
        quantiles = np.array([quantiles])
        squeeze = True
    else:
        squeeze = False

    if axis < 0:
        axis += values.ndim

    res = np.empty(values.shape[:axis] + (len(quantiles),) + values.shape[axis+1:])

    for idx in product(*[range(s) if i != axis else [slice(None)] for i, s in enumerate(values.shape)]):
        res[idx] = weighted_quantiles(values[idx], weights, quantiles, **kwargs)

    if squeeze:
        res = res.squeeze(axis)

    return res

def equally_spaced_quantiles(size):
    step = 1/size
    return np.linspace(step/2, 1-step/2, num=size)


def deterministic_resampling(values, size, weights=None, rng=None, axis=0, shuffle=False):
    """ Deterministic resampling of real-numbered values, with interpolation allowed
    """
    quantiles = equally_spaced_quantiles(size)

    if weights is None:
        resampled = np.percentile(values, quantiles*100, axis=axis)

    else:
        if np.ndim(values) > 1:
            resampled = np.stack([weighted_quantiles(np.take(values, i, axis=axis), weights, quantiles) for i in range(values.shape[axis])], axis=axis)
        else:
            resampled = weighted_quantiles(values, weights, quantiles)

    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(resampled)

    # give back its initial shape
    if axis is not None and axis > 0:
        resampled = resampled.swapaxes(axis, 0)

    return resampled
