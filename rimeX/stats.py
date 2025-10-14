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


def fast_quantile(a, quantiles, dim=None, skipna=False):
    """Compute quantiles along a specified dimension of a DataArray.
    """
    func = np.nanpercentile if skipna else np.percentile
    quantiles = np.asarray(quantiles)
    if np.isscalar(quantiles):
        a = a.reduce(func, quantiles*100, dim=dim)
    else:
        # "percentile" is orders of magnitude faster than "quantile"
        a_np = func(a.values, quantiles*100, axis=a.dims.index(dim))
        a = xa.DataArray(a_np,
                                    coords=[quantiles] + [a.coords[c] for c in a.dims if c != dim],
                                    dims=["quantile"] + [c for c in a.dims if c != dim])
    return a


def weighted_quantiles(values, weights, quantiles=0.5, interpolate=True, skipna=False):
    """Compute weighted quantiles with proper interpolation.
    
    This implementation ensures consistency with np.quantile when all weights
    are equal, while handling weighted cases appropriately. It uses a cumulative
    weight mapping that places samples at positions corresponding to their
    cumulative weight midpoints, then maps quantiles to these positions.
    
    Parameters
    ----------
    values : array-like
        Input values
    weights : array-like
        Weights corresponding to values
    quantiles : float or array-like, default 0.5
        Quantile(s) to compute, must be between 0 and 1
    interpolate : bool, default True
        If True, use linear interpolation between values
    skipna : bool, default False
        If True, ignore NaN values
    
    Returns
    -------
    float or ndarray
        Computed quantile(s)
    
    References
    ----------
    Original implementation: https://stackoverflow.com/a/75321415/2192272
    
    Notes
    -----
    For equal weights, this reduces to NumPy's linear interpolation (type 7).
    The key is to map each value to a fractional position based on cumulative
    weights, creating positions at: (cumsum - weight/2) / total_weight * (n - 1)
    where n is the number of values. This ensures the first value is at position 0
    and the last is at position n-1, matching NumPy's convention.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    quantiles_array = np.asarray(quantiles)
    scalar_quantile = quantiles_array.ndim == 0
    
    if skipna:
        mask = ~np.isnan(values)
        values = values[mask]
        weights = weights[mask]
        if len(values) == 0:
            return np.full_like(quantiles_array, np.nan) if not scalar_quantile else np.nan
    
    # Sort values and weights
    i = np.argsort(values)
    sorted_values = values[i]
    sorted_weights = weights[i]
    
    # Compute cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    
    # Map cumulative weights to positions: each value is at the midpoint of its weight interval
    # This creates positions from 0 to (n-1), matching NumPy's convention
    n = len(sorted_values)
    if n == 1:
        # Special case: single value
        return np.full_like(quantiles_array, sorted_values[0]) if not scalar_quantile else sorted_values[0]
    
    # Positions range from 0 to n-1
    # For the i-th value, its position represents where it falls in the weighted distribution
    # Formula: (cumulative weight before this point) / (total weight - last weight) * (n - 1)
    # This ensures first value is at position 0 and last value is at position n-1
    positions = (cum_weights - sorted_weights) / (total_weight - sorted_weights[-1]) * (n - 1)
    
    if interpolate:
        # Map quantiles to positions in the range [0, n-1]
        target_positions = quantiles_array * (n - 1)
        result = np.interp(target_positions, positions, sorted_values)
    else:
        # For non-interpolating mode, find the value at or after the target position
        target_positions = quantiles_array * (n - 1)
        result = np.array([sorted_values[np.searchsorted(positions, pos, side='right')] 
                          if pos < positions[-1] else sorted_values[-1] 
                          for pos in np.atleast_1d(target_positions)])
    
    return result.item() if scalar_quantile else result

def weighted_quantiles_along_axis(values, weights, quantiles=0.5, axis=-1, skipna=False, **kwargs):

    if np.isscalar(quantiles):
        quantiles = np.array([quantiles])
        squeeze = True
    else:
        squeeze = False

    if axis < 0:
        axis += values.ndim

    res = np.empty(values.shape[:axis] + (len(quantiles),) + values.shape[axis+1:])

    for idx in product(*[range(s) if i != axis else [slice(None)] for i, s in enumerate(values.shape)]):
        res[idx] = weighted_quantiles(values[idx], weights, quantiles, skipna=skipna, **kwargs)

    if squeeze:
        res = res.squeeze(axis)

    return res


def fast_weighted_quantile(a, quantiles, weights=None, dim=None, skipna=False):
    """Compute quantiles along a specified dimension of a DataArray.
    """
    if weights is None:
        return fast_quantile(a, quantiles, dim=dim, skipna=skipna)

    quantiles = np.asarray(quantiles)

    if np.isscalar(quantiles):
        a = a.reduce(weighted_quantiles_along_axis, weights, quantiles, dim=dim, skipna=skipna)

    else:
        a_np = weighted_quantiles_along_axis(a.values, weights, quantiles, axis=a.dims.index(dim), skipna=skipna)
        a = xa.DataArray(a_np,
                                    coords=[a.coords[c] if c != dim else quantiles for c in a.dims],
                                    dims=[c if c != dim else "quantile" for c in a.dims])
    return a


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
