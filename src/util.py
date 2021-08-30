import os
import numpy as np
import xarray as xr

from src.config import OUTPUT_DIR
from src.config import INTERIM_DIR


def turbine_locations(turbines):
    """Extract a numpy array of turbine locations from the turbine DataArray.

    Parameters
    ----------
    turbines : xr.DataArray
        as returned by load_turbines()

    Returns
    -------
    np.ndarray with shape (N, 2)

    """
    turbines_np = np.column_stack(
        [
            turbines.ylat.values,
            turbines.xlong.values,
        ]
    )

    return turbines_np


def centers(edges):
    return 0.5 * (edges[1:] + edges[:-1])


# Version for xr.DataArray:
# def centers(edges, dim='bin_edges', new_dim='bins'):
#     centers_ = 0.5 * (edges.sel({dim: slice(1, None)}) + edges.sel({dim: slice(None, -1)}))
#     return centers_.rename({dim: new_dim})


def edges_to_center(edges):
    if len(edges) <= 1:
        raise ValueError("edges must be of length > 1")
    distances = np.diff(edges)
    if np.any(distances != distances[0]):
        raise ValueError("works only for equidistant edges")

    return edges[:-1] + (edges[1] - edges[0]) / 2.0


def nanratio(d):
    return np.isnan(d).sum() / len(d)


def notnan(d):
    return d[~np.isnan(d)]


def mean_per_bin(d, by, aggreagation="mean"):
    """Calculate the mean for each bin in `d` when grouped by `by`."""
    assert isinstance(d, xr.DataArray), "'d' must be a xarray.DataArray object"  # for grouping
    assert isinstance(by, xr.DataArray), "'by' must be a xarray.DataArray object"  # needs a name
    assert ~np.any(np.isnan(by)), "'by' must not contain NaN values"
    assert ~np.all(np.isnan(d)), "all values in d are NaN, can't calculate mean"

    # replace bins without value with overall aggregation value, i.e. min/mean/max for all values
    has_bin_value = (~np.isnan(d)).groupby(by).sum() == 0
    if np.any(has_bin_value):
        # this if could be removed, it's a noop if there is no bin without value
        # only difference should be: d is copied
        overall_aggregated = getattr(d, aggreagation)()
        bins_without_value = has_bin_value[by.name][has_bin_value]
        d = d.copy()
        d[by.isin(bins_without_value)] = overall_aggregated

    aggreagation_fct = getattr(d.groupby(by), aggreagation)
    mean_per_bin = aggreagation_fct()
    return mean_per_bin.sel({by.name: by}).drop_vars(by.name)


def replace_nans_binmean(d, by, aggreagation="mean"):
    """"Replace each NaN value in `d` by the mean in each bin when grouping by `by`."""
    return d.where(~np.isnan(d), mean_per_bin(d, by, aggreagation=aggreagation))


def choose_samples(*objs, num_samples, dim):
    """Pick random samples from xarray objects.

    Parameters
    ----------
    objs : xr.DataArray or xr.Dataset
    num_samples : int
    dim : str

    Returns
    -------

    """
    assert np.all(objs[0].sizes[dim] == np.array([obj.sizes[dim] for obj in objs])), (
        f"objs must have same length in dimension {dim}, "
        f"sizes are: {list(obj.sizes[dim] for obj in objs)}"
    )

    np.random.seed(42)
    idcs = np.random.choice(objs[0].sizes[dim], size=num_samples)
    idcs.sort()
    return (obj.isel({dim: idcs}) for obj in objs)


def create_folder(path, prefix=INTERIM_DIR):
    if prefix is not None:
        path = prefix / path
    os.makedirs(path, exist_ok=True)
    return path


def filter2010(data):
    return data.sel(time=data.time.dt.year >= 2010)


def write_data_value(name, value):
    with open(OUTPUT_DIR / "data-values" / name, "w") as f:
        f.write(value + "\n")


def calc_abs_slope(data):
    """Slope of input data, i.e. change per time step using regression line."""
    return np.abs(np.polyfit(range(len(data)), data, deg=1)[0])
