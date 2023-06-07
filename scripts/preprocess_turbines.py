import logging
import numpy as np
import xarray as xr

from src.util import create_folder
from src.config import YEARS
from src.config import OUTPUT_DIR
from src.config import SPECIFIC_POWER_RANGE
from src.config import OFFSHORE_TURBINES
from src.load_data import load_turbines_raw
from src.preprocess import estimate_missing
from src.logging_config import setup_logging


from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0088


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


def calc_location_clusters(turbines, min_distance_km=0.5):
    """Calculate a partitioning of locations given in lang/long into clusters using the DBSCAN
    algorithm.

    Runtime: about 10-15 seconds for all turbines.

    Parameters
    ----------
    turbines : xr.DataSet
        as returned by load_turbines()
    min_distance_km : float

    Returns
    -------
    cluster_per_location : xr.DataArray (dims: turbines)
        for each location location the cluster index, -1 for outliers, see
        ``sklearn.cluster.DBSCAN``
    clusters : np.ndarray of shape (M,)
        M is the number of clusters
    cluster_sizes : np.ndarray of shape (M,)
        the size for each cluster

    References
    ----------
    https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size

    """
    locations = turbine_locations(turbines)

    # Parameters for haversine formula
    kms_per_radian = EARTH_RADIUS_KM
    epsilon = min_distance_km / kms_per_radian

    clustering = DBSCAN(eps=epsilon, min_samples=2, algorithm="ball_tree", metric="haversine").fit(
        np.radians(locations)
    )

    cluster_per_location = clustering.labels_
    clusters, cluster_sizes = np.unique(cluster_per_location, return_counts=True)

    cluster_per_location = xr.DataArray(
        cluster_per_location,
        dims="turbines",
        coords={"turbines": turbines.turbines},
        name="cluster_per_location",
    )  # TODO rename to cluster?

    return cluster_per_location, clusters, cluster_sizes


def filter_duplicates(turbines):
    logging.info("Filtering duplicates via geo location...")
    cluster_per_location, clusters, cluster_sizes = calc_location_clusters(
        turbines, min_distance_km=1e-3
    )

    logging.info("Clustering done!")
    turbines_duplicates_ids = [
        turbines_.sortby("uswtdb_version").isel(turbines=0).turbines.values
        for cluster, turbines_ in turbines.groupby(cluster_per_location)
        if cluster != -1
    ]
    turbines_filtered = turbines.drop_sel(turbines=turbines_duplicates_ids)
    return turbines_filtered


def filter_valid_specific_power(turbines):
    """Some turbines contain an unrealistic low or large specific power. Some of them probably have
    a wrong capacity, e.g. 2kW instead of 2000kW. But since there are very few such turbines, we
    simply remove them form the data set.

    This data cleaning step is especially necessary to avoid NaNs when interpolating in power curve
    model."""

    rotor_swept_area = turbines.t_rd**2 * np.pi / 4
    specific_power = turbines.t_cap * 1e3 / rotor_swept_area

    idcs_valid_specific_power = (SPECIFIC_POWER_RANGE[0] <= specific_power) & (
        specific_power <= SPECIFIC_POWER_RANGE[1]
    ) | np.isnan(specific_power)
    turbines = turbines.sel(turbines=idcs_valid_specific_power)

    # this value needs to be changed for other versions of the USWTDB, but the value should always
    # be neglectable small, otherwise we need some kind of correction
    num_filtered_turbines = (~idcs_valid_specific_power).sum()
    assert (
        num_filtered_turbines <= 78
    ), f"unexpected number of turbines with invalid specific power: {num_filtered_turbines}"

    return turbines


def filter_offshore(turbines):
    """There are almost no offshore turbines in the USA. According to a talk by Lucy Pao, there are
    only 7 offshore turbines (with 6 MW capacity each) operating at the moment. The USWTDB seems to
    contain two turbines, which are not covered by the GWA2 (interpolation at the location yields
    NaN). We simply remove these two turbines. This needs to be changed
    """
    # this is just a paranoia check, because trusting on the turbine ids seems to be too dangerous
    for offshore_turbine in OFFSHORE_TURBINES:
        for axis in ("xlong", "ylat"):
            actual = turbines.sel(turbines=offshore_turbine["id"])[axis]
            assert actual == offshore_turbine[axis], (
                f"unexpected turbine location of turbine with turbine ID {offshore_turbine['id']}:"
                f" expected {axis}={offshore_turbine[axis]}, actual {axis}={actual}"
            )

    turbines = turbines.sel(
        turbines=~turbines.turbines.isin([ot["id"] for ot in OFFSHORE_TURBINES])
    )
    return turbines


def preprocess_turbines():
    logging.info("Loading turbine CSV files...")
    turbines_raw = load_turbines_raw()
    turbines_raw = filter_duplicates(turbines_raw)

    output_folder = create_folder("turbines", prefix=OUTPUT_DIR)

    for is_raw in (True, False):
        if is_raw:
            turbines = turbines_raw
        else:
            logging.info("Estimating missing meta data...")
            turbines = estimate_missing(turbines_raw, method="mean")

        turbines = filter_valid_specific_power(turbines)
        turbines = filter_offshore(turbines)

        # we are only using data until 2021, because 2022 is not yet complete
        # note that NaNs should not be excluded here!
        turbines = turbines.sel(turbines=~(turbines.p_year >= YEARS.stop))

        fname_raw = "_raw" if is_raw else ""
        turbines.to_netcdf(
            output_folder / f"turbines{fname_raw}.nc",
        )


if __name__ == "__main__":
    setup_logging()
    preprocess_turbines()
