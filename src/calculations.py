import logging
import dask as da
import numpy as np
import xarray as xr

from dask.diagnostics import ProgressBar

from src.util import centers
from src.constants import AIR_DENSITY_RHO
from src.config import CHUNK_SIZE_TURBINES
from src.config import CHUNK_SIZE_TIME


def calc_wind_speed_at_turbines(wind_velocity, turbines, height=100.0):
    """Interpolate wind velocity at turbine locations and calculate speed from velocity.
    Interpolate height if parameter height is given.

    Parameters
    ----------
    wind_velocity : xr.DataSet (dims: xlong, ylat)
        downloaded ERA5 data, read from NetCDF
    turbines : xr.DataSet
        see load_turbines()
    height : float or None
        interpolate or extrapolate wind speed at given height if != 100. If None, will use
        turbines.t_hh for each turbine. Note: t_hh contains NaNs, i.e. result may contain NaNs!

    Returns
    -------
    wind_speed: xr.DataArray (dims: turbines, time)
        may contain NaNs! (see above)

    """
    # XXX no idea what the right choice is here, but at least for calc_bias_correction() the value
    # False seems to require less RAM and True more than we have.
    # https://docs.dask.org/en/stable/array-slicing.html
    # https://docs.dask.org/en/stable/array-chunks.html#array-chunks
    with da.config.set(**{"array.slicing.split_large_chunks": False}):
        # interpolate at turbine locations
        wind_velocity_at_turbines = wind_velocity.interp(
            longitude=xr.DataArray(turbines.xlong.values, dims="turbines"),
            latitude=xr.DataArray(turbines.ylat.values, dims="turbines"),
            method="linear",
            kwargs={"bounds_error": True},
        )

    # velocity --> speed
    wind_speed100 = (
        wind_velocity_at_turbines.u100**2 + wind_velocity_at_turbines.v100**2
    ) ** 0.5

    height_attr = height

    if height is None:
        # not very nice, because suddenly height is a vector!
        height = turbines.t_hh
        height_attr = 0.0  # ugly special value for extrapolation at hub height
    else:
        # we want wind speed to be NaN if hub height is missing to have a consistent NaN scaling
        height = height * (turbines.t_hh - turbines.t_hh + 1)

    wind_speed10 = (wind_velocity_at_turbines.u10**2 + wind_velocity_at_turbines.v10**2) ** 0.5

    powerlaw_alpha = np.log10(wind_speed100 / wind_speed10)
    wind_speed = wind_speed100 * (height / 100.0) ** powerlaw_alpha

    # None refers to turbine height
    wind_speed.attrs["height"] = height_attr

    return wind_speed


def calc_wind_speed_at_turbines_gwa(wind_speed_gwa, turbines):
    # TODO merge this function with function above simply by renaming x/y variables
    wind_speed_at_turbines_gwa = wind_speed_gwa.interp(
        x=xr.DataArray(turbines.xlong.values, dims="turbines"),
        y=xr.DataArray(turbines.ylat.values, dims="turbines"),
        method="linear",
        kwargs={"bounds_error": True},
    )

    return wind_speed_at_turbines_gwa


def calc_bias_correction(turbines, wind_velocity_era5, wind_speed_gwa, height):
    def mean_era5(turbines):
        with da.config.set(**{"array.slicing.split_large_chunks": False}):
            wind_speed_at_turbines_era5 = calc_wind_speed_at_turbines(
                wind_velocity_era5, turbines, height=height
            )
        return wind_speed_at_turbines_era5.mean(dim="time")

    logging.info("Compute mean ERA5...")

    wind_speed_at_turbines_era5_mean = xr.map_blocks(
        lambda turbines: mean_era5(turbines).compute(),
        turbines,
        template=mean_era5(turbines),
    )

    with ProgressBar():
        wind_speed_at_turbines_era5_mean = wind_speed_at_turbines_era5_mean.compute()

    logging.info("Interpolate GWA at turbines...")
    wind_speed_at_turbines_gwa = calc_wind_speed_at_turbines_gwa(wind_speed_gwa, turbines)

    return wind_speed_at_turbines_gwa / wind_speed_at_turbines_era5_mean


def calc_is_built(turbines, time, include_commission_year=None):
    """

    Parameters
    ----------
    turbines : xr.DataSet
    time : xr.DataArray
    include_commission_year : boolean or None
        True to assume that turbine was already operating in year p_year, False to assume it did
        not generate electricity in p_year, None to let the turbine fade in linearly from the first
        day of the year until the last one

    Returns
    -------
    xr.DataArray
        dims

    """
    # we need chunked versions, otherwise this would require 90GB of RAM
    p_year = turbines.p_year

    if include_commission_year is not None:
        p_year = turbines.p_year.chunk({"turbines": CHUNK_SIZE_TURBINES})
        year = time.dt.year.chunk({"time": CHUNK_SIZE_TIME})

        if include_commission_year is True:
            is_built = (p_year <= year).astype(float)
        elif include_commission_year is False:
            is_built = (p_year < time.dt.year).astype(float)
        else:
            raise ValueError(
                f"invalid value for include_commission_year: {include_commission_year}"
            )
    else:
        is_yearly_aggregated = len(np.unique(time.dt.year)) == len(time)

        if is_yearly_aggregated:
            assert (
                np.all(time.dt.dayofyear == 1)
                and np.all(time.dt.hour == 0)
                and np.all(time.dt.minute == 0)
                and np.all(time.dt.second == 0)
            ), "yearly aggregation of 'time' passed, but not the first hour of the year"
            is_built = (p_year < time.dt.year).astype(float)
            is_built = is_built.where(p_year != time.dt.year, 0.5)
        else:
            assert not (
                np.all(time.dt.dayofyear == 1)
                and np.all(time.dt.hour == 0)
                and np.all(time.dt.minute == 0)
                and np.all(time.dt.second == 0)
            ), "not yearly parameter 'time' passed, but only first hour of the year"
            # beginning of the year as np.datetime64
            p_year_date = p_year.astype(int).astype(str).astype(np.datetime64)
            is_leap_year = p_year_date.dt.is_leap_year.astype(float)

            p_year_date = p_year_date.chunk({"turbines": CHUNK_SIZE_TURBINES})
            time = time.chunk({"time": CHUNK_SIZE_TIME})
            is_leap_year.chunk({"turbines": CHUNK_SIZE_TURBINES})

            # this is where the broadcasting magic takes place
            nanosecs_of_year = (time - p_year_date).astype(float)

            proportion_of_year = nanosecs_of_year / (365 + is_leap_year)
            proportion_of_year = proportion_of_year / (24 * 60 * 60 * 1e9)

            proportion_of_year = proportion_of_year.transpose()
            is_built = proportion_of_year.clip(0, 1)

    return is_built


def calc_rotor_swept_area(turbines, time):
    """Calculate the total rotor swept area per time for all turbines installed at this point in
    time.

    Parameters
    ----------
    turbines : xr.DataSet
        see load_turbines()
    time: xr.DataArray
        a list of years as time stamps

    Returns
    -------
    xr.DataArray
        rotor swept area in m²

    """
    assert np.all(~np.isnan(turbines.t_rd)) and np.all(
        ~np.isnan(turbines.p_year)
    ), "turbines contain NaN values, not allowed here!"

    is_built = calc_is_built(turbines, time)
    rotor_swept_area = (turbines.t_rd) ** 2 * is_built
    rotor_swept_area = rotor_swept_area.sum(dim="turbines") / 4 * np.pi

    return rotor_swept_area


def calc_bounding_box_usa(turbines, extension=1.0):
    # Bounding box can be also manually selected:
    #   https://boundingbox.klokantech.com/

    # assert -180 <= long <= 180, -90 <= lat <= 90
    # FIXME need +180 modulo 360!
    north = turbines.ylat.values.max() + extension
    west = turbines.xlong.values.min() - extension
    south = turbines.ylat.values.min() - extension
    east = turbines.xlong.values.max() + extension

    return north, west, south, east


def calc_irena_correction_factor(turbines, capacity_irena):
    """Assuming that IRENA is correct, but USWTDB is missing turbines or has too many, because
    decommission dates are not always known, this is the factor which corrects time series which
    are proportional to the total turbine capacity."""
    # turbines.t_cap is in KW, IRENA is in MW
    capacity_uswtdb = turbines.groupby("p_year").sum().t_cap.cumsum() * 1e-3
    return capacity_irena / capacity_uswtdb


def calc_capacity_per_year(turbines):
    """Returns Capacity per year in MW. The commissioning year is included."""
    # FIXME no capacity weighting and maybe an off-by-1 error, right? How is the commissioning year
    # counted? How should it be counted?
    return turbines.groupby("p_year").sum().t_cap.cumsum() * 1e-3


def calc_wind_speed_distribution(turbines, wind_speed, bins=100, chunk_size=800):
    """Calculate a distribution of wind speeds for each turbine location, by computing the
    histogram over the whole time span given in ``wind_speed``.

    Parameters
    ----------
    turbines
    wind_speed
    chunk_size : int
        will split turbines into chunks of this size

    Returns
    -------
    xr.Dataset

    Note
    ----

    To improve performance, turbines are split into chunks of given size and wind speed is first
    loaded into RAM before calculating the histogram. The 3rd party library xhistogram might help
    to do this even more efficiently. Also parallelization of the loop should be possible, but
    not really necessary for our purpose.

    """
    num_turbines = turbines.sizes["turbines"]

    wind_probablities = np.empty((num_turbines, bins))
    wind_speed_bins_edges = np.empty((num_turbines, bins + 1))

    chunks_idcs = range(int(np.ceil(num_turbines / chunk_size)))

    for chunk_idx in chunks_idcs:
        logging.debug(f"Progress: {100 * chunk_idx / (num_turbines / chunk_size)}%")
        chunk = slice(chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size)

        turbines_chunk = turbines.isel(turbines=chunk)
        wind_speed_chunk = wind_speed.sel(turbines=turbines_chunk.turbines).load()

        for i, turbine_id in enumerate(turbines_chunk.turbines):
            wind_probablities_, wind_speed_bins_edges_ = np.histogram(
                wind_speed_chunk.sel(turbines=turbine_id), bins=bins, density=True
            )
            wind_probablities[chunk, :] = wind_probablities_
            wind_speed_bins_edges[chunk, :] = wind_speed_bins_edges_

    wind_speed_distribution = xr.Dataset(
        {
            "probability": (["turbines", "wind_speed_bins"], wind_probablities),
            "wind_speed": (["turbines", "wind_speed_bins"], centers(wind_speed_bins_edges.T).T),
            "wind_speed_bin_edge": (["turbines", "wind_speed_bin_edges"], wind_speed_bins_edges),
        },
        coords={
            "turbines": turbines.turbines,
        },
    )

    return wind_speed_distribution


def power_input(wind_speed, rotor_swept_area):
    """Returns power input in W."""
    return 0.5 * AIR_DENSITY_RHO * wind_speed**3 * rotor_swept_area


def calc_turbine_age(turbines, time):
    """Calculate age for each turbine at each time stamp. Returns a xr.DataArray with dimensions
    'turbines' and 'time' with age in years and 0 before being built.
    """
    age = time.dt.year - turbines.p_year
    age = age.where(age > 0, 0).compute()
    return age
