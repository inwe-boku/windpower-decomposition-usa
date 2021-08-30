import logging

import dask as da
import numpy as np
import pandas as pd
import xarray as xr

from src.util import centers
from src.config import CHUNK_SIZE_TURBINES
from src.config import CHUNK_SIZE_TIME
from src.constants import AIR_DENSITY_RHO


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
    # interpolate at turbine locations
    with da.config.set(**{"array.slicing.split_large_chunks": True}):
        wind_velocity_at_turbines = wind_velocity.interp(
            longitude=xr.DataArray(turbines.xlong.values, dims="turbines"),
            latitude=xr.DataArray(turbines.ylat.values, dims="turbines"),
            method="linear",
        )

    # velocity --> speed
    wind_speed100 = (
        wind_velocity_at_turbines.u100 ** 2 + wind_velocity_at_turbines.v100 ** 2
    ) ** 0.5

    height_attr = height

    if height is None:
        # not very nice, because suddenly height is a vector!
        height = turbines.t_hh
        height_attr = 0.0  # ugly special value for extrapolation at hub height
    else:
        # we want wind speed to be NaN if hub height is missing to have a consistent NaN scaling
        height = height * (turbines.t_hh - turbines.t_hh + 1)

    wind_speed10 = (wind_velocity_at_turbines.u10 ** 2 + wind_velocity_at_turbines.v10 ** 2) ** 0.5

    powerlaw_alpha = np.log10(wind_speed100 / wind_speed10)
    wind_speed = wind_speed100 * (height / 100.0) ** powerlaw_alpha

    # None refers to turbine height
    wind_speed.attrs["height"] = height_attr

    return wind_speed


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
        p_year = turbines.p_year.chunk(CHUNK_SIZE_TURBINES)
        year = time.dt.year.chunk(CHUNK_SIZE_TIME)

        if include_commission_year is True:
            is_built = (p_year <= year).astype(np.float)
        elif include_commission_year is False:
            is_built = (p_year < time.dt.year).astype(np.float)
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
            is_built = (p_year < time.dt.year).astype(np.float)
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
            is_leap_year = p_year_date.dt.is_leap_year.astype(np.float)

            p_year_date = p_year_date.chunk(CHUNK_SIZE_TURBINES)
            time = time.chunk(CHUNK_SIZE_TIME)
            is_leap_year.chunk(CHUNK_SIZE_TURBINES)

            # this is where the broadcasting magic takes place
            nanosecs_of_year = (time - p_year_date).astype(np.float)

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


def calc_power_in_wind(wind_speed, turbines, average_wind=False):
    """Calculates yearly aggregated time series for power in wind captured by operating turbines.
    This means 16/27 of this value is the Betz' limit for the expected production.

    Missing values in t_rd and t_hh are scaled. Missing values in p_year are ignored (as if
    turbines are not built at all).

    Parameters
    ----------
    wind_speed : xr.DataArray
        as returned by calc_wind_speed_at_turbines()
    turbines : xr.DataSet
        see load_turbines()
    average_wind : bool
        use mean wind power over the complete time frame

    Returns
    -------
    xr.DataArray
        how much power is contained in wind for all turbines installed in this year per (in GW)

    """
    # this function is a bit crazy because p_in has different dimensions in every line of code
    # and average_wind changes the behavior in surprising ways.

    if average_wind:
        years_int = np.unique(wind_speed.time.dt.year)
        assert (
            years_int.ptp() == len(years_int) - 1
        ), f"years missing, this is not supported, years: {years_int}"

        time_yearly = (
            wind_speed.time.sortby("time")
            .resample(time="1A", label="left", loffset="1D")
            .first()
            .time
        )

        is_built = calc_is_built(turbines, time_yearly.time)

        # TODO this may create a "mean empty slice" warning, because some turbine locations have no
        # wind speed... Not sure how to deal with that.
        p_in = (wind_speed ** 3).mean(dim="time")
    else:
        is_built = calc_is_built(turbines, wind_speed.time)

        p_in = wind_speed ** 3  # here p_in has dims=('turbines', 'time')

    p_in = p_in * turbines.t_rd ** 2 * (np.pi / 4)

    logging.info("Converting is_built to float...")
    logging.info("Applying is_built...")
    p_in = p_in * is_built
    logging.info("Applying is_built done!")

    p_in = p_in.sum(dim="turbines")

    p_in_monthly = None

    if not average_wind:
        p_in_monthly = (
            p_in.sortby("time").resample(time="1M", label="left", loffset="1D").mean(dim="time")
        )
        # TODO code duplication with below :(
        p_in_monthly = 0.5 * AIR_DENSITY_RHO * p_in_monthly * 1e-9  # in GW

        p_in = p_in.sortby("time").resample(time="1A", label="left", loffset="1D").mean(dim="time")

    p_in = 0.5 * AIR_DENSITY_RHO * p_in * 1e-9  # in GW

    logging.info("Returning lazy object...")
    return p_in, p_in_monthly


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


def fit_efficiency_model(
    p_in, p_out, p_in_density, efficiency, use_monthly_dummies=False, use_time=False
):
    # local import to suppress warning in unit tests, see:
    # https://github.com/statsmodels/statsmodels/issues/7139
    from statsmodels.api import OLS
    from statsmodels.tools.tools import add_constant

    X = pd.DataFrame(
        {
            "p_in_density": p_in_density,
        }
    )

    if use_time:
        # not really time, just a sequentially increasing number
        X["time"] = range(len(X))

    if use_monthly_dummies:
        # we add a constant below, so we have to drop one month
        X = X.join(pd.get_dummies(p_in_density.time.dt.month, drop_first=True))

    # other possible parameters:
    #  - specific power
    #  - turbine age

    X = add_constant(X)
    Y = efficiency.values

    model = OLS(Y, X)
    fit_result = model.fit()

    efficiency_without_pin = (
        fit_result.params.const
        + fit_result.params.p_in_density * p_in_density.mean().values
        + fit_result.resid
    )

    if use_time:
        efficiency_without_pin += fit_result.params.time * X["time"]

    # note: this might be broken if lengths of p_in and p_out do not match up
    assert len(p_in) == len(efficiency_without_pin), "input lengths do not match"
    efficiency_without_pin = xr.ones_like(p_in) * efficiency_without_pin

    return fit_result, efficiency_without_pin
