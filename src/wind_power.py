import logging

import numpy as np
import xarray as xr

from dask.diagnostics import ProgressBar

from src.util import create_folder
from src.config import YEARS
from src.config import OUTPUT_DIR
from src.config import REFERENCE_HUB_HEIGHT_M
from src.constants import AIR_DENSITY_RHO
from src.load_data import load_turbines
from src.load_data import load_wind_speed


def calc_p_in(wind_speed, turbines):
    """Calculates yearly aggregated time series for power in wind captured by operating turbines.
    This means 16/27 of this value is the Betz' limit for the expected production.

    Parameters
    ----------
    wind_speed : xr.DataArray
        as returned by calc_wind_speed_at_turbines()
    turbines : xr.DataSet
        see load_turbines()

    Returns
    -------
    xr.DataArray
        how much power is contained in wind for all turbines installed in this year per (in GW)

    """
    p_in = wind_speed**3 * turbines.t_rd**2 * (np.pi / 4)
    p_in = 0.5 * AIR_DENSITY_RHO * p_in * 1e-9  # in GW

    p_in = p_in.sortby("time").resample(time="1A", label="left", loffset="1D").mean(dim="time")

    return p_in


def calc_p_out(turbines, power_curves, wind_speeds, bias_correction_100m, specific_power=None):
    def resample_annually(data):
        return data.sortby("time").resample(time="1A", label="left", loffset="1D").mean(dim="time")

    def worker(wind_speeds, turbines, specific_power, rotor_swept_area, bias_correction_100m):
        if isinstance(specific_power, xr.DataArray):
            capacity = turbines.t_cap * 1e3
        else:
            # assume specific_power is int or float

            # by changing specific power to something constant, we change the capacity and keep the
            # turbines rotor swept area as it is to match up with total rotor swept area used for
            # P_out / A
            capacity = specific_power * rotor_swept_area

        capacity_factors = power_curves.interp(
            specific_power=specific_power,
            wind_speeds=wind_speeds * bias_correction_100m,
            method="linear",
            kwargs={"bounds_error": True},
        )

        # is_built = calc_is_built(turbines), wind_speeds.time, include_commission_year=None)

        # power output for dims: time, turbines
        p_out = capacity_factors * capacity  # * is_built

        # aggregate
        p_out = resample_annually(p_out)

        return p_out.drop_vars(("x", "y", "specific_power")).transpose("time", "turbines")

    rotor_swept_area = (turbines.t_rd**2 * np.pi / 4).compute()
    if specific_power is None:
        specific_power = ((turbines.t_cap * 1e3) / rotor_swept_area).compute()
    bias_correction_100m = bias_correction_100m.load()

    template = resample_annually(wind_speeds)

    logging.info("Start computation...")

    p_out = xr.map_blocks(
        lambda *args: worker(*args).compute(),
        wind_speeds,
        (turbines, specific_power, rotor_swept_area, bias_correction_100m),
        template=template,
    )

    # TODO this is in W, should be in GW! (conversion to GW in load_p_out_model())
    return p_out


def calc_power(name, compute_func, name_postfix=""):
    """Just a tiny wrapper to reduce code duplication between calculation of p_in and p_out."""
    turbines = load_turbines().compute()  # XXX not sure if the compute is good or necessary
    output_folder = create_folder(name, prefix=OUTPUT_DIR)

    params_calculations = [
        {
            "fname": f"{name}{name_postfix}_hubheight_raw",
            "height": None,
        },
        {
            "fname": f"{name}{name_postfix}_refheight_raw",
            "height": REFERENCE_HUB_HEIGHT_M,
        },
    ]

    for params in params_calculations:
        fname = params["fname"]
        logging.info(f"Starting calculation for {fname}...")

        wind_speed = load_wind_speed(YEARS, params["height"])

        power = compute_func(wind_speed, turbines)

        logging.info("Starting compute()...")

        with ProgressBar():
            power = power.compute()

        logging.info("Computation done!")

        logging.info("Saving result to NetCDF...")
        power.to_netcdf(output_folder / f"{fname}.nc")
