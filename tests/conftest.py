import pytest
import numpy as np
import pandas as pd
import xarray as xr


@pytest.fixture
def time():
    num_time_stamps = 17
    timestamps_per_year = [
        pd.date_range(f"{year}-01-01", periods=num_time_stamps, freq="H")
        for year in range(1997, 2012)
    ]
    values = np.concatenate(timestamps_per_year)
    return xr.DataArray(values, dims="time", name="time", coords={"time": values})


@pytest.fixture
def turbines():
    num_turbines = 100
    turbines = xr.Dataset(
        {
            "xlong": ("turbines", np.arange(num_turbines, dtype=np.float64)),
            "ylat": ("turbines", np.arange(num_turbines, dtype=np.float64)),
            "t_rd": ("turbines", 10 * np.ones(num_turbines)),
            "t_hh": ("turbines", 15 * np.ones(num_turbines)),
            "t_cap": ("turbines", 1500 * np.ones(num_turbines)),
            "p_year": (
                "turbines",
                np.repeat(np.arange(1999, 1999 + 10, dtype=np.float), num_turbines / 10),
            ),
        },
        coords={"turbines": np.arange(13, num_turbines + 13)},
    )

    # disabled: used to test NaN scaling, not used any longer
    # turbines["t_rd"][0] = np.nan
    # turbines["t_rd"][1] = np.nan
    # turbines["t_rd"][23] = np.nan
    # turbines["t_rd"][55] = np.nan

    return turbines


@pytest.fixture
def wind_speed(turbines, time):
    num_turbines = turbines.sizes["turbines"]
    wind_speed = xr.DataArray(
        3 * np.ones((len(time), num_turbines)),
        dims=("time", "turbines"),
        coords={"time": time, "longitude": turbines.xlong, "latitude": turbines.ylat},
    )

    # more wind between 2005 and 2011
    wind_speed.loc[{"time": wind_speed.time.dt.year >= 2005}] = 4.0

    # disabled: used to test NaN scaling, not used any longer
    # if turbines.t_hh is NaN, wind_speed is NaN too
    # wind_speed.loc[{"turbines": 15}] = np.nan

    return wind_speed


@pytest.fixture
def wind_velocity(time):
    num_turbines = 100
    wind_velocity_array = np.ones(
        (len(time), num_turbines + 20, num_turbines + 25), dtype=np.float32
    )  # ERA5 data comes in 32bit format!
    wind_velocity = xr.Dataset(
        {
            "u100": (("time", "latitude", "longitude"), 3 * wind_velocity_array),
            "v100": (("time", "latitude", "longitude"), 4 * wind_velocity_array),
            "u10": (("time", "latitude", "longitude"), 3e-1 * wind_velocity_array),
            "v10": (("time", "latitude", "longitude"), 4e-1 * wind_velocity_array),
        },
        coords={
            "latitude": np.arange(-10, num_turbines + 10),
            "longitude": np.arange(-10, num_turbines + 15),
            "time": time,
        },
    )
    return wind_velocity
