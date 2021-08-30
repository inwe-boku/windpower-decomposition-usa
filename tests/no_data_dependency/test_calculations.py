import pytest
import numpy as np
import pandas as pd
import xarray as xr

from src.constants import AIR_DENSITY_RHO
from src.calculations import calc_wind_speed_at_turbines
from src.calculations import calc_is_built
from src.calculations import calc_rotor_swept_area
from src.calculations import calc_power_in_wind
from src.calculations import calc_capacity_per_year


def test_calc_wind_speed_at_turbines(wind_velocity, turbines):
    wind_speed = calc_wind_speed_at_turbines(wind_velocity, turbines, height=100.0)
    assert isinstance(wind_speed, xr.DataArray)
    assert np.all(wind_speed.isel(turbines=0) == 5)


def test_calc_is_built(turbines, time):
    is_built = calc_is_built(turbines, time)

    assert is_built.dims == ("turbines", "time")
    assert np.all(0 <= is_built)
    assert np.all(is_built <= 1)

    assert np.all(is_built.isel(time=0) == 0), "in the beginning no turbine should be built"
    assert np.all(is_built.isel(time=-1) == 1), "at the and all turbines should be built"

    assert is_built.sel(time=time <= pd.to_datetime("2002-01-01")).sum(dim="turbines").max() == 30

    assert is_built.sel(time=time <= pd.to_datetime("2002-01-01 16:00")).sum(
        dim="turbines"
    ).max() == 30.0 + 10 * 16 / (365 * 24)

    is_built_diff = is_built.astype(np.float).diff(dim="time")
    assert np.all(is_built_diff.sum(dim="time") == 1)


def test_calc_rotor_swept_area(turbines, time):
    rotor_swept_area = calc_rotor_swept_area(turbines, time)
    assert isinstance(rotor_swept_area, xr.DataArray)

    num_turbines = 10
    # there are 10 turbines installed every year, starting with year 2000
    # note that there are two rotor diameters missing in t_rd, but this is corrected by nan-scaling
    first_year = "2000-01-01T00:00:00.000000000"
    np.testing.assert_allclose(
        rotor_swept_area.sel(time=first_year), (num_turbines) * 10 ** 2 / 4 * np.pi
    )


@pytest.mark.parametrize("average_wind", [True, False])
def test_calc_power_in_wind(wind_speed, turbines, average_wind):
    num_turbines = 100

    p_in, _ = calc_power_in_wind(wind_speed, turbines, average_wind)
    assert isinstance(p_in, xr.DataArray)
    assert p_in.time[0] == pd.to_datetime("1997-01-01")
    assert p_in.time[-1] == pd.to_datetime("2011-01-01")

    rotor_swept_area = 5 ** 2 * np.pi

    wind_speed_cube_start = 3 ** 3
    wind_speed_cube_end = 4 ** 3

    # expected p_in for years when all turbines are built
    if average_wind:
        wind_speed_cube_avg = (8 * wind_speed_cube_start + 7 * wind_speed_cube_end) / 15
        wind_speed_cube_start = wind_speed_cube_end = wind_speed_cube_avg

    p_in_factors = num_turbines * rotor_swept_area * 0.5 * AIR_DENSITY_RHO * 1e-9

    # no turbines built yet
    assert np.all(p_in.sel(time=p_in.time.dt.year < 1999) == 0.0)

    num_turbines_1999 = 10

    if average_wind:
        # this is weird because of the yearly parameter (workaround) in calc_is_built()
        num_turbines_until_2000 = num_turbines_1999 + 10 / 2
    else:
        # this is a bit weird, because we don't have the time stamps of a full year here, but just
        # the first 17 hours of the year...

        # 10 turbines weighted by the sum of the first 17 hours of the year divided by hours of
        # year
        num_time_stamps = 17
        num_turbines_2000 = 10 * (num_time_stamps - 1) / 2 / (366 * 24)
        num_turbines_until_2000 = num_turbines_1999 + num_turbines_2000

    np.testing.assert_allclose(
        p_in.sel(time=p_in.time.dt.year == 2000),
        wind_speed_cube_start * p_in_factors * num_turbines_until_2000 / num_turbines,
    )

    # all turbines built
    np.testing.assert_allclose(
        p_in.sel(time=p_in.time.dt.year > 2008), wind_speed_cube_end * p_in_factors
    )


@pytest.mark.parametrize("average_wind", [True, False])
def test_calc_power_in_wind_p_year_nans(wind_speed, turbines, average_wind):
    num_turbines = 100
    rotor_swept_area = 5 ** 2 * np.pi

    turbines["p_year"][95] = np.nan
    turbines["p_year"][94] = np.nan

    wind_speed.loc[{"time": wind_speed.time.dt.year >= 0}] = 4.0
    p_in, _ = calc_power_in_wind(wind_speed, turbines, average_wind)

    np.testing.assert_allclose(
        p_in.sel(time=p_in.time.dt.year > 2008),
        98 / 100 * 4 ** 3 * num_turbines * rotor_swept_area * 0.5 * AIR_DENSITY_RHO * 1e-9,
    )


def test_calc_capacity_per_year(turbines):
    capacity_uswtdb = calc_capacity_per_year(turbines)
    assert capacity_uswtdb.sel(p_year=1999) == 15
    assert capacity_uswtdb.sel(p_year=2000) == 30
