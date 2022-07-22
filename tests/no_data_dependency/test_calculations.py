import numpy as np
import pandas as pd
import xarray as xr

from src.wind_power import calc_p_in
from src.constants import AIR_DENSITY_RHO
from src.calculations import calc_wind_speed_at_turbines
from src.calculations import calc_is_built
from src.calculations import calc_rotor_swept_area
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

    # I'm not entirely sure why there is a small numerical error here after changing np.float to
    # float to avoid a warning after a numpy update, but it's about 1e-14 so should be ignorable
    np.testing.assert_allclose(
        is_built.sel(time=time <= pd.to_datetime("2002-01-01 16:00")).sum(dim="turbines").max(),
        30.0 + 10 * 16 / (365 * 24),
    )

    is_built_diff = is_built.astype(float).diff(dim="time")
    assert np.all(is_built_diff.sum(dim="time") == 1)


def test_calc_rotor_swept_area(turbines, time):
    rotor_swept_area = calc_rotor_swept_area(turbines, time)
    assert isinstance(rotor_swept_area, xr.DataArray)

    num_turbines = 10
    # there are 10 turbines installed every year, starting with year 2000
    # note that there are two rotor diameters missing in t_rd, but this is corrected by nan-scaling
    first_year = "2000-01-01T00:00:00.000000000"
    np.testing.assert_allclose(
        rotor_swept_area.sel(time=first_year), (num_turbines) * 10**2 / 4 * np.pi
    )


def test_calc_p_in(wind_speed, turbines):
    num_turbines = 100

    p_in = calc_p_in(wind_speed, turbines)
    p_in = p_in.sum(dim="turbines")
    assert isinstance(p_in, xr.DataArray)
    assert p_in.time[0] == pd.to_datetime("1997-01-01")
    assert p_in.time[-1] == pd.to_datetime("2011-01-01")

    rotor_swept_area = 5**2 * np.pi

    wind_speed_cube_start = 3**3
    wind_speed_cube_end = 4**3

    p_in_factors = num_turbines * rotor_swept_area * 0.5 * AIR_DENSITY_RHO * 1e-9

    np.testing.assert_allclose(
        p_in.sel(time=p_in.time.dt.year == 2000),
        wind_speed_cube_start * p_in_factors,
    )

    # all turbines built
    np.testing.assert_allclose(
        p_in.sel(time=p_in.time.dt.year > 2008), wind_speed_cube_end * p_in_factors
    )


def test_calc_p_in_p_year_nans(wind_speed, turbines):
    # this test became kinda useless since calc_p_in() ignores atm if p_year is nan
    num_turbines = 100
    rotor_swept_area = 5**2 * np.pi

    turbines["p_year"][95] = np.nan
    turbines["p_year"][94] = np.nan

    wind_speed.loc[{"time": wind_speed.time.dt.year >= 0}] = 4.0
    p_in = calc_p_in(wind_speed, turbines)
    p_in = p_in.sum(dim="turbines")

    np.testing.assert_allclose(
        p_in.sel(time=p_in.time.dt.year > 2008),
        4**3 * num_turbines * rotor_swept_area * 0.5 * AIR_DENSITY_RHO * 1e-9,
    )


def test_calc_capacity_per_year(turbines):
    capacity_uswtdb = calc_capacity_per_year(turbines)
    assert capacity_uswtdb.sel(p_year=1999) == 15
    assert capacity_uswtdb.sel(p_year=2000) == 30
