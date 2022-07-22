import xarray as xr
import numpy as np

from src import load_data


def test_load_turbines():
    turbines = load_data.load_turbines()
    assert np.isnan(turbines.t_cap).sum() == 0
    assert turbines.p_year.min() == 1981
    assert turbines.p_year.max() == 2021


def test_load_turbines_with_nans():
    turbines_with_nans = load_data.load_turbines(replace_nan_values=False)
    assert (np.isnan(turbines_with_nans.t_cap)).sum().load() == 6595


def test_load_p_out_eia():
    p_out_eia_monthly = load_data.load_p_out_eia_monthly()

    assert p_out_eia_monthly.sel(time="2001-01-01") == 389.25
    assert p_out_eia_monthly.sel(time="2013-12-01") == 13967.05881
    assert len(p_out_eia_monthly) == 255
    assert np.max(p_out_eia_monthly) == 43230.38297

    assert p_out_eia_monthly.dtype == np.float64
    assert isinstance(p_out_eia_monthly, xr.DataArray)
    assert p_out_eia_monthly.dims == ("time",)


def test_load_wind_velocity():
    year = 2017
    month = 3
    wind_velocity = load_data.load_wind_velocity(year, month)
    assert len(wind_velocity.time) == 744
    assert float(wind_velocity.u100.isel(time=0, longitude=3, latitude=2)) == 3.2684133052825928
