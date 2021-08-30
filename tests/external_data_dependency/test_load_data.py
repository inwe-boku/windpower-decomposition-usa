import xarray as xr
import numpy as np

from src import load_data


def test_load_turbines():
    turbines = load_data.load_turbines()
    assert np.isnan(turbines.t_cap).sum() == 0
    assert turbines.p_year.min() == 1981
    assert turbines.p_year.max() == 2019


def test_load_turbines_with_nans():
    turbines_with_nans = load_data.load_turbines(replace_nan_values=False)
    assert (np.isnan(turbines_with_nans.t_cap)).sum() == 7231


def test_load_generated_energy_gwh():
    generated_energy_gwh = load_data.load_generated_energy_gwh()

    assert generated_energy_gwh.sel(time="2001-01-01") == 389.25
    assert generated_energy_gwh.sel(time="2013-12-01") == 13967.05881
    assert len(generated_energy_gwh) == 238
    assert np.max(generated_energy_gwh) == 29871.72279

    assert generated_energy_gwh.dtype == np.float
    assert isinstance(generated_energy_gwh, xr.DataArray)
    assert generated_energy_gwh.dims == ("time",)


def test_load_wind_velocity():
    year = 2017
    month = 3
    wind_velocity = load_data.load_wind_velocity(year, month)
    assert len(wind_velocity.time) == 744
    assert float(wind_velocity.u100.isel(time=0, longitude=3, latitude=2)) == 3.2684133052825928
