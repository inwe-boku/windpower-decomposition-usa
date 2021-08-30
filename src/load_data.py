import json

import numpy as np
import pandas as pd
import xarray as xr

from src.config import INPUT_DIR
from src.config import INTERIM_DIR
from src.preprocess import estimate_missing

from src.config import CHUNK_SIZE_TURBINES
from src.config import CHUNK_SIZE_TIME


def load_turbines(decommissioned=True, replace_nan_values="mean"):
    """Load list of all turbines from CSV file. Includes location, capacity,
    etc. Missing values are replaced with NaN values.

    The file uswtdb_v1_2_20181001.xml contains more information about the fields.

    Parameters
    ----------
    decommissioned : bool
        if True merge datasets from official CSV with Excel sheet received via e-mail
    replace_nan_values : str
        use data imputation to set missing values for turbine diameters and hub heights, set to ""
        to disable

    Returns
    -------
    xr.DataSet

    """
    turbines_dataframe = pd.read_csv(
        INPUT_DIR / "wind_turbines_usa" / "uswtdb_v3_0_1_20200514.csv"
    )

    # TODO is this really how it is supposed to be done?
    turbines_dataframe.index = turbines_dataframe.index.rename("turbines")
    turbines = xr.Dataset.from_dataframe(turbines_dataframe)

    # Lets not use the turbine on Guam (avoids a huge bounding box for the USA)
    neglected_capacity_kw = turbines.sel(turbines=turbines.xlong >= 0).t_cap.sum()
    assert (
        neglected_capacity_kw == 275
    ), f"unexpected total capacity filtered: {neglected_capacity_kw}"
    turbines = turbines.sel(turbines=turbines.xlong < 0)
    turbines = turbines.set_index(turbines="case_id")

    turbines["is_decomissioned"] = xr.zeros_like(turbines.p_year, dtype=np.bool)

    if not decommissioned:
        return turbines

    turbines_decomissioned = pd.read_excel(
        INPUT_DIR / "wind_turbines_usa" / "decom_clean_032520.xlsx",
        engine="openpyxl",
    )
    turbines_decomissioned = xr.Dataset(turbines_decomissioned).rename(dim_0="turbines")
    turbines_decomissioned = turbines_decomissioned.set_index(turbines="case_id")

    turbines = xr.merge((turbines, turbines_decomissioned))

    turbines["is_decomissioned"] = turbines.decommiss == "yes"
    turbines = turbines.drop_vars("decommiss")

    if replace_nan_values:
        turbines = estimate_missing(turbines, method=replace_nan_values)

    turbines = turbines.chunk(CHUNK_SIZE_TURBINES)

    return turbines


def load_generated_energy_gwh():
    with open(
        INPUT_DIR / "energy_generation" / "ELEC.GEN.WND-US-99.M.json",
        "r",
    ) as f:
        generated_energy_json = json.load(f)

    date, value = zip(*generated_energy_json["series"][0]["data"])

    # unit = thousand megawatthours
    generated_energy_gwh = pd.Series(value, index=pd.to_datetime(date, format="%Y%m"))

    return xr.DataArray(
        generated_energy_gwh,
        dims="time",
        name="Generated energy per month [GWh]",
    )


def load_generated_energy_gwh_yearly():
    """Returns xr.DataArray with dims=time and timestamp as coords"""
    # TODO this should probably have dims='year' and int as coords

    generated_energy_gwh_yearly = (
        load_generated_energy_gwh()
        .sortby("time")
        .resample(time="A", label="left", loffset="1D")
        .sum()
    )
    generated_energy_gwh_yearly = generated_energy_gwh_yearly[
        generated_energy_gwh_yearly.time.dt.year < 2020
    ]
    return generated_energy_gwh_yearly


def load_generated_energy_gwh_yearly_irena():
    """Returns xr.DataArray with dims=year and integer as coords, not timestamp!"""
    generated_energy_twh = pd.read_csv(
        INPUT_DIR / "energy_generation_irena" / "irena-us-generation.csv",
        delimiter=";",
        names=("year", "generation"),
    )
    generated_energy_twh_xr = xr.DataArray(
        generated_energy_twh.generation,
        dims="year",
        coords={"year": generated_energy_twh.year},
    )
    return 1e3 * generated_energy_twh_xr


def load_capacity_irena():
    """Installed capacity in MW."""
    irena_capacity = pd.read_feather(INPUT_DIR / "irena-database" / "irena-2020-02-26-1.7.feather")
    irena_usa_capacity = irena_capacity[
        (irena_capacity.Country == "USA")
        & (irena_capacity.Indicator == "Capacity")
        & (irena_capacity.Variable == "Wind energy")
    ]

    capacity_irena = xr.DataArray(
        irena_usa_capacity.Value, dims="p_year", coords={"p_year": irena_usa_capacity.Year}
    )

    return capacity_irena


def load_wind_velocity(year, month):
    """month/year can be list or int"""
    try:
        iter(year)
    except TypeError:
        year = [year]

    try:
        iter(month)
    except TypeError:
        month = [month]

    fnames = [
        INPUT_DIR / "wind_velocity_usa_era5" / "wind_velocity_usa_{y}-{m:02d}.nc".format(m=m, y=y)
        for m in month
        for y in year
    ]

    wind_velocity_datasets = [
        xr.open_dataset(fname, chunks={"time": CHUNK_SIZE_TIME}) for fname in fnames
    ]

    wind_velocity = xr.concat(wind_velocity_datasets, dim="time")

    # ERA5 data provides data as float32 values
    return wind_velocity.astype(np.float64)


def load_wind_speed(years, height):
    """Load wind speed from processed data files.

    Parameters
    ----------
    years : int or list of ints
    height : float or None

    Returns
    -------
    xr.DataArray

    """
    try:
        iter(years)
    except TypeError:
        years = [years]

    height_name = "hubheight" if height is None else height
    fnames = [
        INTERIM_DIR / "wind_speed" / f"wind_speed_height_{height_name}_{year}.nc" for year in years
    ]

    # TODO is combine='by_coords' correct? does it make a difference?
    wind_speed = xr.open_mfdataset(
        fnames,
        combine="by_coords",
        chunks={"turbines": CHUNK_SIZE_TURBINES, "time": CHUNK_SIZE_TIME},
    )

    if len(wind_speed.data_vars) != 1:
        raise ValueError("This is not a DataArray")

    return wind_speed.__xarray_dataarray_variable__
