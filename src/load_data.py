import json

import numpy as np
import pandas as pd
import xarray as xr

from numpy import interp
from scipy.interpolate import interp1d

from src.config import YEARS
from src.config import INPUT_DIR
from src.config import INTERIM_DIR
from src.config import OUTPUT_DIR
from src.config import MAX_WIND_SPEED
from src.config import OFFSHORE_TURBINES
from src.config import SPECIFIC_POWER_RANGE
from src.config import RESOLUTION_POWER_CURVE_MODEL
from src.config import LOSS_CORRECTION_FACTOR
from src.constants import HOURS_PER_YEAR
from src.preprocess import estimate_missing
from src.calculations import calc_is_built

from src.config import CHUNK_SIZE_TURBINES
from src.config import CHUNK_SIZE_TIME


def _filter_valid_specific_power(turbines):
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


def _filter_offshore(turbines):
    """There are almost no offshore turbines in the USA. According to a talk by Lucy Pao, there are
    only 7 offshore turbines (with 6MW capacity each) operating at the moment. The USWTDB seems to
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

    def read_turbines_csv(fname):
        df = pd.read_csv(INPUT_DIR / "wind_turbines_usa" / fname, encoding="latin_1")
        df = df.set_index("case_id")

        # TODO is this really how it is supposed to be done?
        df.index = df.index.rename("turbines")
        return df

    # FIXME 3% of turbines (measured in capacity) are removed between v3.0.1 and v4.1!
    # also 500MW of capacity between v4.1 and v5.0
    turbines_v301_df = read_turbines_csv("uswtdb_v3_0_1_20200514.csv")
    turbines_v41_df = read_turbines_csv("uswtdb_v4_1_20210721.csv")
    turbines_v50_df = read_turbines_csv("uswtdb_v5_0_20220427.csv")

    # note: conflicts are not checked here
    turbines_df = turbines_v41_df.combine_first(turbines_v301_df)
    turbines_df = turbines_v50_df.combine_first(turbines_df)

    turbines = xr.Dataset.from_dataframe(turbines_df)

    # Lets not use the turbine on Guam (avoids a huge bounding box for the USA)
    neglected_capacity_kw = turbines.sel(turbines=turbines.xlong >= 0).t_cap.sum()
    assert (
        neglected_capacity_kw == 275
    ), f"unexpected total capacity filtered: {neglected_capacity_kw}"
    turbines = turbines.sel(turbines=turbines.xlong < 0)

    turbines["is_decomissioned"] = xr.zeros_like(turbines.p_year, dtype=bool)

    if decommissioned:
        turbines_decomissioned = pd.read_excel(
            INPUT_DIR / "wind_turbines_usa" / "uswtdb_decom_clean_091521.xlsx",
            engine="openpyxl",
        )
        turbines_decomissioned = xr.Dataset(turbines_decomissioned).rename(dim_0="turbines")
        turbines_decomissioned = turbines_decomissioned.set_index(turbines="case_id")

        # note: conflicts are not checked here
        turbines = turbines.combine_first(turbines_decomissioned)

        turbines["is_decomissioned"] = turbines.decommiss == "yes"
        turbines = turbines.drop_vars("decommiss")

    if replace_nan_values:
        turbines = estimate_missing(turbines, method=replace_nan_values)

    turbines = _filter_valid_specific_power(turbines)
    turbines = _filter_offshore(turbines)

    # we are only using data until 2021, because 2022 is not yet complete
    # note that NaNs should not be excluded here!
    turbines = turbines.sel(turbines=~(turbines.p_year >= YEARS.stop))

    turbines = turbines.chunk({"turbines": CHUNK_SIZE_TURBINES})

    return turbines


def load_rotor_swept_area():
    rotor_swept_area = xr.open_dataarray(
        OUTPUT_DIR / "turbine-time-series" / "rotor_swept_area_yearly.nc"
    )
    return rotor_swept_area


def load_is_built():
    is_built_yearly = xr.open_dataarray(OUTPUT_DIR / "turbine-time-series" / "is_built_yearly.nc")
    return is_built_yearly


def _aggregate_p_raw(p_raw, avgwind, aggregate):
    # XXX can we safely load turbines with default data imputation here?
    turbines = load_turbines()

    is_built = calc_is_built(turbines, p_raw.time)

    if avgwind:
        # TODO do we want to be variable in time range of averaging?
        # XXX not entirely correct because of leap years
        p_raw = p_raw.mean(dim="time")

    p = is_built * p_raw

    if aggregate:
        return p.sum(dim="turbines")
    else:
        return p


def load_p_in(
    avgwind=False,
    refheight=False,
    monthly=False,
    bias_correction=True,
    bias_correction_height=100,
    aggregate=True,
):
    # TODO a bit weird to include BIAS correction and is_built here...
    # TODO it's also a bit slow, like 5s to do all the computations here

    assert not monthly, "monthly p_in currently not supported"

    fname = f"p_in_{'refheight' if refheight else 'hubheight'}_raw.nc"
    p_in_raw = xr.open_dataarray(OUTPUT_DIR / "p_in" / fname)

    if bias_correction:
        bias_correction_factors = xr.open_dataarray(
            OUTPUT_DIR
            / "bias_correction"
            / f"bias_correction_factors_gwa2_{bias_correction_height}m.nc"
        )
        p_in_raw = bias_correction_factors**3 * p_in_raw

    return _aggregate_p_raw(p_in_raw, avgwind=avgwind, aggregate=aggregate)


def load_p_out_model(
    avgwind=False,
    refheight=False,
    aggregate=True,
    refspecpower=None,
):
    refspecpower_str = f"_refspecpower_{refspecpower}Wm2" if refspecpower is not None else ""
    fname = f"p_out_model{refspecpower_str}_{'refheight' if refheight else 'hubheight'}_raw.nc"
    p_out_raw = xr.open_dataarray(OUTPUT_DIR / "p_out_model" / fname)

    # TODO * 1e-9 should be moved to calc_p_out
    return (
        LOSS_CORRECTION_FACTOR
        * 1e-9
        * _aggregate_p_raw(p_out_raw, avgwind=avgwind, aggregate=aggregate)
    )


def load_p_out_eia_monthly(state="US"):
    # TODO this function should not be used outside, make it internal
    assert True, "this is not in GW yet, convert!"

    with open(
        INPUT_DIR / "p_out_eia" / f"ELEC.GEN.WND-{state}-99.M.json",
        "r",
    ) as f:
        generated_energy_json = json.load(f)

    date, value = zip(*generated_energy_json["series"][0]["data"])

    # unit = thousand megawatthours
    generated_energy_gwh = pd.Series(value, index=pd.to_datetime(date, format="%Y%m"))

    return xr.DataArray(
        generated_energy_gwh,
        dims="time",
        name=f"Power output per month [GWh] for {state}",
    )


def load_p_out_eia(state="US"):
    """Returns xr.DataArray with dims=time and timestamp as coords in GW"""
    # TODO this should probably have dims='year' and int as coords

    generated_energy_gwh_yearly = (
        load_p_out_eia_monthly(state=state)
        .sortby("time")
        .resample(time="A", label="left", loffset="1D")
        .sum()
    )

    # incomplete year not used
    generated_energy_gwh_yearly = generated_energy_gwh_yearly[
        generated_energy_gwh_yearly.time.dt.year < YEARS.stop
    ]

    generated_energy_gwh_yearly.name = "Power output (GW)"

    # XXX should we take care of leap years and properly scale?
    return generated_energy_gwh_yearly / HOURS_PER_YEAR


def load_p_out_irena():
    """Returns xr.DataArray with dims=time in GW"""
    p_out_irena_twh = pd.read_csv(
        INPUT_DIR / "p_out_irena" / "irena-us-generation.csv",
        delimiter=";",
        names=("year", "generation"),
    )
    p_out_irena_gw = xr.DataArray(
        1e3 * p_out_irena_twh.generation / HOURS_PER_YEAR,
        dims="time",
        coords={"time": pd.to_datetime(p_out_irena_twh.year.astype(str))},
    )
    return p_out_irena_gw


def load_d_out_miller():
    """Power per area of land use, in W/m2.

    Values from:

    Lee M Miller and David W Keith (2019)
    https://doi.org/10.1088/1748-9326/aaf9cf
    """
    return xr.DataArray(
        np.array(
            [
                [2010, 0.93],
                [2011, 0.96],
                [2012, 0.93],
                [2013, 0.92],
                [2014, 0.94],
                [2015, 0.86],
                [2016, 0.90],
            ]
        ).T[1],
        dims="time",
        coords={"time": pd.date_range("2010", "2016", freq="YS")},
    )


def load_capacity_irena():
    """Installed capacity in MW."""
    irena_capacity = pd.read_feather(INPUT_DIR / "capacity_irena" / "irena-2022-06-03-2.feather")
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
        xr.open_dataset(fname, chunks={"time": CHUNK_SIZE_TIME}, autoclose=True)
        for fname in fnames
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


def load_wind_speed_gwa(height):
    wind_speed_gwa = xr.open_dataarray(
        INTERIM_DIR / "wind_speed_gwa" / f"wind_speed_gwa{height}.nc",
        chunks=1_000,
        autoclose=True,
    )
    return wind_speed_gwa


def load_power_curve_model():
    """Implements the Power curve model by Ryberg et al.
    See Appendix A in DOI: 10.1016/j.energy.2019.06.052

    Returns a xr.DataArray object with dims specific_power (W/m2) and wind_speeds (m/s), which
    contains capacity factors.

    """
    # TODO we could add a cut-off wind speed, but probably this won't make a difference
    AB = pd.read_csv(
        INPUT_DIR / "power_curve_modell" / "table_a_b_constants.csv", delimiter=r"\s+"
    )
    AB = AB.to_xarray().rename_dims(index="capacity_factor")
    AB = AB.assign_coords(capacity_factor=AB.CF)
    AB = AB.drop_vars(("index", "CF"))
    AB["capacity_factor"] = AB.capacity_factor / 100.0

    # XXX what about wind speeds above 25? How to deal with min/max values here?
    # XXX what if specific power is too large?
    specific_power = xr.DataArray(
        np.linspace(
            SPECIFIC_POWER_RANGE[0],
            SPECIFIC_POWER_RANGE[1],
            num=RESOLUTION_POWER_CURVE_MODEL["specific_power"],
        ),
        dims="specific_power",
    )
    wind_speeds_grid = xr.DataArray(
        np.linspace(0, MAX_WIND_SPEED, num=RESOLUTION_POWER_CURVE_MODEL["wind_speeds"]),
        dims="wind_speeds",
    )

    wind_speeds_model = np.exp(AB.A) * specific_power**AB.B

    # note: this interp() does not return NaN values, but does constant continuation
    power_curves_list = [
        interp(
            wind_speeds_grid,
            wind_speeds_model.isel(specific_power=idx).values,  # XXX increasing?
            wind_speeds_model.capacity_factor.values,
        )
        for idx in range(len(specific_power))
    ]
    power_curves = xr.DataArray(power_curves_list, coords=(specific_power, wind_speeds_grid))

    return power_curves


_power_curve_ge15_77 = None


def power_curve_ge15_77(wind_speed):
    """Power curve for GE1.5-77: returns power output in W.

    https://www.ge.com/in/wind-energy/1.5-MW-wind-turbine

    Hub height: 65 / 80m

    https://www.nrel.gov/docs/fy15osti/63684.pdf  page 21
    https://geosci.uchicago.edu/~moyer/GEOS24705/Readings/GEA14954C15-MW-Broch.pdf

    """
    global _power_curve_ge15_77

    if _power_curve_ge15_77 is None:
        ms_to_seconds = 1 / 100.0
        capacity_kw = 1500
        data = pd.read_csv(
            INPUT_DIR / "power_curves" / "power-curves-5_0_01ms_with_0w_smoother.csv"
        )
        wind_speeds = data["GE.1.5sle"].index.values * ms_to_seconds
        generation_kw = data["GE.1.5sle"].values * capacity_kw

        _power_curve_ge15_77 = interp1d(wind_speeds, generation_kw * 1e3, kind="cubic")

    return _power_curve_ge15_77(wind_speed)
