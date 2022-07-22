import os
import json
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr

from src.config import YEARS
from src.config import MONTHS
from src.config import INPUT_DIR
from src.config import OUTPUT_DIR
from src.config import DATA_DIR
from src.config import OFFSHORE_TURBINES
from src.util import create_folder
from src.constants import HOURS_PER_YEAR
from src.load_data import load_turbines
from src.logging_config import setup_logging
from src.calculations import calc_bounding_box_usa


def create_turbines(save_to_file=True):
    np.random.seed(12)

    num_turbines = 4000

    # case_id has gaps in the real dataset, so we generate 20% more IDs and pick randomly
    start_index = 3000001
    case_id_sequential = np.arange(
        start_index,
        start_index + num_turbines * 1.2,
        dtype=np.int64,
    )
    case_id = np.random.choice(case_id_sequential, size=num_turbines, replace=False)
    case_id.sort()

    ylat = np.random.uniform(17, 66, size=num_turbines)
    xlong = np.random.uniform(-171, -65, size=num_turbines)

    # note: a positive commissioning rate, means that newly built turbines increase linearly (with
    # the given rate), meaning that total number of built turbines increases quadratically
    comissioning_rate = 0
    p_year_min = 1981
    p_year_max = 2020
    num_years = p_year_max - p_year_min + 1

    assert num_turbines % num_years == 0, (
        f"invalid testset config, num_turbines={num_turbines} not "
        f"divisible by num_years={num_years}"
    )

    assert comissioning_rate * (num_years - 1) % 2 == 0, (
        f"invalid testset config, neither comissioning_rate={comissioning_rate} "
        f"nor num_years={num_years} is even"
    )

    num_turbines_start_year = int(
        num_turbines / num_years - 0.5 * comissioning_rate * (num_years - 1)
    )

    num_turbines_per_year = num_turbines_start_year + comissioning_rate * np.arange(num_years)

    p_year = np.repeat(
        np.arange(p_year_min, p_year_max + 1),
        num_turbines_per_year,
    ).astype(np.float64)

    def fill_nans(d, ratio):
        """Fill array with NaNs, approximately ratio of lenght of vector. Modifies input."""
        size = len(d)
        idcs = np.random.randint(size, size=int(ratio * size))
        d[idcs] = np.nan

    fill_nans(p_year, 0.03)

    def normal_random(start, end, size, minimum, nanratio):
        loc = np.linspace(start, end, num=size)
        d = np.random.normal(loc=loc, scale=loc * 0.1)
        d = d.clip(min=minimum)
        fill_nans(d, nanratio)
        return d

    # TODO we might need a different distribution of missing values over time for better simulation
    t_hh = normal_random(
        start=50,
        end=180,
        size=num_turbines,
        minimum=10.0,
        nanratio=0.18,
    )
    t_rd = normal_random(
        start=100,
        end=130,
        size=num_turbines,
        minimum=10.0,
        nanratio=0.12,
    )
    t_cap = normal_random(
        start=2600,
        end=2600,
        size=num_turbines,
        minimum=30.0,
        nanratio=0.10,
    )

    # these turbines are offshore and discarded in load_turbines()
    case_id[-len(OFFSHORE_TURBINES) :] = [turbine["id"] for turbine in OFFSHORE_TURBINES]
    xlong[-len(OFFSHORE_TURBINES) :] = [turbine["xlong"] for turbine in OFFSHORE_TURBINES]
    ylat[-len(OFFSHORE_TURBINES) :] = [turbine["ylat"] for turbine in OFFSHORE_TURBINES]
    p_year[-len(OFFSHORE_TURBINES) :] = [2020 for _ in OFFSHORE_TURBINES]

    turbines = xr.Dataset(
        {
            "turbines": case_id,
            "xlong": ("turbines", xlong),
            "ylat": ("turbines", ylat),
            "p_year": ("turbines", p_year),
            "t_hh": ("turbines", t_hh),
            "t_rd": ("turbines", t_rd),
            "t_cap": ("turbines", t_cap),
        }
    )

    # in the real dataset there are no rotor diameters in 2020
    turbines["t_rd"] = turbines.t_rd.where(turbines.p_year != 2020)

    if not save_to_file:
        return turbines

    turbines_df = turbines.to_dataframe()
    turbines_df.index.names = ["case_id"]
    fname = create_folder("wind_turbines_usa", prefix=INPUT_DIR) / "uswtdb_v3_0_1_20200514.csv"

    # this is just too dangerous...
    if op.exists(fname):
        raise RuntimeError(
            "CSV file for turbines already exists, won't overwrite, " f"path: {fname}"
        )

    turbines_df.to_csv(fname)

    # add just one (made up) turbine to the decommissioning set, to test reading the Excel file
    turbines_decomissioned = pd.DataFrame(
        [
            # [
            #    "3011181",
            #    "251 Wind",
            #    "1995",
            #    "3084",
            #    "",
            #    "Vestas North America",
            #    "Unknown Vestas",
            #    "105",
            #    "100",
            #    "",
            #    "",
            #    "1",
            #    "3",
            #    "",
            #    "yes",
            #    "",
            #    "-108",
            #    "35",
            # ]
        ],
        columns=[
            "case_id",
            "p_name",
            "p_year",
            "p_tnum",
            "p_cap",
            "t_manu",
            "t_model",
            "t_cap",
            "t_hh",
            "t_rd",
            "t_ttlh",
            "t_conf_atr",
            "t_conf_loc",
            "t_img_date",
            "decommiss",
            "d_year",
            "xlong",
            "ylat",
        ],
    )
    turbines_decomissioned = turbines_decomissioned.set_index("case_id")
    turbines_decomissioned.to_excel(
        INPUT_DIR / "wind_turbines_usa" / "uswtdb_decom_clean_091521.xlsx",
        engine="openpyxl",
    )

    header = (
        "case_id,faa_ors,faa_asn,usgs_pr_id,eia_id,t_state,t_county,t_fips,p_name,p_year,"
        "p_tnum,p_cap,t_manu,t_model,t_cap,t_hh,t_rd,t_rsa,t_ttlh,retrofit,retrofit_year,"
        "t_conf_atr,t_conf_loc,t_img_date,t_img_srce,xlong,ylat\n"
    )

    turbine_str = (
        "3063607,,2013-WTW-2712-OE,,,GU,Guam,66010,Guam Power Authority Wind Turbine,2016,1,"
        "0.275,Vergnet,GEV MP-C,275,55,32,804.25,71,0,,2,3,8/10/2017,Digital Globe,144.722656,"
        "13.389381\n"
    )
    for fname in ("uswtdb_v4_1_20210721.csv", "uswtdb_v5_0_20220427.csv"):
        # just a static CSV file with one turbine which is actually removed in load_turbines()
        with open(INPUT_DIR / "wind_turbines_usa" / fname, "w") as f:
            f.write(header)
            f.write(turbine_str)

    return turbines


def create_wind_velocity():
    turbines = load_turbines()

    dims = ("time", "latitude", "longitude")

    north, west, south, east = calc_bounding_box_usa(turbines)

    longitude = np.arange(west, east, step=0.25, dtype=np.float32)
    latitude = np.arange(south, north, step=0.25, dtype=np.float32)

    np.random.seed(42)

    for year in YEARS:
        for month in MONTHS:
            time_ = pd.date_range(f"{year}-{month}-01", periods=4, freq="7d")
            data = np.ones(
                (len(time_), len(latitude), len(longitude)),
                dtype=np.float32,
            )

            wind_velocity = xr.Dataset(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "time": time_,
                    "u100": (
                        dims,
                        3 * data + np.random.normal(scale=2.5, size=(len(time_), 1, 1)),
                    ),
                    "v100": (dims, -4 * data),
                    "u10": (dims, data + np.random.normal(scale=0.5, size=(len(time_), 1, 1))),
                    "v10": (dims, -data + np.random.normal(scale=0.5, size=(len(time_), 1, 1))),
                }
            )

            fname = "wind_velocity_usa_{year}-{month:02d}.nc".format(month=month, year=year)
            path = (
                create_folder(
                    "wind_velocity_usa_era5",
                    prefix=INPUT_DIR,
                )
                / fname
            )

            if op.exists(path):
                raise RuntimeError(
                    "wind velocity file already exists, won't overwrite, " f"path: {path}"
                )
            wind_velocity.to_netcdf(path)


def create_p_out_eia():
    timestamps = [f"{year}{month:02d}" for year in YEARS for month in MONTHS]

    timestamps += [f"{YEARS[-1] + 1}{month:02d}" for month in MONTHS[:3]]

    energy_yearly = (
        16 / 27 * 0.8 * np.linspace(2.19448551, 4.38897103, num=len(YEARS)) * HOURS_PER_YEAR
    )
    values = np.repeat(energy_yearly / 12.0, 12)

    p_out_eia = {"series": [{"data": list(zip(timestamps, values))}]}
    path = create_folder("p_out_eia", prefix=INPUT_DIR)
    with open(path / "ELEC.GEN.WND-US-99.M.json", "w") as f:
        json.dump(p_out_eia, f)


def create_p_out_irena():
    fname = create_folder("p_out_irena", prefix=INPUT_DIR) / "irena-us-generation.csv"
    with open(fname, "w") as f:
        f.write(
            "2010;95.148\n"
            "2011;120.987\n"
            "2012;140.222\n"
            "2013;160.000\n"
            "2014;180.000\n"
            "2015;190.000\n"
            "2016;215.000\n"
            "2017;250.000\n"
            "2018;275.000\n"
            "2019;295.456\n"
            "2020;300.123\n"
        )


def create_irena_capcity_db():
    irena_db = pd.DataFrame(
        {
            "Year": np.arange(2000, 2020, dtype=float),
            "Country": "USA",
            "Indicator": "Capacity",
            "Unit": "MW",
            "Variable": "Wind energy",
            "Value": np.linspace(2377.0, 99401.0, num=20),
        },
        columns=["Country", "Year", "Variable", "Indicator", "Unit", "Value"],
    )

    # no idea what this is, but it's present in real data too...
    irena_db.at[19, "Value"] = np.nan
    irena_db.at[19, "Year"] = np.nan

    fname = create_folder("capacity_irena", prefix=INPUT_DIR) / "irena-2022-06-03-2.feather"
    irena_db.to_feather(fname)


def create_biascorrection():
    output_path = create_folder("bias_correction", prefix=OUTPUT_DIR)

    turbines = load_turbines()
    bias_correction = xr.DataArray(
        np.ones(turbines.sizes["turbines"]),
        dims="turbines",
        coords={
            "turbines": turbines.turbines,
            "x": turbines.xlong,
            "y": turbines.ylat,
            "longitude": turbines.xlong,
            "latitude": turbines.ylat,
        },
    )

    for height in (50, 100, 250):
        ((1 + height / 1e3) * bias_correction).to_netcdf(
            output_path / f"bias_correction_factors_gwa2_{height}m.nc"
        )


def create_power_curve_modell():
    fname = create_folder("power_curve_modell", prefix=INPUT_DIR) / "table_a_b_constants.csv"

    # this creates a linear power curve from 0m/s to 25m/s for all specific powers
    capacity_factors = np.arange(1, 101, dtype=float)  # in percent
    max_wind = 25
    A = max_wind / 100 * capacity_factors
    A = np.log(A)

    AB = pd.DataFrame(
        {
            "CF": capacity_factors,
            "A": A,
            "B": np.zeros(100),
        }
    )
    AB.to_csv(fname, sep="\t")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    setup_logging()
    create_p_out_eia()
    create_p_out_irena()
    create_turbines()
    create_wind_velocity()
    create_irena_capcity_db()
    create_biascorrection()
    create_power_curve_modell()
