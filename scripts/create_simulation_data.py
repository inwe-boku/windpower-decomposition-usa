import os
import json
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr

from src.config import YEARS
from src.config import MONTHS
from src.config import INPUT_DIR
from src.config import DATA_DIR
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
        end=200,
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

    if save_to_file:
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
        INPUT_DIR / "wind_turbines_usa" / "decom_clean_032520.xlsx",
        engine="openpyxl",
    )

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


def create_energy_generation():
    timestamps = [f"{year}{month:02d}" for year in YEARS for month in MONTHS]

    timestamps += [f"{YEARS[-1] + 1}{month:02d}" for month in MONTHS[:3]]

    energy_yearly = (
        16 / 27 * 0.8 * np.linspace(2.19448551, 4.38897103, num=len(YEARS)) * HOURS_PER_YEAR
    )
    values = np.repeat(energy_yearly / 12.0, 12)

    energy_generation = {"series": [{"data": list(zip(timestamps, values))}]}
    path = create_folder("energy_generation", prefix=INPUT_DIR)
    with open(path / "ELEC.GEN.WND-US-99.M.json", "w") as f:
        json.dump(energy_generation, f)


def create_irena_capcity_db():
    irena_db = pd.DataFrame(
        {
            "Year": np.arange(2000, 2020, dtype=np.float),
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

    fname = create_folder("irena-database", prefix=INPUT_DIR) / "irena-2020-02-26-1.7.feather"
    irena_db.to_feather(fname)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    setup_logging()
    create_energy_generation()
    create_turbines()
    create_wind_velocity()
    create_irena_capcity_db()
