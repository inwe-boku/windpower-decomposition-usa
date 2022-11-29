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
from src.util import create_folder
from src.constants import HOURS_PER_YEAR
from src.load_data import load_turbines
from src.logging_config import setup_logging
from src.calculations import calc_bounding_box_usa


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


def create_power_curve_model():
    fname = create_folder("power_curve_model", prefix=INPUT_DIR) / "table_a_b_constants.csv"

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
    create_wind_velocity()
    create_irena_capcity_db()
    create_biascorrection()
    create_power_curve_model()
