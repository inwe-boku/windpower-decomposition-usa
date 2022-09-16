import os
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr

from src.util import create_folder
from src.config import INPUT_DIR
from src.config import DATA_DIR
from src.config import OFFSHORE_TURBINES
from src.logging_config import setup_logging


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
    for fname in (
        "uswtdb_v4_1_20210721.csv",
        "uswtdb_v5_0_20220427.csv",
        "uswtdb_v5_1_20220729",
        "uswtdb_v5_1_20220729.csv",
    ):
        # just a static CSV file with one turbine which is actually removed in load_turbines()
        with open(INPUT_DIR / "wind_turbines_usa" / fname, "w") as f:
            f.write(header)
            f.write(turbine_str)

    return turbines


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    setup_logging()
    create_turbines()
