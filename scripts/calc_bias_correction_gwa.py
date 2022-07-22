import logging
from dask.diagnostics import ProgressBar

from src.util import create_folder
from src.config import OUTPUT_DIR
from src.config import MONTHS
from src.load_data import load_turbines
from src.load_data import load_wind_velocity
from src.load_data import load_wind_speed_gwa
from src.calculations import calc_bias_correction

from src.logging_config import setup_logging


setup_logging()

turbines = load_turbines()

# XXX move time range to config or use YEARS!
wind_velocity_era5 = load_wind_velocity(range(2010, 2019), MONTHS)


for height in (50, 100, 200):
    logging.info(f"Computing BIAS correction using height={height}...")
    wind_speed_gwa = load_wind_speed_gwa(height=height)
    fname = (
        create_folder("bias_correction", prefix=OUTPUT_DIR)
        / f"bias_correction_factors_gwa2_{height}m.nc"
    )

    bias_correction = calc_bias_correction(turbines, wind_velocity_era5, wind_speed_gwa, height)

    logging.info("Now compute...")
    with ProgressBar():
        bias_correction = bias_correction.compute()

    logging.info(f"Saving result to NetCDF at {fname}...")
    bias_correction.to_netcdf(fname)
