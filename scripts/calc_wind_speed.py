import logging
from dask.diagnostics import ProgressBar

from src.config import MONTHS
from src.config import YEARS
from src.config import REFERENCE_HUB_HEIGHT_M
from src.util import create_folder
from src.load_data import (
    load_turbines,
    load_generated_energy_gwh,
)
from src.load_data import load_wind_velocity
from src.calculations import calc_wind_speed_at_turbines

from src.logging_config import setup_logging


setup_logging()

turbines = load_turbines()
generated_energy_gwh = load_generated_energy_gwh()


output_folder = create_folder("wind_speed")


for height in (None, REFERENCE_HUB_HEIGHT_M):
    logging.info(f"Calculating wind speed at turbines with height={height}...")

    height_name = "hubheight" if height is None else height

    for year in YEARS:
        logging.info(f"year={year}...")
        wind_velocity = load_wind_velocity(year, MONTHS)
        with ProgressBar():
            wind_speed = calc_wind_speed_at_turbines(wind_velocity, turbines, height)
            wind_speed.to_netcdf(output_folder / f"wind_speed_height_{height_name}_{year}.nc")
