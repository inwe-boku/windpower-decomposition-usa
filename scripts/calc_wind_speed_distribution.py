from src.util import create_folder
from src.config import YEARS
from src.config import OUTPUT_DIR
from src.load_data import load_turbines
from src.load_data import load_wind_speed
from src.logging_config import setup_logging
from src.calculations import calc_wind_speed_distribution


setup_logging()

turbines = load_turbines()
wind_speed = load_wind_speed(YEARS, None)

bins = 100

wind_speed_distribution = calc_wind_speed_distribution(
    turbines,
    wind_speed,
    bins=100,
    chunk_size=800,
)

output_folder = create_folder("wind_speed_distribution", prefix=OUTPUT_DIR)

wind_speed_distribution.to_netcdf(output_folder / "wind_speed_distribution.nc")
