import logging
from src.util import create_folder
from src.config import OUTPUT_DIR
from src.load_data import load_turbines
from src.load_data import load_capacity_irena
from src.load_data import load_p_out_eia_monthly
from src.load_data import load_p_out_eia
from src.calculations import calc_is_built
from src.calculations import calc_rotor_swept_area
from src.calculations import calc_irena_correction_factor

from src.logging_config import setup_logging


setup_logging()

turbines = load_turbines()

time = load_p_out_eia_monthly().time
time_yearly = load_p_out_eia().time

output_folder = create_folder("turbine-time-series", prefix=OUTPUT_DIR)

rotor_swept_area = calc_rotor_swept_area(turbines, time=time)
rotor_swept_area.to_netcdf(output_folder / "rotor_swept_area.nc")

rotor_swept_area_yearly = calc_rotor_swept_area(turbines, time=time_yearly)
rotor_swept_area_yearly.to_netcdf(output_folder / "rotor_swept_area_yearly.nc")

is_built = calc_is_built(turbines, time)
is_built_yearly = calc_is_built(turbines, time_yearly)

is_built.to_netcdf(output_folder / "is_built.nc")
is_built_yearly.to_netcdf(output_folder / "is_built_yearly.nc")

logging.info("Calculate irena_correction_factor...")
capacity_irena = load_capacity_irena()
irena_correction_factor = calc_irena_correction_factor(turbines, capacity_irena)
irena_correction_factor.to_netcdf(output_folder / "irena_correction_factor.nc")
