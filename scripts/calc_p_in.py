from src.wind_power import calc_p_in
from src.wind_power import calc_power
from src.logging_config import setup_logging

setup_logging()
calc_power(compute_func=calc_p_in, name="p_in")
