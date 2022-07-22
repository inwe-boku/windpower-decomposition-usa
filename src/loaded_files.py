"""
This module simply calls all load functions. This way a consistent variable naming for the loaded
data can be achieved. At the same time, files are loaded only once due to Python's module caching
mechanism.

Importing of this module might be slow. Use local imports to keep other imports fast.
"""
import logging
import numpy as np

from src.config import AGEING_PER_YEAR
from src.calculations import calc_turbine_age
from src.load_data import load_turbines
from src.load_data import load_p_out_model
from src.load_data import load_p_out_eia
from src.load_data import load_p_out_irena
from src.load_data import load_p_in
from src.load_data import load_rotor_swept_area
from src.load_data import load_is_built


logging.info("Loading files...")

turbines = load_turbines()

p_in = load_p_in()
p_in_avgwind = load_p_in(avgwind=True)
p_in_avgwind_refheight = load_p_in(avgwind=True, refheight=True)

p_out_model = load_p_out_model()
p_out_model_avgwind = load_p_out_model(avgwind=True)
p_out_model_avgwind_refheight = load_p_out_model(avgwind=True, refheight=True)

p_out_model_avgwind_raw = load_p_out_model(avgwind=True, aggregate=False)
p_out_model_raw = load_p_out_model(avgwind=False, aggregate=False)

turbine_age = calc_turbine_age(turbines, p_out_model.time)
age_correction = 1 - turbine_age * AGEING_PER_YEAR
p_out_model_aging_avgwind = (p_out_model_avgwind_raw * age_correction).sum(dim="turbines")
p_out_model_aging = (p_out_model_raw * age_correction).sum(dim="turbines")

p_out_eia = load_p_out_eia()
p_out_irena = load_p_out_irena()

rotor_swept_area = load_rotor_swept_area()
is_built = load_is_built()

# in W/m2
d_out = 1e9 * p_out_model / rotor_swept_area
d_out_avgwind = 1e9 * p_out_model_avgwind / rotor_swept_area

d_in = 1e9 * p_in / rotor_swept_area
d_in_avgwind = 1e9 * p_in_avgwind / rotor_swept_area
d_in_avgwind_refheight = 1e9 * p_in_avgwind_refheight / rotor_swept_area

num_turbines_built = is_built.sum(dim="turbines")

rotor_swept_area_avg = rotor_swept_area / num_turbines_built

efficiency = 100 * p_out_model / p_in
efficiency_avgwind = 100 * p_out_model_avgwind / p_in_avgwind

# in W/m2
specific_power = 1e3 * turbines.t_cap / (turbines.t_rd**2 * np.pi / 4)
specifc_power_per_year = (specific_power * is_built).sum(dim="turbines") / num_turbines_built

total_capacity_kw = (is_built * turbines.t_cap).sum(dim="turbines")
capacity_factors_model = 100 * 1e6 * p_out_model / total_capacity_kw
capacity_factors_model_avgwind = 100 * 1e6 * p_out_model_avgwind / total_capacity_kw
capacity_factors_eia = 100 * 1e6 * load_p_out_eia() / total_capacity_kw

logging.info("Loading files done!")
