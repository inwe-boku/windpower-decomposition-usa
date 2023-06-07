"""
This module serves as configuration for figures. Importing of the file might be slow, because a lot
of data needs to be loaded.

Importing of this module might be slow. Use local imports to keep other imports fast.
"""

from copy import deepcopy
from collections import namedtuple

import numpy as np

from src.config import LOSS_CORRECTION_FACTOR
from src.visualize import TURBINE_COLORS
from src.load_data import load_p_in
from src.load_data import load_p_out_model
from src.load_data import load_d_out_miller

from src.loaded_files import d_out
from src.loaded_files import d_out_avgwind
from src.loaded_files import d_in
from src.loaded_files import d_in_avgwind
from src.loaded_files import p_in
from src.loaded_files import p_in_avgwind
from src.loaded_files import p_out_eia
from src.loaded_files import p_out_irena
from src.loaded_files import p_out_model_aging_avgwind
from src.loaded_files import p_out_model_aging
from src.loaded_files import rotor_swept_area
from src.loaded_files import efficiency
from src.loaded_files import efficiency_avgwind
from src.loaded_files import num_turbines_built
from src.loaded_files import rotor_swept_area_avg
from src.loaded_files import capacity_factors_model
from src.loaded_files import capacity_factors_model_avgwind
from src.loaded_files import capacity_factors_eia


LineParam = namedtuple("LineParam", "data label color linestyle linewidth", defaults=[None, None])
FigureParam = namedtuple("FigureParam", "name unit lines absolute_plot", defaults=[True])


efficiency_line_param = LineParam(
    data=efficiency,
    label="System efficiency $\\frac{P_\\mathrm{out}}{P_\\mathrm{in}}$",
    color=TURBINE_COLORS[3],
)

d_in_line_param = LineParam(
    data=d_in,
    label="Input power density $\\frac{P_\\mathrm{in}}{A}$",
    color=TURBINE_COLORS[4],
)

p_out_decomposition_figure_param = FigureParam(
    name="p_out_decomposition",
    unit="%",
    absolute_plot=False,
    lines=[
        LineParam(
            rotor_swept_area_avg,
            label="Average rotor swept area $A$",
            color=TURBINE_COLORS[1],
        ),
        LineParam(
            data=num_turbines_built * 1e-3,
            label="Number of operating turbines $N$",
            # unit="in thousands",
            color=TURBINE_COLORS[2],
        ),
        efficiency_line_param,
        d_in_line_param,
    ],
)

d_in_figure_param = FigureParam(
    name="d_in",
    unit="Input power density (W/m²)",
    lines=[
        LineParam(
            data=d_in_avgwind,
            label="Without annual variations in wind conditions",
            color=TURBINE_COLORS[4],
        ),
        d_in_line_param._replace(
            linestyle="--",
            label="With annual variations in wind conditions",
        ),
        # TODO do we need refheight here?
    ],
)

d_out_figure_param = FigureParam(
    name="d_out",
    unit="Output power density (W/m²)",
    lines=[
        LineParam(
            data=d_out_avgwind,
            label="Without annual variations in wind conditions",
            color=TURBINE_COLORS[4],
        ),
        LineParam(
            data=d_out,
            label="With annual variations in wind conditions",
            color=TURBINE_COLORS[4],
            linestyle="--",
        ),
    ],
)

specific_power_avg = 300
d_out_refspecpower_figure_param = deepcopy(d_out_figure_param)._replace(name="d_out_refspecpower")
d_out_refspecpower_figure_param.lines.extend(
    [
        LineParam(
            data=1e9 * load_p_out_model(refspecpower=specific_power_avg) / rotor_swept_area,
            label="Constant specific power, with annual variations in wind conditions",
            color="#000000",
            linestyle="--",
            linewidth=0.7,
        ),
        LineParam(
            data=1e9
            * load_p_out_model(refspecpower=specific_power_avg, avgwind=True)
            / rotor_swept_area,
            label="Constant specific power, without annual variations in wind conditions",
            color="#000000",
            linestyle="-",
            linewidth=0.7,
        ),
    ]
)

efficiency_figure_param = FigureParam(
    name="system_efficiency",
    unit="System efficiency (%)",
    lines=[
        LineParam(
            data=efficiency_avgwind,
            label="Without annual variations in wind conditions",
            color=TURBINE_COLORS[3],
        ),
        efficiency_line_param._replace(
            linestyle="--",
            label="With annual variations in wind conditions",
        ),
        # TODO do we need refheight here?
    ],
)

efficiency_refspecpower_figure_param = deepcopy(efficiency_figure_param)._replace(
    name="efficiency_refspecpower"
)
efficiency_refspecpower_figure_param.lines.extend(
    [
        LineParam(
            data=100 * load_p_out_model(refspecpower=specific_power_avg) / p_in,
            label="Constant specific power, with annual variations in wind conditions",
            color="#000000",
            linestyle="--",
            linewidth=0.7,
        ),
        LineParam(
            data=100
            * load_p_out_model(refspecpower=specific_power_avg, avgwind=True)
            / p_in_avgwind,
            label="Constant specific power, without annual variations in wind conditions",
            color="#000000",
            linestyle="-",
            linewidth=0.7,
        ),
    ]
)


capacity_factors_param = FigureParam(
    name="capacity_factors",
    unit="Capacity factor (%)",
    lines=[
        LineParam(
            data=capacity_factors_model_avgwind,
            label="Simulation using power curve model, without annual variations in wind "
            "conditions",
            color=TURBINE_COLORS[3],
        ),
        LineParam(
            data=capacity_factors_model,
            label="Simulation using power curve model, with annual variations in wind conditions",
            linestyle="--",
            color=TURBINE_COLORS[3],
        ),
        LineParam(
            data=capacity_factors_eia,
            label="Observation data provided by EIA",
            color=TURBINE_COLORS[4],
        ),
    ],
)


d_out_validation_figure_param = deepcopy(d_out_figure_param)
d_out_validation_figure_param = d_out_validation_figure_param._replace(name="d_out_validation")
d_out_validation_figure_param.lines[0] = d_out_validation_figure_param.lines[0]._replace(
    label="Simulation using power curve model, without annual variations in wind conditions",
)
d_out_validation_figure_param.lines[1] = d_out_validation_figure_param.lines[1]._replace(
    label="Simulation using power curve model, with annual variations in wind conditions",
)
d_out_validation_figure_param.lines.extend(
    [
        LineParam(
            data=1e9 * p_out_eia / rotor_swept_area,
            label="Observation data provided by the EIA",
            color="#ba9f7c",
            linestyle="-",
            linewidth=0.8,
        ),
        LineParam(
            data=1e9 * p_out_irena / rotor_swept_area,
            label="Observation data provided by IRENA",
            color="#f0c220",
            linestyle="-",
            linewidth=0.8,
        ),
        LineParam(
            data=6 * 13.5 / np.pi * 4 * load_d_out_miller(),
            label="Calculated from land use power density (Miller et al., 2019)",
            color="#1b494d",
            linestyle="-",
            linewidth=0.8,
        ),
    ]
)

efficiency_validation_figure_param = deepcopy(efficiency_figure_param)
efficiency_validation_figure_param = efficiency_validation_figure_param._replace(
    name="efficiency_validation"
)
efficiency_validation_figure_param.lines[0] = efficiency_validation_figure_param.lines[0]._replace(
    label="Simulation using power curve model, without annual variations in wind conditions"
)
efficiency_validation_figure_param.lines[1] = efficiency_validation_figure_param.lines[1]._replace(
    label="Simulation using power curve model, with annual variations in wind conditions",
)
efficiency_validation_figure_param.lines.extend(
    [
        LineParam(
            data=100 * p_out_eia / p_in,
            label="Observation data provided by the EIA",
            color="#ba9f7c",
            linestyle="-",
            linewidth=0.8,
        ),
        LineParam(
            data=100 * p_out_irena / p_in,
            label="Observation data provided by IRENA",
            color="#f0c220",
            linestyle="-",
            linewidth=0.8,
        ),
    ]
)

average_cp_figure_param = deepcopy(efficiency_figure_param)
average_cp_figure_param = average_cp_figure_param._replace(name="average_cp")
average_cp_figure_param = average_cp_figure_param._replace(unit="%")
average_cp_figure_param.lines[0] = average_cp_figure_param.lines[0]._replace(
    label="System efficiency, without annual variations in wind conditions"
)
average_cp_figure_param.lines[1] = average_cp_figure_param.lines[1]._replace(
    label="System efficiency, with annual variations in wind conditions",
)
p_out_model_raw = load_p_out_model(aggregate=False)
p_in_raw = load_p_in(aggregate=False)
p_out_model_avgwind_raw = load_p_out_model(aggregate=False, avgwind=True)
p_in_avgwind_raw = load_p_in(aggregate=False, avgwind=True)
average_cp_figure_param.lines.extend(
    [
        LineParam(
            data=(100 * p_out_model_avgwind_raw / LOSS_CORRECTION_FACTOR / p_in_avgwind_raw).mean(
                dim="turbines"
            ),
            label="Average $C_p$, without annual variations in wind conditions",
            color="#19484c",
            linestyle="-",
        ),
        LineParam(
            data=(100 * p_out_model_raw / LOSS_CORRECTION_FACTOR / p_in_raw).mean(dim="turbines"),
            label="Average $C_p$, with annual variations in wind conditions",
            color="#19484c",
            linestyle="--",
        ),
    ]
)


efficiency_aging_figure_param = deepcopy(efficiency_figure_param)
efficiency_aging_figure_param = efficiency_aging_figure_param._replace(name="efficiency_aging")
efficiency_aging_figure_param.lines.extend(
    [
        LineParam(
            data=100 * p_out_model_aging_avgwind / p_in_avgwind,
            label="Aging loss subtracted, without annual variations in wind conditions",
            color="#7a6952",
        ),
        LineParam(
            data=100 * p_out_model_aging / p_in,
            label="Aging loss subtracted, with annual variations in wind conditions",
            color="#7a6952",
            linestyle="--",
        ),
    ]
)

d_out_aging_figure_param = deepcopy(d_out_figure_param)
d_out_aging_figure_param = d_out_aging_figure_param._replace(name="d_out_aging")
d_out_aging_figure_param.lines.extend(
    [
        LineParam(
            data=1e9 * p_out_model_aging_avgwind / rotor_swept_area,
            label="Aging loss subtracted, without annual variations in wind conditions",
            color="#7a6952",
        ),
        LineParam(
            data=1e9 * p_out_model_aging / rotor_swept_area,
            label="Aging loss subtracted, with annual variations in wind conditions",
            color="#7a6952",
            linestyle="--",
        ),
    ]
)


figure_params = [
    p_out_decomposition_figure_param,
    d_out_refspecpower_figure_param,
    efficiency_refspecpower_figure_param,
    d_in_figure_param,
    d_out_figure_param,
    efficiency_figure_param,
    capacity_factors_param,
    d_out_validation_figure_param,
    efficiency_validation_figure_param,
    average_cp_figure_param,
    efficiency_aging_figure_param,
    d_out_aging_figure_param,
]
