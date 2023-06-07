from src.util import create_folder
from src.config import OUTPUT_DIR

import src.loaded_files
from src.logging_config import setup_logging


setup_logging()


output_folder = create_folder("results", prefix=OUTPUT_DIR)

variables = [
    ("p_in", "Power input (actual wind conditions, hub height)", "GW"),
    ("p_in_avgwind", "Power input (long-term average wind conditions, hub height)", "GW"),
    (
        "p_in_avgwind_refheight",
        "Power input (long-term average wind conditions, reference height)",
        "GW",
    ),
    ("p_out_model", "Power output (actual wind conditions, hub height)", "GW"),
    ("p_out_model_avgwind", "Power output (long-term average wind conditions, hub height)", "GW"),
    (
        "p_out_model_avgwind_refheight",
        "Power output (long-term average wind conditions, reference height)",
        "GW",
    ),
    (
        "p_out_model_aging",
        "Power output (actual wind conditions, hub height, aging loss subtracted)",
        "GW",
    ),
    (
        "p_out_model_aging_avgwind",
        "Power output (long-term average wind conditions, hub height, aging loss subtracted)",
        "GW",
    ),
    ("d_out", "Input power density (actual wind conditions, hub height)", "W/m^2"),
    (
        "d_out_avgwind",
        "Input power density (long-term average wind conditions, hub height)",
        "W/m^2",
    ),
    ("d_in", "Input power density (actual wind conditions, hub height)", "W/m^2"),
    (
        "d_in_avgwind",
        "Input power density (long-term average wind conditions, hub height)",
        "W/m^2",
    ),
    (
        "d_in_avgwind_refheight",
        "Input power density (long-term average wind conditions, reference height)",
        "W/m^2",
    ),
    ("num_turbines_built", "Number of operating turbines", "dimensionless"),
    ("rotor_swept_area", "Total rotor swept area of operating turbines", "m^2"),
    ("rotor_swept_area_avg", "Average rotor swept area for operating turbines", "m^2"),
    (
        "efficiency",
        "System efficiency of operating turbines, i.e. P_out/P_in (actual wind conditions)",
        "dimensionless",
    ),
    (
        "efficiency_avgwind",
        "System efficiency of operating turbines, i.e. P_out,avgwind/P_in,avgwind (long-term "
        "average wind conditions)",
        "dimensionless",
    ),
    (
        "specific_power_per_year",  # TODO rename this to avg_specific_power
        "Average specific power of operating turbines per year",
        "W/m^2",
    ),
    ("total_capacity_kw", "Total capacity of operating turbines", "KW"),
    (
        "capacity_factors_model",
        "Capacity factors (simulation using power curve model, actual wind conditions)",
        "%",
    ),
    (
        "capacity_factors_model_avgwind",
        "Capacity factors (simulation using power curve model, long-term average wind conditions)",
        "%",
    ),
    ("capacity_factors_eia", "Capacity factors (using observation data provided by EIA)", "%"),
    ("avg_hubheights", "Average hub heights of operating turbines", "m"),
]


def print_line(name, long_name, unit, char=" "):
    print(
        f"| {name}".ljust(100, char),
        f"| {long_name}".ljust(110, char),
        f"| {unit}".ljust(17, char),
        "|",
    )


print_line("File", "Description", "Unit")
print_line("", "", "", char="-")

for name, long_name, unit in variables:
    variable = getattr(src.loaded_files, name)
    variable.attrs["long_name"] = long_name
    variable.attrs["unit"] = unit
    variable.to_netcdf(output_folder / f"{name}.nc")

    print_line(f"[{name}.nc](data/output/results/{name}.nc)", long_name, unit)
