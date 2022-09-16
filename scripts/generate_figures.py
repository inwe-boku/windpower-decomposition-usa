import os
import logging

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR
from src.config import YEARS
from src.visualize import savefig
from src.visualize import plot_growth_of_wind_power
from src.visualize import plot_timeseries_figure
from src.visualize import plot_waterfall
from src.visualize import plot_effect_trends_power
from src.visualize import plot_irena_capacity_validation
from src.visualize import plot_missing_uswtdb_data
from src.visualize import plot_scatter_efficiency_input_power_density
from src.visualize import plot_irena_poweroutput_validation
from src.visualize import plot_efficiency_ge1577_example
from src.visualize import plot_efficiency_ge1577_example_zoom
from src.visualize import plot_growth_and_specific_power
from src.visualize import TURBINE_COLORS
from src.load_data import power_curve_ge15_77
from src.load_data import load_wind_speed
from src.load_data import load_turbines
from src.load_data import load_p_out_eia
from src.load_data import load_p_out_irena
from src.load_data import load_p_out_model
from src.load_data import load_p_in
from src.logging_config import setup_logging
from src.calculations import power_input

# https://matplotlib.org/users/usetex.html
# https://matplotlib.org/gallery/userdemo/pgf_texsystem.html
# TODO this is probably the failed try to make matplotlib and latex fonts equal
# plt.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "pgf.rcfonts": False,
#     "pgf.preamble": [
#          r"\usepackage[T1]{fontenc}",
#          r"\usepackage{cmbright}",
#          ]
# })
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['mathtext.fontset'] = 'stixsans'

plt.rcParams["font.size"] = "13"


def save_growth_of_wind_power():
    logging.info("Plotting growth_of_wind_power.pdf...")
    plot_growth_of_wind_power()
    savefig(FIGURES_DIR / "growth_of_wind_power.pdf")


def savefig_growth_and_specific_power():
    logging.info("Plotting growth_and_specific_power.pdf...")
    plot_growth_and_specific_power()
    savefig(FIGURES_DIR / "growth_and_specific_power.pdf")


def save_timeseries():
    from src.figure_params import figure_params

    for figure_param in figure_params:
        plot_timeseries_figure(figure_param)
        logging.info(f"Plotting {figure_param.name}...")
        savefig(FIGURES_DIR / f"{figure_param.name}.pdf")


def savefig_decomposition_powerdensity():
    # TODO this function should be partially moved to visualize.py
    logging.info("Plotting decomposition_powerdensity...")
    from src.loaded_files import rotor_swept_area

    from src.figure_params import d_out_figure_param
    from src.figure_params import d_in_figure_param

    params = (
        ("p_in", "Input power", load_p_in, d_in_figure_param),
        ("p_out", "Output power", load_p_out_model, d_out_figure_param),
    )

    for name, long_name, load_func, density_figure_param in params:
        power = load_func()
        power_avgwind = load_func(avgwind=True)
        power_avgwind_refheight = load_func(avgwind=True, refheight=True)

        baseline = (power_avgwind_refheight / rotor_swept_area * 1e9).mean()

        datasets_with_labels = (
            (
                1e9 * power_avgwind_refheight / rotor_swept_area,
                "Wind power change due to new locations",
                f"{long_name} density at reference height, wind averaged",
                ":",
            ),
            (
                1e9 * power_avgwind / rotor_swept_area,
                "Effect of hub height change",
                f"{long_name} density, wind averaged",
                "--",
            ),
            (
                1e9 * power / rotor_swept_area,
                "Annual variations",
                f"{long_name} density",
                "-",
            ),
        )

        datasets, labels, labels_total, linestyles = zip(*datasets_with_labels)

        for total in (True, False):
            plot_waterfall(
                *datasets,
                labels=labels,
                colors=[TURBINE_COLORS[1], *TURBINE_COLORS[3:5]],
                bottom=baseline,
                total=total,
                labels_total=labels_total,
                linestyles=linestyles,
            )
            plt.axhline(baseline, color="k", linewidth=1)
            plt.ylabel(f"{long_name} density (W/m²)")

            total_str = "_total" if total else ""
            savefig(FIGURES_DIR / f"decomposition_{name}-waterfall{total_str}.pdf")

        plot_effect_trends_power(
            name=name,
            datasets=datasets,
            baseline=baseline,
            labels=labels,
            colors=[TURBINE_COLORS[1], *TURBINE_COLORS[3:5]],
        )
        savefig(FIGURES_DIR / f"decomposition_{name}-effect_trends.pdf")

        # TODO missing FIGSIZE
        fig, axes = plt.subplots(2, figsize=(12, 7.5))

        plot_timeseries_figure(density_figure_param, ax=axes[0], fig=fig)
        plot_effect_trends_power(
            name=name,
            datasets=datasets,
            baseline=baseline,
            labels=labels,
            colors=[TURBINE_COLORS[1], *TURBINE_COLORS[3:5]],
            ax=axes[1],
            fig=fig,
        )
        savefig(FIGURES_DIR / f"{name}_with_trends.pdf")


def savefig_scatter_efficiency_input_power_density():
    logging.info("Plotting scatter_efficiency_input_power_density...")
    plot_scatter_efficiency_input_power_density()
    savefig(FIGURES_DIR / "scatter_efficiency_input_power_density.pdf")


def savefig_irena_capacity_validation():
    from src.loaded_files import turbines

    turbines_with_nans = load_turbines(replace_nan_values="")
    plot_irena_capacity_validation(turbines, turbines_with_nans)
    savefig(FIGURES_DIR / "irena_capacity_validation.pdf")


def savefig_missing_uswtdb_data():
    plot_missing_uswtdb_data()
    savefig(FIGURES_DIR / "missing_uswtdb_data.pdf")


def savefig_irena_poweroutput_validation():
    p_out_eia = load_p_out_eia()
    p_out_irena = load_p_out_irena()

    plot_irena_poweroutput_validation(p_out_eia, p_out_irena)
    savefig(FIGURES_DIR / "irena_poweroutput_validation.pdf")


def save_efficiency_ge1577_example():
    rotor_diameter = 77
    rotor_swept_area = rotor_diameter**2 / 4 * np.pi

    # some arbitrarily selected GE-1.5-77 turbines
    turbine_idcs = [
        # high wind:
        3016844,
        3024179,
        # low wind:
        3014793,
        3017455,
        # unknown:
        3026509,
        3028224,
        3029444,
    ]
    wind_speed = load_wind_speed(YEARS, None).sel(turbines=turbine_idcs).load()

    colors = TURBINE_COLORS[1:] + ("#1b494d",)

    pout_monthly_aggregated = xr.apply_ufunc(power_curve_ge15_77, wind_speed)
    pout_monthly_aggregated = pout_monthly_aggregated.resample(time="1M").mean()

    pin_monthly_aggregated = xr.apply_ufunc(power_input, wind_speed, rotor_swept_area)
    pin_monthly_aggregated = pin_monthly_aggregated.resample(time="1M").mean()

    wind_speed_linspace = np.linspace(0.001, 20, num=500)
    p_out = power_curve_ge15_77(wind_speed_linspace)
    p_in = power_input(wind_speed_linspace, rotor_swept_area)
    c_p = p_out / p_in

    plot_efficiency_ge1577_example(
        wind_speed,
        wind_speed_linspace,
        p_in,
        p_out,
        c_p,
        pout_monthly_aggregated,
        pin_monthly_aggregated,
        rotor_swept_area,
        turbine_idcs,
        colors,
    )
    savefig(FIGURES_DIR / "efficiency_ge1577_example.pdf")

    plot_efficiency_ge1577_example_zoom(
        p_in, rotor_swept_area, c_p, pin_monthly_aggregated, pout_monthly_aggregated, colors
    )
    savefig(FIGURES_DIR / "efficiency_ge1577_example_zoom.pdf")


def save_figures():
    save_timeseries()
    # TODO selected turbine not available in latest USTWDB anylonger
    # save_efficiency_ge1577_example()
    save_growth_of_wind_power()
    savefig_decomposition_powerdensity()
    savefig_irena_capacity_validation()
    savefig_missing_uswtdb_data()
    savefig_scatter_efficiency_input_power_density()
    savefig_irena_poweroutput_validation()
    savefig_growth_and_specific_power()


if __name__ == "__main__":
    try:
        os.mkdir(FIGURES_DIR)
    except FileExistsError:
        logging.debug("Output folder already exists.")

    setup_logging()
    save_figures()
