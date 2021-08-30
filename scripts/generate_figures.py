import os
import logging

import xarray as xr
import matplotlib.pyplot as plt

from src.util import filter2010
from src.logging_config import setup_logging
from src.config import FIGURES_DIR
from src.config import OUTPUT_DIR
from src.constants import HOURS_PER_YEAR
from src.visualize import savefig
from src.visualize import plot_growth_of_wind_power
from src.visualize import plot_decomposition_p_out
from src.visualize import plot_waterfall
from src.visualize import plot_effect_trends_pin
from src.visualize import plot_irena_capacity_validation
from src.visualize import plot_missing_uswtdb_data
from src.visualize import plot_scatter_efficiency_input_power_density
from src.visualize import plot_system_effiency
from src.visualize import plot_capacity_factors
from src.visualize import plot_irena_poweroutput_validation
from src.visualize import TURBINE_COLORS
from src.load_data import load_turbines
from src.load_data import load_generated_energy_gwh
from src.load_data import load_generated_energy_gwh_yearly
from src.load_data import load_generated_energy_gwh_yearly_irena
from src.calculations import fit_efficiency_model

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
    plot_growth_of_wind_power()
    savefig(FIGURES_DIR / "growth_of_wind_power.pdf")


def savefig_decomposition_pout():
    generated_energy_gwh = load_generated_energy_gwh()
    generated_energy_gwh_yearly = load_generated_energy_gwh_yearly()

    rotor_swept_area = xr.load_dataarray(
        OUTPUT_DIR / "turbine-time-series" / "rotor_swept_area_yearly.nc"
    )

    rotor_swept_area_monthly = xr.load_dataarray(
        OUTPUT_DIR / "turbine-time-series" / "rotor_swept_area.nc"
    )

    p_in = xr.open_dataarray(OUTPUT_DIR / "power_in_wind" / "p_in.nc")
    p_in_avg = xr.open_dataarray(OUTPUT_DIR / "power_in_wind" / "p_in_avg.nc")
    p_in_avg80 = xr.open_dataarray(OUTPUT_DIR / "power_in_wind" / "p_in_avg80.nc")
    p_in_monthly = xr.open_dataarray(OUTPUT_DIR / "power_in_wind" / "p_in_monthly.nc")

    is_built_yearly = xr.open_dataarray(OUTPUT_DIR / "turbine-time-series" / "is_built_yearly.nc")

    num_turbines_built = is_built_yearly.sum(dim="turbines")

    args = (
        num_turbines_built,
        rotor_swept_area,
        generated_energy_gwh_yearly,
        p_in,
        p_in_avg,
        p_in_avg80,
        generated_energy_gwh,
        p_in_monthly,
        rotor_swept_area_monthly,
    )

    savefig_decomposition_pout_timesel("", *args)
    savefig_decomposition_pout_timesel("_from_2010", *(filter2010(arg) for arg in args))


def savefig_decomposition_pout_timesel(
    fname_postfix,
    num_turbines_built,
    rotor_swept_area,
    generated_energy_gwh_yearly,
    p_in,
    p_in_avg,
    p_in_avg80,
    generated_energy_gwh,
    p_in_monthly,
    rotor_swept_area_monthly,
):
    plot_decomposition_p_out(
        num_turbines_built,
        rotor_swept_area,
        generated_energy_gwh_yearly,
        p_in,
        p_in_avg,
        p_in_avg80,
        generated_energy_gwh,
        p_in_monthly,
        rotor_swept_area_monthly,
        plot_only=[
            "rotor_swept_area",
            "number",
            "efficiency",
            "powerdensity",
        ],
    )

    savefig(FIGURES_DIR / f"decomposition_power_generation{fname_postfix}.pdf")

    for graph_name in (
        "rotor_swept_area",
        "number",
        "efficiency",
        "powerdensity",
        [
            "powerdensity",
            "powerdensity-avg",
        ],
        [
            "powerdensity",
            "powerdensity-avg",
            "powerdensity-avg-80",
        ],
        "outputpowerdensity",
    ):
        if isinstance(graph_name, list):
            plot_only = graph_name
            graph_name = "-".join(graph_name)
        else:
            plot_only = [graph_name]
        # if graph_name == "powerdensity-avg-80":
        #    plot_only = ["powerdensity", "powerdensity-avg", "powerdensity-avg-80"]
        logging.info(f"Plotting {graph_name}...")
        plot_decomposition_p_out(
            num_turbines_built,
            rotor_swept_area,
            generated_energy_gwh_yearly,
            p_in,
            p_in_avg,
            p_in_avg80,
            generated_energy_gwh,
            p_in_monthly,
            rotor_swept_area_monthly,
            plot_only=plot_only,
            absolute_plot=True,
        )
        savefig(FIGURES_DIR / f"decomposition_power_generation-{graph_name}{fname_postfix}.pdf")


def savefig_decomposition_pin():
    rotor_swept_area = xr.load_dataarray(
        OUTPUT_DIR / "turbine-time-series" / "rotor_swept_area_yearly.nc"
    )

    p_in = xr.open_dataarray(OUTPUT_DIR / "power_in_wind" / "p_in.nc")
    p_in_avg = xr.open_dataarray(OUTPUT_DIR / "power_in_wind" / "p_in_avg.nc")
    p_in_avg80 = xr.open_dataarray(OUTPUT_DIR / "power_in_wind" / "p_in_avg80.nc")

    savefig_decomposition_pin_timesel(
        rotor_swept_area, p_in, p_in_avg, p_in_avg80, fname_postfix=""
    )
    savefig_decomposition_pin_timesel(
        filter2010(rotor_swept_area),
        filter2010(p_in),
        filter2010(p_in_avg),
        filter2010(p_in_avg80),
        fname_postfix="_from_2010",
    )


def savefig_scatter_efficiency_input_power_density():
    logging.info("Plotting scatter_efficiency_input_power_density...")

    rotor_swept_area_monthly = xr.load_dataarray(
        OUTPUT_DIR / "turbine-time-series" / "rotor_swept_area.nc"
    )
    p_in_monthly = xr.open_dataarray(OUTPUT_DIR / "power_in_wind" / "p_in_monthly.nc")

    # p_out_monthly is in GW, same as p_in_monthly
    p_out_monthly = load_generated_energy_gwh()
    p_out_monthly = p_out_monthly / p_out_monthly.time.dt.days_in_month / 24
    p_out_monthly = p_out_monthly.sortby("time")

    efficiency_monthly = 100 * p_out_monthly / p_in_monthly

    plot_scatter_efficiency_input_power_density(
        p_in_monthly, rotor_swept_area_monthly, efficiency_monthly
    )
    savefig(FIGURES_DIR / "scatter_efficiency_input_power_density.pdf")

    plot_scatter_efficiency_input_power_density(
        filter2010(p_in_monthly),
        filter2010(rotor_swept_area_monthly),
        filter2010(efficiency_monthly),
    )
    savefig(FIGURES_DIR / "scatter_efficiency_input_power_density_from_2010.pdf")


def savefig_decomposition_pin_timesel(rotor_swept_area, p_in, p_in_avg, p_in_avg80, fname_postfix):
    baseline = (p_in_avg80 / rotor_swept_area * 1e9).mean()

    datasets_with_labels = (
        (
            1e9 * p_in_avg80 / rotor_swept_area,
            "Wind change due to new locations",
            "Input power density at 76m, wind averaged",
            ":",
        ),
        (
            1e9 * p_in_avg / rotor_swept_area,
            "Effect of hub height change",
            "Input power density, wind averaged",
            "--",
        ),
        (
            1e9 * p_in / rotor_swept_area,
            "Annual variations",
            "Input power density",
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
        plt.ylabel("Input power density (W/m²)")

        total_str = "_total" if total else ""
        savefig(FIGURES_DIR / f"decomposition_pin-waterfall{total_str}{fname_postfix}.pdf")

    plot_effect_trends_pin(
        datasets=datasets,
        baseline=baseline,
        labels=labels,
        colors=[TURBINE_COLORS[1], *TURBINE_COLORS[3:5]],
    )
    savefig(FIGURES_DIR / f"decomposition_pin-effect_trends{fname_postfix}.pdf")


def savefig_irena_capacity_validation():
    turbines = load_turbines()
    turbines_with_nans = load_turbines(replace_nan_values="")
    plot_irena_capacity_validation(turbines, turbines_with_nans)
    savefig(FIGURES_DIR / "irena_capacity_validation.pdf")


def savefig_missing_uswtdb_data():
    plot_missing_uswtdb_data()
    savefig(FIGURES_DIR / "missing_uswtdb_data.pdf")


def _calc_system_efficiency(monthly=False):
    rotor_swept_area = xr.load_dataarray(
        OUTPUT_DIR / "turbine-time-series" / f"rotor_swept_area{'' if monthly else '_yearly'}.nc"
    )

    p_in = xr.load_dataarray(
        OUTPUT_DIR / "power_in_wind" / f"p_in{'_monthly' if monthly else ''}.nc"
    )
    p_in = p_in.sortby("time")

    if monthly:
        p_out = load_generated_energy_gwh()
        p_out = p_out / p_out.time.dt.days_in_month / 24
    else:
        p_out = load_generated_energy_gwh_yearly()
        p_out = p_out / HOURS_PER_YEAR
    p_out = p_out.sortby("time")

    p_in = filter2010(p_in)
    p_out = filter2010(p_out)
    rotor_swept_area = filter2010(rotor_swept_area)

    efficiency = p_out / p_in
    p_in_density = p_in / rotor_swept_area * 1e9

    fit_result, efficiency_without_pin = fit_efficiency_model(
        p_in, p_out, p_in_density, efficiency
    )

    if monthly:
        resample_params = dict(time="1A", label="left", loffset="1D")
        efficiency_without_pin = efficiency_without_pin.resample(**resample_params).mean()
        efficiency = efficiency.resample(**resample_params).mean()

    return efficiency, efficiency_without_pin


def savefig_system_efficiency():
    efficiency, efficiency_without_pin = _calc_system_efficiency()

    plot_system_effiency(efficiency, efficiency_without_pin)
    savefig(FIGURES_DIR / "system_efficiency.pdf")

    plot_system_effiency(
        efficiency, efficiency_without_pin, *_calc_system_efficiency(monthly=True)
    )
    savefig(FIGURES_DIR / "system_efficiency_with_monthly_avg_ratios.pdf")


def savefig_capacity_factors():
    generated_energy_gwh_yearly = load_generated_energy_gwh_yearly()
    turbines = load_turbines()
    is_built_yearly = xr.open_dataarray(OUTPUT_DIR / "turbine-time-series" / "is_built_yearly.nc")

    plot_capacity_factors(turbines, generated_energy_gwh_yearly, is_built_yearly)

    savefig(FIGURES_DIR / "capacity_factors.pdf")


def savefig_irena_poweroutput_validation():
    generated_energy_gwh_yearly = load_generated_energy_gwh_yearly()
    p_out_eia = xr.DataArray(
        generated_energy_gwh_yearly,
        dims="year",
        coords={"year": generated_energy_gwh_yearly.time.dt.year.values},
    )
    p_out_irena = load_generated_energy_gwh_yearly_irena()

    plot_irena_poweroutput_validation(p_out_eia, p_out_irena)
    savefig(FIGURES_DIR / "irena_poweroutput_validation.pdf")


def save_figures():
    savefig_system_efficiency()
    save_growth_of_wind_power()
    savefig_decomposition_pout()
    savefig_decomposition_pin()
    savefig_irena_capacity_validation()
    savefig_missing_uswtdb_data()
    savefig_scatter_efficiency_input_power_density()
    savefig_capacity_factors()
    savefig_irena_poweroutput_validation()


if __name__ == "__main__":
    try:
        os.mkdir(FIGURES_DIR)
    except FileExistsError:
        logging.debug("Output folder already exists.")

    setup_logging()
    save_figures()
