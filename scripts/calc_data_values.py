import numpy as np
import xarray as xr

from src.config import OUTPUT_DIR
from src.constants import HOURS_PER_YEAR
from src.constants import KILO_TO_ONE
from src.util import filter2010
from src.util import write_data_value
from src.util import nanratio
from src.util import calc_abs_slope
from src.load_data import load_generated_energy_gwh
from src.load_data import load_generated_energy_gwh_yearly
from src.load_data import load_turbines
from src.calculations import calc_rotor_swept_area
from src.calculations import calc_is_built


def calc_correlation_efficiency_vs_input_power_density():
    rotor_swept_area = xr.load_dataarray(
        OUTPUT_DIR / "turbine-time-series" / "rotor_swept_area.nc"
    )

    p_in = xr.load_dataarray(OUTPUT_DIR / "power_in_wind" / "p_in_monthly.nc")
    p_in = p_in.sortby("time")

    p_out = load_generated_energy_gwh()
    p_out = p_out / p_out.time.dt.days_in_month / 24
    p_out = p_out.sortby("time")

    p_in = filter2010(p_in)
    p_out = filter2010(p_out)
    rotor_swept_area = filter2010(rotor_swept_area)

    efficiency = p_out / p_in
    p_in_density = p_in / rotor_swept_area * 1e9

    correlation = np.corrcoef(p_in_density, efficiency)[0, 1]

    write_data_value(
        "correlation-efficiency-vs-input-power-density",
        f"{correlation:.3f}",
    )


def number_of_turbines():
    turbines = load_turbines()
    (turbines.p_year <= 2010).sum().compute()

    write_data_value(
        "number-of-turbines-start",
        f"{(turbines.p_year <= 2010).sum().values:,d}",
    )
    write_data_value(
        "number-of-turbines-end",
        f"{(turbines.p_year <= 2019).sum().values:,d}",
    )


def rotor_swept_area_avg():
    rotor_swept_area = xr.load_dataarray(
        OUTPUT_DIR / "turbine-time-series" / "rotor_swept_area_yearly.nc"
    )

    turbines = load_turbines()
    time = rotor_swept_area.time

    calc_rotor_swept_area(turbines, time)
    is_built = calc_is_built(turbines, time)

    rotor_swept_area_avg = (
        calc_rotor_swept_area(turbines, time) / is_built.sum(dim="turbines")
    ).compute()

    write_data_value(
        "rotor_swept_area_avg-start",
        f"{int(rotor_swept_area_avg.sel(time='2010').values.round()):,d}",
    )
    write_data_value(
        "rotor_swept_area_avg-end",
        f"{int(rotor_swept_area_avg.sel(time='2019').values.round()):,d}",
    )


def missing_commissioning_year():
    turbines = load_turbines()
    turbines_with_nans = load_turbines(replace_nan_values="")
    write_data_value(
        "percentage_missing_commissioning_year",
        f"{nanratio(turbines_with_nans.p_year).values * 100:.1f}",
    )

    missing2010 = (
        np.isnan(turbines_with_nans.p_year).sum() / (turbines_with_nans.p_year <= 2010).sum()
    ).values

    write_data_value(
        "percentage_missing_commissioning_year_2010",
        f"{missing2010 * 100:.1f}",
    )

    write_data_value(
        "num_available_decommissioning_year",
        f"{(~np.isnan(turbines.d_year)).sum().values:,d}",
    )
    write_data_value(
        "num_decommissioned_turbines",
        f"{turbines.is_decomissioned.sum().values:,d}",
    )

    lifetime = 25
    num_further_old_turbines = (
        (turbines.sel(turbines=~turbines.is_decomissioned).p_year < (2019 - lifetime)).sum().values
    )
    write_data_value(
        "num_further_old_turbines",
        f"{num_further_old_turbines:,d}",
    )

    write_data_value(
        "missing_ratio_rd_hh",
        f"{100 * nanratio(turbines_with_nans.t_hh + turbines_with_nans.t_rd).values:.1f}",
    )


def calculate_slopes():
    p_in = filter2010(xr.open_dataarray(OUTPUT_DIR / "power_in_wind" / "p_in.nc"))
    generated_energy_gwh_yearly = filter2010(load_generated_energy_gwh_yearly())
    rotor_swept_area = filter2010(
        xr.load_dataarray(OUTPUT_DIR / "turbine-time-series" / "rotor_swept_area_yearly.nc")
    )
    is_built_yearly = xr.open_dataarray(OUTPUT_DIR / "turbine-time-series" / "is_built_yearly.nc")

    num_turbines_built = filter2010(is_built_yearly.sum(dim="turbines"))

    data = {
        "outputpowerdensity": (
            1e9 * generated_energy_gwh_yearly / HOURS_PER_YEAR / rotor_swept_area
        ),
        "inputpowerdensity": 1e9 * p_in / rotor_swept_area,
        "rotor_swept_area_avg": rotor_swept_area / num_turbines_built,
        "efficiency": 100 * generated_energy_gwh_yearly / p_in / HOURS_PER_YEAR,
    }
    # do not forget to filter2010!

    for key, values in data.items():
        relative_to_2010 = 100 * values / values[0]
        write_data_value(
            f"{key}_relative_abs_slope",
            f"{calc_abs_slope(relative_to_2010):.1f}",
        )

    outputpowerdensity = data["outputpowerdensity"].values
    write_data_value(
        "outputpowerdensity-start",
        f"{outputpowerdensity[0]:.0f}",
    )
    write_data_value(
        "outputpowerdensity-end",
        f"{outputpowerdensity[-1]:.0f}",
    )
    write_data_value(
        "outputpowerdensity_abs_slope",
        f"{calc_abs_slope(outputpowerdensity):.1f}",
    )

    percentage_poweroutput_per_area = 100 * outputpowerdensity[-1] / outputpowerdensity[0]
    write_data_value(
        "percentage_poweroutput_per_area",
        f"{percentage_poweroutput_per_area:.0f}",
    )
    write_data_value(
        "less_poweroutput_per_area",
        f"{100 - percentage_poweroutput_per_area:.0f}",
    )

    growth_num_turbines_built = num_turbines_built[-1] / num_turbines_built[0] * 100
    write_data_value(
        "growth_num_turbines_built",
        f"{growth_num_turbines_built.values:.0f}",
    )
    rotor_swept_area_avg = data["rotor_swept_area_avg"]
    growth_rotor_swept_area_avg = rotor_swept_area_avg[-1] / rotor_swept_area_avg[0] * 100
    write_data_value(
        "growth_rotor_swept_area_avg",
        f"{growth_rotor_swept_area_avg.values:.0f}",
    )
    # TODO double check values here, especially if time series are the correct time range!


def specific_power():
    turbines = load_turbines()
    rotor_swept_area = turbines.t_rd ** 2 / 4 * np.pi
    specific_power = (
        (turbines.t_cap * KILO_TO_ONE / rotor_swept_area).groupby(turbines.p_year).mean()
    )
    write_data_value(
        "specific-power-start",
        f"{specific_power.sel(p_year=2010).values:.0f}",
    )
    write_data_value(
        "specific-power-end",
        f"{specific_power.sel(p_year=2019).values:.0f}",
    )


def capacity_growth():
    turbines = load_turbines()

    for year in (2010, 2019):
        installed_capacity_gw = (
            turbines.sel(turbines=turbines.p_year <= year).t_cap.sum().values * 1e-6
        )
        write_data_value(
            f"installed_capacity_gw_{year}",
            f"{installed_capacity_gw:.0f}",
        )


if __name__ == "__main__":
    calc_correlation_efficiency_vs_input_power_density()
    number_of_turbines()
    calculate_slopes()
    specific_power()
    capacity_growth()
    missing_commissioning_year()
