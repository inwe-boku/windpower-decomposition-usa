import numpy as np

from src.config import YEARS
from src.util import write_data_value
from src.util import nanratio
from src.load_data import load_turbines
from src.logging_config import setup_logging


def start_end_years():
    start_year = list(YEARS)[0]
    end_year = list(YEARS)[-1]

    write_data_value(
        "start-year",
        f"{start_year:.0f}",
    )

    write_data_value(
        "end-year",
        f"{end_year:.0f}",
    )


def calc_correlation_efficiency_vs_input_power_density():
    from src.loaded_files import d_in
    from src.loaded_files import efficiency

    correlation = np.corrcoef(d_in, efficiency)[0, 1]
    write_data_value(
        "correlation-efficiency-vs-input-power-density",
        f"{correlation:.3f}",
    )


def number_of_turbines():
    from src.loaded_files import turbines

    start_year = list(YEARS)[0]
    end_year = list(YEARS)[-1]
    num_turbines_start = (turbines.p_year <= start_year).sum()
    num_turbines_end = (turbines.p_year <= end_year).sum()

    growth_num_turbines_built = num_turbines_end / num_turbines_start * 100

    write_data_value(
        "growth_num_turbines_built",
        f"{growth_num_turbines_built.values:.0f}",
    )

    # we cannot use is_built here, because
    write_data_value(
        "number-of-turbines-start",
        f"{num_turbines_start.values:,d}",
    )
    write_data_value(
        "number-of-turbines-end",
        f"{num_turbines_end.values:,d}",
    )


def rotor_swept_area():
    from src.loaded_files import rotor_swept_area_avg

    growth_rotor_swept_area_avg = rotor_swept_area_avg[-1] / rotor_swept_area_avg[0] * 100
    write_data_value(
        "growth_rotor_swept_area_avg",
        f"{growth_rotor_swept_area_avg.values:.0f}",
    )

    # TODO double check values here, especially if time series are the correct time range!
    # is the last year really here really the right choice? Or is there an incomplete year at the
    # end?
    write_data_value(
        "rotor_swept_area_avg-start",
        f"{int(rotor_swept_area_avg[0].values.round()):,d}",
    )
    write_data_value(
        "rotor_swept_area_avg-end",
        f"{int(rotor_swept_area_avg[-1].values.round()):,d}",
    )


def missing_commissioning_year():
    from src.loaded_files import turbines

    start_year = list(YEARS)[0]
    end_year = list(YEARS)[-1]

    turbines_with_nans = load_turbines(replace_nan_values="")
    write_data_value(
        "percentage_missing_commissioning_year",
        f"{nanratio(turbines_with_nans.p_year).values * 100:.1f}",
    )

    missing_start = (
        np.isnan(turbines_with_nans.p_year).sum() / (turbines_with_nans.p_year <= start_year).sum()
    ).values
    write_data_value(
        "percentage_missing_commissioning_year_start",
        f"{missing_start * 100:.1f}",
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
        (turbines.sel(turbines=~turbines.is_decomissioned).p_year < (end_year - lifetime))
        .sum()
        .values
    )
    write_data_value(
        "num_further_old_turbines",
        f"{num_further_old_turbines:,d}",
    )

    write_data_value(
        "missing_ratio_rd_hh",
        f"{100 * nanratio(turbines_with_nans.t_hh + turbines_with_nans.t_rd).values:.1f}",
    )


def input_power_density():
    from src.loaded_files import d_in
    from src.loaded_files import d_in_avgwind
    from src.loaded_files import d_in_avgwind_refheight

    write_data_value(
        "d_in_avgwind_start",
        f"{d_in_avgwind[0].values:.1f}",
    )
    write_data_value(
        "d_in_avgwind_end",
        f"{d_in_avgwind[-1].values:.1f}",
    )

    # effect of hub height change
    effect_huhheights = d_in_avgwind - d_in_avgwind_refheight
    write_data_value(
        "d_in_effect-of-hub-height-change",
        f"{(effect_huhheights[-1] - effect_huhheights[0]).values:.1f}",
    )

    # change due to new locations
    write_data_value(
        "d_in_abs-change-new_locations",
        f"{abs(d_in_avgwind_refheight[-1] - d_in_avgwind_refheight[0]).values:.1f}",
    )

    # annual variations
    write_data_value(
        "d_in_variations_max",
        f"{(d_in - d_in_avgwind).max().values:.1f}",
    )
    write_data_value(
        "d_in_variations_min",
        f"{(d_in - d_in_avgwind).min().values:.1f}",
    )


def output_power_density():
    from src.loaded_files import d_out
    from src.loaded_files import d_out_avgwind

    write_data_value(
        "d_out_avgwind_start",
        f"{d_out_avgwind[0].values:.1f}",
    )
    write_data_value(
        "d_out_avgwind_end",
        f"{d_out_avgwind[-1].values:.1f}",
    )
    write_data_value(
        "d_out_avgwind_max",
        f"{d_out_avgwind.max().values:.1f}",
    )
    write_data_value(
        "d_out_avgwind_idxmax",
        f"{d_out_avgwind.idxmax().dt.year.values:.0f}",
    )
    write_data_value(
        "d_out_variations_max",
        f"{(d_out - d_out_avgwind).max().values:.1f}",
    )
    write_data_value(
        "d_out_variations_min",
        f"{(d_out - d_out_avgwind).min().values:.1f}",
    )
    write_data_value(
        "d_out_variations_std",
        f"{(d_out - d_out_avgwind).std().values:.1f}",
    )
    # not sure if this is really relevant... should be zero anyway (and it is)
    write_data_value(
        "d_out_variations_mean",
        f"{(d_out - d_out_avgwind).mean().values:.1f}",
    )

    write_data_value(
        "d_out_less_percentages_end",
        f"{(100 * (1 - d_out_avgwind[-1] / d_out_avgwind.max())).values:.1f}",
    )
    # TODO check if all values above are used correctly

    # d_out_avgwind_yearly_increase
    # d_out_avgwind_yearly_decrease
    # d_out_avgwind_decline_percent


def efficiency():
    from src.loaded_files import efficiency
    from src.loaded_files import efficiency_avgwind

    write_data_value(
        "efficiency_start",
        f"{efficiency.values[0]:.1f}",
    )
    write_data_value(
        "efficiency_end",
        f"{efficiency.values[-1]:.1f}",
    )
    decline_per_year = (efficiency_avgwind[-1] - efficiency_avgwind[0]) / len(efficiency_avgwind)
    write_data_value(
        "efficiency_avgwind-decline-per-year",
        f"{abs(decline_per_year).values:.2f}",
    )


def specific_power():
    from src.loaded_files import specifc_power_per_year

    write_data_value(
        "specific-power-start",
        f"{specifc_power_per_year.isel(time=0).values:.0f}",
    )
    write_data_value(
        "specific-power-end",
        f"{specifc_power_per_year.isel(time=-1).values:.0f}",
    )


if __name__ == "__main__":
    setup_logging()

    number_of_turbines()
    rotor_swept_area()
    input_power_density()
    output_power_density()
    efficiency()
    specific_power()
    missing_commissioning_year()
    calc_correlation_efficiency_vs_input_power_density()
