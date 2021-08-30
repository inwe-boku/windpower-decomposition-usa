import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

from src.config import FIGSIZE
from src.constants import HOURS_PER_YEAR
from src.calculations import calc_capacity_per_year
from src.load_data import load_turbines
from src.load_data import load_generated_energy_gwh_yearly
from src.load_data import load_capacity_irena
from src.util import write_data_value
from src.util import filter2010
from src.util import calc_abs_slope

# this is actually 1 extra color, we have 4 models ATM
TURBINE_COLORS = (
    "#000000",
    "#f0c220",
    "#fbd7a8",
    "#0d8085",
    "#c72321",
)


def savefig(fname):
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def plot_growth_of_wind_power():
    turbines = load_turbines()
    generated_energy_gwh_yearly = load_generated_energy_gwh_yearly()

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    per_year = turbines.t_cap.groupby(turbines.p_year)
    capacity_yearly_gw = per_year.sum(dim="turbines").cumsum() * 1e-6
    capacity_yearly_gw = capacity_yearly_gw.isel(
        p_year=capacity_yearly_gw.p_year >= generated_energy_gwh_yearly.time.dt.year.min()
    )

    capacity_yearly_gw.plot(
        label="Total installed capacity (GW)",
        ax=ax,
        marker="o",
        color="#efc220",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend()
    plt.xlabel("Year")
    plt.ylabel("Capacity (GW)")
    plt.grid(True)

    ax2 = ax.twinx()
    ax2.plot(
        generated_energy_gwh_yearly.time.dt.year,
        generated_energy_gwh_yearly * 1e-3,
        label="Yearly power generation (TWh/year)",
        marker="o",
        color="#0d8085",
    )
    plt.ylabel("Power generation (TWh/year)")
    ax2.legend(loc=1)

    return fig


def plot_relative_change(data, label="", ax=None):
    # this is probably not a good idea... very weird stuff happens when calculating changes
    # in percent points...
    if ax is None:
        ax = plt  # ok that's a bit crazy
    ax.plot(
        data.time[1:],
        100 * np.diff(data / data[0]),
        "o-",
        label=label,
    )


def plot_relative(data, data_monthly=None, unit="", ax=None, **kwargs):
    if ax is None:
        ax = plt  # ok that's a bit crazy
    ax.plot(data.time[:], 100 * data / data[0], "o-", **kwargs)


def plot_absolute(data, data_monthly=None, unit="", ax=None, **kwargs):
    if ax is None:
        ax = plt  # ok that's a bit crazy
    ax.plot(data.time[:], data, "o-", **kwargs)
    if data_monthly is not None:
        data_monthly.plot.line("-", alpha=0.4, **kwargs)

    if unit:
        plt.ylabel(unit)


def plot_decomposition_p_out(
    num_turbines_built,
    rotor_swept_area,
    generated_energy_gwh_yearly,
    p_in,
    p_in_avg,
    p_in_avg80,
    generated_energy_gwh,
    p_in_monthly,
    rotor_swept_area_monthly,
    plot_only=None,
    absolute_plot=False,
    fig=None,
    ax=None,
):
    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    plot_it = plot_absolute if absolute_plot else plot_relative

    if plot_only is None or "rotor_swept_area" in plot_only:
        plot_it(
            rotor_swept_area / num_turbines_built,
            label="Average rotor swept area",
            unit="m²",
            ax=ax,
            color=TURBINE_COLORS[1],
        )

    if plot_only is None or "number" in plot_only:
        plot_it(
            num_turbines_built * 1e-3,
            label="Number of operating turbines",
            unit="in thousands",
            ax=ax,
            color=TURBINE_COLORS[2],
        )

    if plot_only is None or "efficiency" in plot_only:
        # FIXME "12 / HOURS_PER_YEAR" is a rough estimate, because hours per month vary
        plot_it(
            data=100 * generated_energy_gwh_yearly / p_in / HOURS_PER_YEAR,
            # TODO doesn't make sense that way, needs to be shifted by 6 monhts
            # data_monthly=(generated_energy_gwh / HOURS_PER_YEAR / p_in_monthly * 100 * 12),
            label="System efficiency",
            unit="%",
            ax=ax,
            color=TURBINE_COLORS[3],
        )

    if "outputpowerdensity" in plot_only:
        outputpowerdensity = 1e9 * generated_energy_gwh_yearly / HOURS_PER_YEAR / rotor_swept_area
        plot_it(
            outputpowerdensity,
            # TODO doesn't make sense that way, needs to be shifted by 6 monhts
            # data_monthly=(1e9 * p_in_monthly / rotor_swept_area_monthly),
            label="Output power density",
            unit="W/m²",
            ax=ax,
            color=TURBINE_COLORS[4],
        )

    if plot_only is None or "powerdensity" in plot_only:
        plot_it(
            1e9 * p_in / rotor_swept_area,
            # TODO doesn't make sense that way, needs to be shifted by 6 monhts
            # data_monthly=(1e9 * p_in_monthly / rotor_swept_area_monthly),
            label="Input power density",
            unit="W/m²",
            ax=ax,
            color=TURBINE_COLORS[4],
        )

    if plot_only is None or "powerdensity-avg" in plot_only:
        plot_it(
            1e9 * p_in_avg / rotor_swept_area,
            label="Input power density, wind averaged",
            unit="W/m²",
            ax=ax,
            color=TURBINE_COLORS[4],
            linestyle="--",
        )
    if plot_only is None or "powerdensity-avg-80" in plot_only:
        plot_it(
            1e9 * p_in_avg80 / rotor_swept_area,
            label="Input power density at 76m, wind averaged",
            unit="W/m²",
            ax=ax,
            color=TURBINE_COLORS[4],
            linestyle=":",
        )

    if not absolute_plot:
        plt.axhline(100, color="k", linewidth=1)
        plt.ylabel(f"Relative to {int(rotor_swept_area.time.dt.year[0])} (%)")

    plt.legend()

    plt.grid()

    return fig, ax


def plot_waterfall(
    *datasets,
    x=None,
    labels=None,
    width=0.18,
    gap=0.07,
    bottom=0,
    colors=None,
    total=True,
    labels_total=None,
    linestyles=None,
):
    """Plot components of a time series. Each ``dataset`` in ``datasets`` is a time series of one
    component (positive or negative). It is assumed that the sum of all components is meaningful
    in some way (for each time stamp).

    The term waterfall plot is typically used for something slightly different, this function
    should probably be renamed in future.

    Parameters
    ----------
    datasets : iterable of xr.DataArray (dims: time)
    x : arraylike or None
        used as labels for xticks, if None years of the first dataset will be used
    labels : iterable of strings
        labels for legend
    ...

    """
    assert np.all(
        len(datasets[0]) == np.array([len(dataset) for dataset in datasets])
    ), "all datasets must be of same length"

    indices = np.arange(len(datasets[0]))

    previous = bottom * (1 + 0 * datasets[0])  # xarray does not have a np.zeros_like()... :(

    if labels is None:
        labels = [None] * len(datasets)

    if colors is None:
        colors = [None] * len(datasets)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    gap_shift = gap * (len(datasets) - 1) / 2.0

    def bar_centers(i):
        return indices + i * (width + gap) - gap_shift

    for i, (label, dataset, color) in enumerate(zip(labels, datasets, colors)):
        ax.bar(
            bar_centers(i),
            dataset.values - previous,
            width,
            previous.values,
            label=label,
            zorder=10,  # for some reason this needs to be >= 2, set it to 10 to be sure :)
            color=color,
        )

        # horizontal lines to connect bars
        if i < len(datasets) - 1:
            ax.hlines(
                dataset.values,
                bar_centers(i) - 0.5 * width,
                bar_centers(i + 1) + 0.5 * width,
                color="grey",
                linewidth=1.0,
                zorder=15,
            )

        previous = dataset

    for i, (label_total, dataset, linestyle) in enumerate(zip(labels_total, datasets, linestyles)):
        if total:
            plt.plot(
                bar_centers(i),
                dataset.values,
                "o-k",
                markersize=5,
                zorder=15,
                label=label_total,
                linestyle=linestyle,
            )

    if x is None:
        x = datasets[0].time.dt.year.values

    plt.xticks(indices + 0.5 * (len(datasets) - 1) * width, x)

    # grid needs to be sent to background explicitly... (see also zorder above)
    ax.grid(zorder=0)

    if any(label is not None for label in labels):
        ax.legend(loc="lower right").set_zorder(50)

    return fig, ax


def plot_effect_trends_pin(datasets, baseline, labels, colors):
    assert np.all(
        len(datasets[0]) == np.array([len(dataset) for dataset in datasets])
    ), "all datasets must be of same length"

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    # this was used only for the one-slide-presentation for the EGU
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()

    previous = baseline
    for i, (label, dataset, color) in enumerate(zip(labels, datasets, colors)):
        dataset_relative = dataset - previous
        dataset_relative.plot.line("o-", label=label, color=color, zorder=25)
        previous = dataset

        # this is a bit ugly :-/
        if dataset.isel(time=0).time.dt.year == 2010:
            label_no_space = label.replace(" ", "-").lower()
            write_data_value(
                f"inputpowerdensity_{label_no_space}",
                f"{(dataset_relative[-1] - dataset_relative[0]).values:.1f}",
            )
            if label == "Wind change due to new locations":
                write_data_value(
                    f"inputpowerdensity_{label_no_space}_until2013_abs",
                    f"{abs(float(dataset_relative.sel(time='2013') - dataset_relative[0])):.1f}",
                )
                write_data_value(
                    f"inputpowerdensity_{label_no_space}_since2013",
                    f"{float(dataset_relative[-1] - dataset_relative.sel(time='2013')):.1f}",
                )

            if label == "Annual variations":
                for extremum in ("min", "max"):
                    write_data_value(
                        f"inputpowerdensity_{label_no_space}_{extremum}",
                        f"{getattr(dataset_relative, extremum)().values:.1f}",
                    )

    for label in ax.get_xmajorticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")

    plt.axhline(0, color="k", linewidth=1, zorder=5)

    plt.legend()
    ax.grid(zorder=0)
    plt.ylabel("Change in input power density (W/m²)")
    plt.xlabel("")

    return fig, ax


def plot_irena_capacity_validation(turbines, turbines_with_nans):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    capacity_irena = load_capacity_irena()
    capacity_uswtdb = calc_capacity_per_year(turbines)
    capacity_uswtdb_no_capnans = calc_capacity_per_year(turbines_with_nans)

    rel_errors = []

    def compare_to_irena(capacity_uswtdb, label, **kwargs):
        rel_error = 100 * (capacity_uswtdb - capacity_irena) / capacity_irena
        rel_error.plot.line(label=label, ax=ax, **kwargs)
        rel_errors.append(rel_error)

    capacity_uswtdb_no_decom = calc_capacity_per_year(
        turbines.sel(turbines=~turbines.is_decomissioned)
    )

    # more scenarios
    # COLORS = ("#0f4241", "#273738", "#136663", "#246b71", "#6a9395", "#84bcbf", "#9bdade")
    # LIFETIMES = (15, 18, 19, 20, 25, 30, 35)

    COLORS = ("#273738", "#246b71", "#6a9395", "#84bcbf", "#9bdade")
    LIFETIMES = (15, 20, 25, 30, 35)

    for lifetime, color in zip(LIFETIMES, COLORS):
        capacity_uswtdb_no_old = capacity_uswtdb - capacity_uswtdb.shift(p_year=lifetime).fillna(
            0.0
        )
        compare_to_irena(capacity_uswtdb_no_old, f"{lifetime} years lifetime", color=color)

    for lifetime, color in zip(LIFETIMES, COLORS):
        # the same thing again without the t_cap NaN replacement
        capacity_uswtdb_no_old = capacity_uswtdb_no_capnans - capacity_uswtdb_no_capnans.shift(
            p_year=lifetime
        ).fillna(0.0)
        compare_to_irena(
            capacity_uswtdb_no_old,
            "",  # f"lifetime {lifetime} (without capacity data imputation)",
            linestyle="--",
            color=color,
        )

    compare_to_irena(
        capacity_uswtdb_no_decom,
        "exclude decommissioned turbines",
        linewidth=4,
        color="#ffde65",
    )

    compare_to_irena(
        capacity_uswtdb, "include decommissioned turbines", linewidth=4, color="#c42528"
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid()

    xlim = ax.get_xlim()
    plt.xlim(*xlim)
    plt.axvspan(xlim[0] - 10, 2010, facecolor="k", alpha=0.07)

    handles, labels = plt.gca().get_legend_handles_labels()
    line = Line2D([0], [0], label="without data imputation", linestyle="--", color="k")
    handles.insert(-2, line)
    plt.legend(handles=handles)

    plt.tight_layout()

    plt.axhline(0.0, color="k")

    plt.xlabel("")
    plt.ylabel("Relative difference (%)")

    rel_errors = xr.concat(rel_errors, dim="scenarios")
    max_abs_error = (
        np.abs(rel_errors.isel(p_year=(rel_errors.p_year >= 2010).values)).max().compute()
    )

    # note: using ceil, because text says "less than"
    write_data_value(
        "irena_uswtdb_validation_max_abs_error",
        f"{float(np.ceil(max_abs_error)):.0f}",
    )

    return fig


def plot_missing_uswtdb_data():
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    turbines = load_turbines(replace_nan_values="")

    is_metadata_missing_hh = np.isnan(turbines.t_hh)
    is_metadata_missing_rd = np.isnan(turbines.t_rd)
    is_metadata_missing_cap = np.isnan(turbines.t_cap)

    num_turbines_per_year = turbines.p_year.groupby(turbines.p_year).count()

    num_missing_hh_per_year = is_metadata_missing_hh.groupby(turbines.p_year).sum()
    num_missing_rd_per_year = is_metadata_missing_rd.groupby(turbines.p_year).sum()
    num_missing_cap_per_year = is_metadata_missing_cap.groupby(turbines.p_year).sum()

    # note: this assumes that a turbine with installation year x is already operating in year x
    (100 * num_missing_hh_per_year.cumsum() / num_turbines_per_year.cumsum()).plot.line(
        label="Hub height",
        color=TURBINE_COLORS[1],
        ax=ax,
    )
    (100 * num_missing_rd_per_year.cumsum() / num_turbines_per_year.cumsum()).plot(
        label="Rotor diameter",
        color=TURBINE_COLORS[3],
        ax=ax,
    )
    percent_missing_cap_per_year = (
        100 * num_missing_cap_per_year.cumsum() / num_turbines_per_year.cumsum()
    )
    percent_missing_cap_per_year.plot(
        label="Capacity",
        color=TURBINE_COLORS[4],
        ax=ax,
    )
    for year in (2000, 2010):
        write_data_value(
            f"percent_missing_capacity_per_year{year}",
            f"{percent_missing_cap_per_year.sel(p_year=year).values:.0f}",
        )

    plt.legend()
    plt.ylabel("Turbines with missing metadata (%)")
    plt.xlabel("")
    plt.grid()

    return fig


def plot_scatter_efficiency_input_power_density(
    p_in_monthly, rotor_swept_area_monthly, efficiency_monthly
):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    plt.scatter(
        1e9 * p_in_monthly / rotor_swept_area_monthly,
        efficiency_monthly,
        c=p_in_monthly.time.dt.year,
        cmap="cividis",
    )
    plt.colorbar()
    plt.xlabel("Power density (W/m²)")
    plt.ylabel("Efficiency (%)")
    plt.grid()

    return fig


def plot_system_effiency(
    efficiency,
    efficiency_without_pin,
    efficiency_monthly=None,
    efficiency_without_pin_monthly=None,
):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    (100 * efficiency).plot.line("o-", color=TURBINE_COLORS[3], label="System efficiency")
    (100 * efficiency_without_pin).plot.line(
        "--o", color=TURBINE_COLORS[3], label="Scenario with constant input power density"
    )

    if efficiency_monthly is not None:
        (100 * efficiency_monthly).plot.line(
            "o-", color=TURBINE_COLORS[4], label="Monthly system efficiency, aggregated yearly"
        )
        (100 * efficiency_without_pin_monthly).plot.line(
            "--o",
            color=TURBINE_COLORS[4],
            label=("Scenario using monthly time series, aggregated yearly"),
        )

    for label in ax.get_xmajorticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")

    plt.grid()

    plt.legend()
    plt.xlabel("")
    plt.ylabel("System efficiency (%)")

    # this should be better placed in scripts/calc_data_values.py but would cause a lot of code
    # duplication without large re-organization, so let's keep it here
    if efficiency_monthly is None:
        write_data_value(
            "efficiency_without_pin_yearly_start",
            f"{100 * efficiency_without_pin[0].values:.1f}",
        )
        write_data_value(
            "efficiency_without_pin_yearly_end",
            f"{100 * efficiency_without_pin[-1].values:.1f}",
        )
        write_data_value(
            "efficiency_yearly_start",
            f"{100 * efficiency.values[0]:.1f}",
        )
        write_data_value(
            "efficiency_yearly_end",
            f"{100 * efficiency.values[-1]:.1f}",
        )

        write_data_value(
            "efficiency_without_pin_yearly_slope",
            f"{100 * calc_abs_slope(efficiency_without_pin):.2f}",
        )
        write_data_value(
            "efficiency_yearly_slope",
            f"{100 * calc_abs_slope(efficiency):.2f}",
        )

    return fig


def plot_capacity_factors(turbines, generated_energy_gwh_yearly, is_built_yearly):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    total_capacity_kw = (is_built_yearly * turbines.t_cap).sum(dim="turbines")

    capacity_factors = (
        100
        * 1e6
        * filter2010(generated_energy_gwh_yearly / HOURS_PER_YEAR)
        / filter2010(total_capacity_kw)
    )

    capacity_factors.plot.line("o-", color=TURBINE_COLORS[4])

    plt.xlabel("")
    plt.ylabel("Capacity factor (%)")
    plt.grid()

    return fig


def plot_irena_poweroutput_validation(p_out_eia, p_out_irena):
    fig, axes = plt.subplots(2, figsize=FIGSIZE, sharex=True)

    (1e-3 * p_out_eia).plot.line(label="EIA", ax=axes[0], color=TURBINE_COLORS[3], marker="o")
    (1e-3 * p_out_irena).plot.line(label="IRENA", ax=axes[0], color=TURBINE_COLORS[4], marker="o")
    axes[0].set_ylabel("Power output (TWh/Year)")
    axes[0].set_xlabel("")
    axes[0].grid()
    axes[0].legend()

    rel_difference = 100 * p_out_irena / p_out_eia - 100
    rel_difference.plot.line(label="Relative difference (IRENA - EIA)", color="k", marker="o")

    plt.ylabel("Relative difference (%)")
    plt.xlabel("")
    axes[1].grid()
    axes[1].legend()
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    write_data_value(
        "irena_poweroutput_max_deviation",
        f"{float(rel_difference.max()):.1f}",
    )

    return fig
