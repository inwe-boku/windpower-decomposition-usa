import warnings

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

from src.config import FIGSIZE
from src.constants import HOURS_PER_YEAR
from src.calculations import calc_capacity_per_year
from src.calculations import power_input
from src.load_data import load_turbines
from src.load_data import load_p_out_eia
from src.load_data import load_capacity_irena
from src.load_data import load_p_in
from src.util import write_data_value


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
    p_out_eia = load_p_out_eia()

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    per_year = turbines.t_cap.groupby(turbines.p_year)
    capacity_yearly_gw = per_year.sum(dim="turbines").cumsum() * 1e-6
    capacity_yearly_gw = capacity_yearly_gw.isel(
        p_year=capacity_yearly_gw.p_year >= p_out_eia.time.dt.year.min()
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
        p_out_eia.time.dt.year,
        p_out_eia * 1e-3 * HOURS_PER_YEAR,
        label="Yearly power generation (TWh/year)",
        marker="o",
        color="#0d8085",
    )
    plt.ylabel("Power generation (TWh/year)")
    ax2.legend(loc=1)

    return fig


def plot_growth_and_specific_power():
    from src.loaded_files import turbines
    from src.loaded_files import is_built
    from src.loaded_files import num_turbines_built
    from src.loaded_files import specific_power_per_year
    from src.loaded_files import rotor_swept_area_avg

    fig, axes = plt.subplots(4, figsize=FIGSIZE, sharex=True)

    # might fix spacing for titles
    fig.tight_layout(h_pad=2)

    ((turbines.t_hh * is_built).sum(dim="turbines") / num_turbines_built).plot.line(
        color="k", ax=axes[0]
    )
    axes[0].set_title("Average hub height")
    axes[0].set_ylabel("m")

    ((turbines.t_cap * is_built).sum(dim="turbines") / num_turbines_built).plot.line(
        color="k", ax=axes[1]
    )
    axes[1].set_title("Average capacity")
    axes[1].set_ylabel("KW")

    rotor_swept_area_avg.plot.line(color="k", ax=axes[2])
    axes[2].set_title("Average rotor swept area")
    axes[2].set_ylabel("m²")

    specific_power_per_year.plot.line(color="k", ax=axes[3])
    axes[3].set_title("Average specific power (W/m²)")
    axes[3].set_ylabel("W/m²")

    for ax in axes:
        ax.set_xlabel("")
        ax.grid()

    return fig


def plot_relative(data, unit="", ax=None, **kwargs):
    if ax is None:
        ax = plt  # ok that's a bit crazy
    ax.plot(
        data.time[:],
        100 * data / data[0],
        # "o-",
        **kwargs,
    )


def plot_absolute(data, unit="", ax=None, **kwargs):
    if ax is None:
        ax = plt  # ok that's a bit crazy
    ax.plot(
        data.time[:],
        data,
        # "o-",
        **kwargs,
    )

    if unit:
        ax.set_ylabel(unit)


def plot_timeseries_figure(figure_params, ax=None, fig=None):
    plot_line = plot_absolute if figure_params.absolute_plot else plot_relative

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    for line_params in figure_params.lines:
        plot_line(
            data=line_params.data,
            unit=figure_params.unit,
            label=line_params.label,
            color=line_params.color,
            linestyle=line_params.linestyle,
            linewidth=line_params.linewidth,
            ax=ax,
        )

    if not figure_params.absolute_plot:
        ax.axhline(100, color="k", linewidth=1)
        ax.set_ylabel(f"Relative to {int(line_params.data.time.dt.year[0])} (%)")

    ax.legend()
    ax.grid()

    return fig, ax


def _rotate_labels(ax, rotation):
    for label in ax.get_xmajorticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment("center")


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
                "ok",
                markersize=5,
                zorder=15,
                label=label_total,
                linestyle=linestyle,
            )

    if x is None:
        x = datasets[0].time.dt.year.values

    plt.xticks(indices + 0.5 * (len(datasets) - 1) * width, x)

    _rotate_labels(ax, rotation=90)

    # grid needs to be sent to background explicitly... (see also zorder above)
    ax.grid(zorder=0)

    if any(label is not None for label in labels):
        ax.legend(loc="lower right").set_zorder(50)

    return fig, ax


def plot_effect_trends_power(name, datasets, baseline, labels, colors, ax=None, fig=None):
    assert np.all(
        len(datasets[0]) == np.array([len(dataset) for dataset in datasets])
    ), "all datasets must be of same length"

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    previous = baseline
    for i, (label, dataset, color) in enumerate(zip(labels, datasets, colors)):
        dataset_relative = dataset - previous
        dataset_relative.plot.line("-", label=label, color=color, zorder=25)
        previous = dataset

    _rotate_labels(ax, rotation=0)

    ax.axhline(0, color="k", linewidth=1, zorder=5)

    ax.legend()
    ax.grid(zorder=0)
    name_tex = "P_\\mathrm{" + name[2:] + "}"
    ax.set_ylabel("Change in $\\frac{" + name_tex + "}{A}$ (W/m²)")
    ax.set_xlabel("")

    return fig, ax


def plot_irena_capacity_validation(turbines, turbines_with_nans, ax=None):
    if ax is None:
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

    # Plot gray area to indicate period of uncertainty:
    # xlim = ax.get_xlim()
    # plt.xlim(*xlim)
    # plt.axvspan(xlim[0] - 10, 2010, facecolor="k", alpha=0.07)

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


def plot_missing_uswtdb_data():
    from src.loaded_files import is_built
    from src.loaded_files import num_turbines_built

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    turbines = load_turbines(replace_nan_values="")

    is_metadata_missing_hh = np.isnan(turbines.t_hh)
    is_metadata_missing_rd = np.isnan(turbines.t_rd)
    is_metadata_missing_cap = np.isnan(turbines.t_cap)

    num_missing_hh_per_year = (is_metadata_missing_hh * is_built).sum(dim="turbines")
    num_missing_rd_per_year = (is_metadata_missing_rd * is_built).sum(dim="turbines")
    num_missing_cap_per_year = (is_metadata_missing_cap * is_built).sum(dim="turbines")

    # note: this assumes that a turbine with installation year x is already operating in year x
    (100 * num_missing_hh_per_year / num_turbines_built).plot.line(
        label="Hub height",
        color=TURBINE_COLORS[1],
        ax=ax,
    )
    (100 * num_missing_rd_per_year / num_turbines_built).plot(
        label="Rotor diameter",
        color=TURBINE_COLORS[3],
        ax=ax,
    )
    percent_missing_cap_per_year = 100 * num_missing_cap_per_year / num_turbines_built
    percent_missing_cap_per_year.plot(
        label="Capacity",
        color=TURBINE_COLORS[4],
        ax=ax,
    )

    for year in (2001, 2008):
        write_data_value(
            f"percent_missing_capacity_per_year{year}",
            f"{percent_missing_cap_per_year.sel(time=str(year)).values[0]:.0f}",
        )

    plt.legend()
    plt.ylabel("Turbines with missing metadata (%)")
    plt.xlabel("")
    plt.grid()

    return fig


def plot_scatter_efficiency_input_power_density():
    from src.loaded_files import p_out_model_raw
    from src.loaded_files import d_in
    from src.loaded_files import efficiency
    from src.loaded_files import specific_power
    from src.loaded_files import turbines

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    p_in_raw = load_p_in(avgwind=False, refheight=False, aggregate=False)

    rotor_swept_area_raw = turbines.t_rd**2 * np.pi / 4
    d_in_raw = (1e9 * p_in_raw / rotor_swept_area_raw).load()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=RuntimeWarning,
            message="invalid value encountered in true_divide",
        )
        efficiency_raw = 100 * (p_out_model_raw / p_in_raw).load()

    N = 2000
    np.random.seed(42)
    idcs = np.unique(np.random.randint(0, p_in_raw.sizes["turbines"], N))

    plt.grid()

    plt.scatter(
        d_in,
        efficiency,
        c=TURBINE_COLORS[4],
        label="All turbines aggregated",
        zorder=10,
    )

    scatter_turbines = plt.scatter(
        d_in_raw.isel(turbines=idcs),
        efficiency_raw.isel(turbines=idcs),
        c=xr.broadcast(specific_power.isel(turbines=idcs), p_in_raw.isel(turbines=idcs))[0],
        cmap="cividis",
        s=3,
        label=f"{N} random samples of single turbines",
    )

    plt.colorbar(label="Specific power (W/m²)")

    handles, labels = plt.gca().get_legend_handles_labels()
    handles[1] = scatter_turbines.legend_elements(prop="sizes")[0][0]

    plt.xlabel("Input power density (W/m²)")
    plt.ylabel("System efficiency (%)")
    plt.legend(handles, labels)

    return fig


def plot_irena_poweroutput_validation(p_out_eia, p_out_irena):
    # TODO this plot is obsolete, remove or just keep?
    fig, axes = plt.subplots(2, figsize=FIGSIZE, sharex=True)

    (1e-3 * HOURS_PER_YEAR * p_out_eia).plot.line(
        label="EIA",
        ax=axes[0],
        color=TURBINE_COLORS[3],
    )
    (1e-3 * HOURS_PER_YEAR * p_out_irena).plot.line(
        label="IRENA",
        ax=axes[0],
        color=TURBINE_COLORS[4],
    )
    axes[0].set_ylabel("Power output (TWh/Year)")
    axes[0].set_xlabel("")
    axes[0].grid()
    axes[0].legend()

    rel_difference = 100 * p_out_irena / p_out_eia - 100
    rel_difference.plot.line(label="Relative difference (IRENA - EIA)", color="k")

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


def plot_efficiency_ge1577_example(
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
):

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 16))  # , sharex='col', sharey='row')

    # set shared x and y axis
    for row in (0, 1):
        for col in (0, 1):
            axes[row, 0].sharey(axes[row, 1])
            axes[row, col].sharex(axes[row + 1, col])
            plt.setp(axes[row, col].get_xticklabels(), visible=False)

    for ax1 in axes:
        for ax2 in ax1:
            ax2.grid()

    axes[0, 0].plot(
        wind_speed_linspace, p_out * 1e-3, color="k", label="Power curve (GE-1.5-77sl model)"
    )
    axes[0, 0].set_ylabel("Power output (KW)")
    axes[0, 0].legend()

    axes[0, 0].set_xlim(0, 20)

    axes[1, 0].plot(wind_speed_linspace, c_p, "k", label="$C_p$ (GE-1.5-77sl model)")
    axes[1, 0].axhline(16 / 27, label="Betz' limit", color="k", linestyle=":")
    axes[1, 0].set_ylabel("Efficiency (%)")
    axes[1, 0].legend()

    for turbine_idx, color in zip(turbine_idcs, colors):
        wind_speed_location = wind_speed.sel(turbines=turbine_idx)
        axes[2, 0].hist(
            wind_speed_location,
            histtype="step",
            bins=50,
            color=color,
            density=True,
            alpha=1,
        )
        axes[2, 1].hist(
            power_input(wind_speed_location, rotor_swept_area) / rotor_swept_area,
            bins=50,
            histtype="step",
            color=color,
            density=True,
            alpha=1,
        )

    axes[2, 0].set_ylabel("Probability density")
    axes[2, 0].set_xlabel("Wind speed (m/s)")

    axes[0, 1].plot(p_in / rotor_swept_area, p_out * 1e-3, color="k")
    axes[0, 1].set_ylabel("Power output (KW)")

    _location_scatter_plot(
        axes[1, 1],
        p_in,
        rotor_swept_area,
        c_p,
        pin_monthly_aggregated,
        pout_monthly_aggregated,
        colors,
    )
    axes[1, 1].axhline(16 / 27, label="Betz' limit", color="k", linestyle=":")
    axes[1, 1].set_xlim(0, (p_in / rotor_swept_area)[-1])

    for col, label in zip((0, 1), ["Wind speeds", "Input power density"]):
        marker = Line2D([], [], color="k", label=f"{label} at selected turbine locations")
        axes[2, col].legend(handles=[marker])

    axes[2, 1].set_xlabel("Input power density (W/m²)")
    axes[2, 1].set_ylabel("Probability density")

    # no idea why the rect parameter necessary to avoid clipping of axis labels, but seems to look
    # good now by choosing arbitrary values
    # https://stackoverflow.com/a/6776578/859591
    plt.tight_layout(rect=(0.0, 0.5, 1.0, 1.0))

    return fig


def _location_scatter_plot(
    ax,
    p_in,
    rotor_swept_area,
    c_p,
    pin_monthly_aggregated,
    pout_monthly_aggregated,
    colors,
    zoom=False,
):
    ax.plot(
        p_in / rotor_swept_area,
        c_p,
        "-k",
        label="$C_p$ (GE-1.5-77sl model)",
    )

    ax.set_ylabel("Efficiency (%)")
    markersize = 3 if zoom else 1.2
    ax.set_prop_cycle(color=colors)
    ax.plot(
        pin_monthly_aggregated / rotor_swept_area,
        pout_monthly_aggregated / pin_monthly_aggregated,
        "o",
        markersize=markersize,
        alpha=1 if zoom else 0.5,
        markeredgewidth=0,
    )

    handles, labels = ax.get_legend_handles_labels()

    marker = Line2D(
        [],
        [],
        color="k",
        marker="o",
        markersize=markersize,
        label="Efficiency monthly aggregated",
        linewidth=0,
    )
    ax.legend(handles=handles + [marker])

    if zoom:
        ax.set_xlabel("Input power density (W/m²)")
        ax.set_xlim(20, 1000, auto=True)
        ax.set_ylim(0.2, 0.45, auto=True)


def plot_efficiency_ge1577_example_zoom(
    p_in, rotor_swept_area, c_p, pin_monthly_aggregated, pout_monthly_aggregated, colors
):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    _location_scatter_plot(
        ax,
        p_in,
        rotor_swept_area,
        c_p,
        pin_monthly_aggregated,
        pout_monthly_aggregated,
        colors,
        zoom=True,
    )

    return fig
