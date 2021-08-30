import numpy as np

from src.util import replace_nans_binmean


def estimate_missing(turbines, method="mean", params=("t_rd", "t_hh", "t_cap")):
    """Replace NaN values in `turbines` with estimates."""
    # If we don't have a year, we can't use the turbine at all...
    turbines = turbines.sel(turbines=~np.isnan(turbines.p_year))

    # 2020 turbines don't have any rotor diamters, let's throw them away for now...
    assert np.all(
        np.isnan(turbines.sel(turbines=turbines.p_year == 2020).t_rd)
    ), "Rotor diameter in 2020 detected"
    turbines = turbines.sel(turbines=turbines.p_year != 2020)

    if method == "linear_fit":
        is_complete = ~(np.isnan(turbines.t_hh) | np.isnan(turbines.t_rd))
        k, d = np.polyfit(turbines.t_rd[is_complete], turbines.t_hh[is_complete], deg=1)

        # rotor diameter is simply the mean rotor diameter per year
        turbines["t_rd"] = replace_nans_binmean(turbines["t_rd"], turbines.p_year)

        # replace hub heights by linear fit of rotor diameter, also for the estimated rotor
        # diameters
        turbines["t_hh"] = turbines.t_hh.where(~np.isnan(turbines.t_hh), k * turbines.t_rd + d)
    elif method in ("min", "max", "mean"):
        for param in params:
            turbines[param] = replace_nans_binmean(
                turbines[param], turbines.p_year, aggreagation=method
            )
    else:
        raise RuntimeError(f"unknown method '{method}'")

    return turbines
