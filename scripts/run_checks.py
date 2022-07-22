import numpy as np
import xarray as xr

from src.logging_config import setup_logging
from src.config import OUTPUT_DIR
from src.load_data import load_turbines
from src.load_data import load_p_in


def check_turbines():
    turbines = load_turbines(replace_nan_values=False)
    num_turbines = turbines.sizes["turbines"]

    assert np.isnan(turbines.xlong).sum() == 0
    assert np.isnan(turbines.ylat).sum() == 0

    # possible asserts: dtype, range from to, min np.diff?

    assert (
        0.01 < np.isnan(turbines.p_year).sum() / num_turbines < 0.1
    ), "wrong number of NaNs in p_year's simulation data"


def check_pin():
    p_in = load_p_in()
    p_in_avgwind = load_p_in(avgwind=True)

    # TODO add test for p_in_avg80

    # TODO not sure what is a good threshold here
    assert np.abs((p_in - p_in_avgwind).mean()) < 0.5


def check_is_built():
    is_built = xr.open_dataarray(OUTPUT_DIR / "turbine-time-series" / "is_built.nc")

    assert is_built.dims == ("turbines", "time")
    assert np.all(0 <= is_built)
    assert np.all(is_built <= 1)

    is_built_diff = is_built.astype(float).diff(dim="time")

    # this should be less than a 1/24/60, but simulation data contains only one time stamp every
    # 28-31 days, so we are not that strict here...

    # ok this was wrong

    # time is monthly here! Is this an issue?
    assert np.all(0 <= is_built_diff)
    assert np.all(is_built_diff <= 31 / 364)

    # breakpoint()
    # assert np.all((is_built_diff.sum(dim="time") == 1) | (is_built_diff.sum(dim="time") == 0))


def check_rotor_swept_area():
    is_built = xr.open_dataarray(OUTPUT_DIR / "turbine-time-series" / "is_built.nc")
    num_turbines = is_built.sum(dim="turbines")

    rotor_swept_area = xr.open_dataarray(
        OUTPUT_DIR / "turbine-time-series" / "rotor_swept_area.nc"
    )
    min_rotor_diameter = 10
    max_rotor_diameter = 180
    avg_rotor_swept_area = rotor_swept_area / num_turbines
    assert np.all(min_rotor_diameter**2 / 4 * np.pi < avg_rotor_swept_area) & np.all(
        avg_rotor_swept_area < max_rotor_diameter**2 / 4 * np.pi
    ), "implausible average rotor diameter"


if __name__ == "__main__":
    setup_logging()
    check_turbines()
    check_pin()
    check_is_built()
    check_rotor_swept_area()
    # TODO check efficiency, Betz limit!
