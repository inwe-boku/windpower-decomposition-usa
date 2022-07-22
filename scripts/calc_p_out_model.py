import logging

import xarray as xr

from src.config import OUTPUT_DIR
from src.load_data import load_power_curve_model
from src.wind_power import calc_p_out
from src.wind_power import calc_power
from src.logging_config import setup_logging


def main():
    logging.info("Load input values...")
    # for height in (50, 100, 200):
    # height = 100
    bias_correction_100m = xr.open_dataarray(
        OUTPUT_DIR / "bias_correction" / "bias_correction_factors_gwa2_100m.nc", chunks=100
    )
    power_curves = load_power_curve_model()

    def compute_func(wind_speed, turbines):
        p_out = calc_p_out(turbines, power_curves, wind_speed, bias_correction_100m)
        return p_out

    calc_power(name="p_out_model", compute_func=compute_func)

    from src.loaded_files import specific_power

    for sp in (200, 250, 300, 400, float(specific_power.mean().compute())):

        def compute_func_refspecpower(wind_speed, turbines):
            p_out = calc_p_out(
                turbines,
                power_curves,
                wind_speed,
                bias_correction_100m,
                specific_power=sp,
            )
            return p_out

        calc_power(
            name="p_out_model",
            name_postfix=f"_refspecpower_{int(sp)}Wm2",
            compute_func=compute_func_refspecpower,
        )


if __name__ == "__main__":
    setup_logging()
    main()
