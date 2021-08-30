import logging
import dask as da
from dask.diagnostics import ProgressBar

from src.util import create_folder
from src.config import OUTPUT_DIR
from src.config import YEARS
from src.config import REFERENCE_HUB_HEIGHT_M
from src.load_data import load_turbines
from src.load_data import load_wind_speed
from src.calculations import calc_power_in_wind

from src.logging_config import setup_logging


def main():
    turbines = load_turbines()

    output_folder = create_folder("power_in_wind", prefix=OUTPUT_DIR)

    params_calculations = [
        {
            "name": "p_in",
            "average_wind": False,
            "height": None,
        },
        {
            "name": "p_in_avg",
            "average_wind": True,
            "height": None,
        },
        {
            "name": "p_in_avg80",
            "average_wind": True,
            "height": REFERENCE_HUB_HEIGHT_M,
        },
    ]

    results = []
    fnames = []

    for params in params_calculations:
        with ProgressBar():
            name = params["name"]
            logging.info(f"Starting calculation for {name}...")

            wind_speed = load_wind_speed(YEARS, params["height"])

            logging.info("Calculating power in wind...")
            p_in, p_in_monthly = calc_power_in_wind(
                wind_speed,
                turbines,
                average_wind=params["average_wind"],
            )
            results.append(p_in)
            fnames.append(output_folder / f"{name}.nc")

            if p_in_monthly is not None:
                logging.info("Calculating (lazy) monthly time p_in...")
                results.append(p_in_monthly)
                fnames.append(output_folder / f"{name}_monthly.nc")

    logging.info("Starting compute()...")
    results_computed = []
    for result in results:
        with ProgressBar():
            results_computed.append(da.compute(result)[0])
    logging.info("Computation done!")

    for data, fname in zip(results_computed, fnames):
        logging.info(f"Saving result to NetCDF at {fname}...")
        data.to_netcdf(fname)


if __name__ == "__main__":
    setup_logging()
    # client = da.distributed.Client(memory_limit="45GB", n_workers=2, threads_per_worker=4)
    main()
