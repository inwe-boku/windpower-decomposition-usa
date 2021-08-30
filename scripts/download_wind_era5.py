import os
import logging

from multiprocessing import Pool

import cdsapi

from src.config import YEARS
from src.config import MONTHS
from src.config import INPUT_DIR

from src.util import create_folder
from src.logging_config import setup_logging
from src.load_data import load_turbines
from src.calculations import calc_bounding_box_usa


# How many downloads to start in parallel (one process for each worker)
NUM_WORKERS = 8


def main():
    # API documentation for downloading a subset:
    # https://confluence.ecmwf.int/display/CKB/Global+data%3A+Download+data+from+ECMWF+for+a+particular+area+and+resolution
    # https://retostauffer.org/code/Download-ERA5/

    download_dir = create_folder("wind_velocity_usa_era5", prefix=INPUT_DIR)

    setup_logging()

    turbines = load_turbines()
    north, west, south, east = calc_bounding_box_usa(turbines)

    # Format for downloading ERA5: North/West/South/East
    bounding_box = "{}/{}/{}/{}".format(north, west, south, east)

    logging.info(
        "Downloading bounding_box=%s for years=%s and months=%s",
        bounding_box,
        YEARS,
        MONTHS,
    )

    with Pool(processes=NUM_WORKERS) as pool:
        for year in YEARS:
            for month in MONTHS:
                pool.apply_async(
                    download_one_month,
                    (
                        bounding_box,
                        download_dir,
                        month,
                        year,
                    ),
                )

        pool.close()
        pool.join()


def download_one_month(bounding_box, download_dir, month, year):
    # TODO newer versions of csapi support asynchronous download, could be used instead of
    # multiprocessing, also retries are built in now

    # the progress bar looks broken when running multiple processes in parallel
    c = cdsapi.Client(progress=False)

    filename = download_dir / f"wind_velocity_usa_{year}-{month:02d}.nc"

    if filename.exists():
        logging.info(f"Skipping {filename}, already exists!")
        return

    logging.info(f"Starting download of {filename}...")
    for i in range(5):
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": [
                        "100m_u_component_of_wind",
                        "100m_v_component_of_wind",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                    ],
                    "year": f"{year}",
                    "month": [f"{month:02d}"],
                    "area": bounding_box,
                    "day": [f"{day:02d}" for day in range(1, 32)],
                    "time": [f"{hour:02d}:00" for hour in range(24)],
                },
                f"{filename}.part",
            )
        except Exception as e:
            logging.warning(
                "Download failed for %s/%s: %s",
                year,
                month,
                e,
            )
        else:
            logging.info(f"Download of {filename} successful!")
            os.rename(f"{filename}.part", filename)
            break
    else:
        logging.warning("Download failed permanently!")


if __name__ == "__main__":
    main()
