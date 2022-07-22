import logging
import xarray as xr
from dask.diagnostics import ProgressBar

from src.util import create_folder
from src.config import INPUT_DIR
from src.config import INTERIM_DIR
from src.load_data import load_turbines
from src.calculations import calc_bounding_box_usa

from src.logging_config import setup_logging


setup_logging()
turbines = load_turbines()


north, west, south, east = calc_bounding_box_usa(turbines)

# according to Katharina, heights and bands correlate in the following way:
#     50m  = band=1
#    100m = band=2
#    200m = band=3
#
# https://github.com/KatharinaGruber/windpower_GWA/blob/master/BRA/cut_GWA2.py#L17
height_to_band = {
    50: 1,
    100: 2,
    200: 3,
}

wind_speed_gwa = xr.open_rasterio(INPUT_DIR / "wind_speed_gwa2" / "gwa2_250_ws_DEFLATE.tif")

for height in (50, 100, 200):
    with ProgressBar():
        logging.info(f"Loading rasterio file for height={height}...")
        fname = create_folder(INTERIM_DIR / "wind_speed_gwa") / f"wind_speed_gwa{height}.nc"

        wind_speed_gwa_usa = wind_speed_gwa.isel(
            x=(west <= wind_speed_gwa.x) & (wind_speed_gwa.x <= east),
            y=(south <= wind_speed_gwa.y) & (wind_speed_gwa.y <= north),
        ).sel(band=height_to_band[height])

        wind_speed_gwa_usa = wind_speed_gwa_usa.drop_vars("band")

        # add NaN values
        wind_speed_gwa_usa = wind_speed_gwa_usa.where(wind_speed_gwa_usa != -999)

        logging.info("Saving to netcdf...")
        wind_speed_gwa_usa.to_netcdf(fname)
