import os
import pathlib

NUM_PROCESSES = 8

# used for downloading, calculation of time series etc
MONTHS = range(1, 13)
YEARS = range(2009, 2020)

# This could be done automatically, note that last year is excluded because it's incomplete:
# YEARS = range(
#     int(generated_energy_gwh.time.min().dt.year),
#     int(generated_energy_gwh.time.max().dt.year),
# )

REPO_ROOT_DIR = pathlib.Path(__file__).parent.parent

simulation = "-simulation" if "SIMULATION" in os.environ and os.environ["SIMULATION"] else ""

DATA_DIR = REPO_ROOT_DIR / f"data{simulation}"

LOG_FILE = DATA_DIR / "logfile.log"

INPUT_DIR = DATA_DIR / "input"

INTERIM_DIR = DATA_DIR / "interim"

OUTPUT_DIR = DATA_DIR / "output"

FIGURES_DIR = DATA_DIR / "figures"

FIGSIZE = (12, 7.5)

# this hub height is used as fixed reference hub height to simulate no hub height change
REFERENCE_HUB_HEIGHT_M = 76.0

CHUNK_SIZE_TURBINES = 1_000
CHUNK_SIZE_TIME = 1_000
