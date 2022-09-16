import os
import pathlib

NUM_PROCESSES = 8

# used for downloading, calculation of time series etc
MONTHS = range(1, 13)
YEARS = range(2001, 2022)

# 43.23m/s is the maximum in the current wind data set
MAX_WIND_SPEED = 45

# increasing resolution may require more RAM, but not change the result a lot, probably way less
# than 1%, TODO not really tested yet, only for the wind_speeds a bit by varying MAX_WIND_SPEED
RESOLUTION_POWER_CURVE_MODEL = {
    "wind_speeds": 90,
    "specific_power": 30,
}

# turbines with outside of this range are discarded as unrealistic
SPECIFIC_POWER_RANGE = 100, 1000

# this corrects for downtime, curtailing and wakes, see notebook: estimate_loss_corretion_factor
LOSS_CORRECTION_FACTOR = 0.8890028225626951

# From Staffel and Green 2014: https://doi.org/10.1016/j.renene.2013.10.041
AGEING_PER_YEAR = 1.6e-2  # 16 percent per decade, 1.6% annually

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
# 80m is the median height of all turbines, but it's still a pretty arbitrary value
REFERENCE_HUB_HEIGHT_M = 80.0

CHUNK_SIZE_TURBINES = 4_000
CHUNK_SIZE_TIME = None  # 200

# US states which are assumed to have fewer data issues according to Gruber et al.
GOOD_STATES = (
    "IA,AZ,ID,CA,CO,IN,KS,HI,MD,ME,MI,NH,MN,MO,NM,MT,NY,ND,OK,UT,OR,PA,WA,WV,WY,TX".split(",")
)


# Turbines which are discarded, because they are offshore
OFFSHORE_TURBINES = [
    {
        "id": 3105266,
        "xlong": -75.491577,
        "ylat": 36.886841,
    },
    {
        "id": 3105267,
        "xlong": -75.491638,
        "ylat": 36.896313,
    },
]
