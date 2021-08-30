METER_TO_KM = 1e-3
ONE_TO_KILO = 1e3

KM_TO_METER = 1e3
KILO_TO_ONE = 1e3

# Average earth radius, see https://en.wikipedia.org/wiki/Earth_radius
EARTH_RADIUS_KM = 6371.0088

# in reality air density varies between 1.14 and 1.42 in kg/m^3
AIR_DENSITY_RHO = 1.225

# of course this introduces a small mistake due to leap years, but in average it's quite ok
# Warning: in most cases it might be better to use mean() instead of sum()/HOURS_PER_YEAR
HOURS_PER_YEAR = 8765.812536
