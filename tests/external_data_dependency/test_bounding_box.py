from src.load_data import load_turbines
from src.calculations import calc_bounding_box_usa


def test_calc_bounding_box_usa():
    turbines = load_turbines()
    north, west, south, east = calc_bounding_box_usa(turbines)

    # TODO this might be a very specific test, testing also turbines file...
    assert "{}".format(north) == "67.839905"
    assert (west, south, east) == (-172.713074, 16.970871, -64.610001)
