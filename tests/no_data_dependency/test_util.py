import pytest
import numpy as np

from src.util import turbine_locations
from src.util import edges_to_center


def test_turbine_locations(turbines):
    locations = turbine_locations(turbines)
    assert locations.shape == (turbines.sizes["turbines"], 2)


def test_edges_to_center():
    edges = np.array([0.0, 2.0, 4.0, 6.0])
    centers = edges_to_center(edges)
    assert np.all(centers == [1.0, 3.0, 5.0])


def test_edges_to_center_wrong():
    edges = np.array([0.0, 2.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="works only for equidist"):
        edges_to_center(edges)


def test_edges_to_center_short():
    edges = np.array([0.0])
    with pytest.raises(ValueError, match="must be of length"):
        edges_to_center(edges)
