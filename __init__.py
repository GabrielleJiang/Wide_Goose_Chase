"""
A ride-hailing simulation package using polar coordinates.

This package simulates a ride-hailing system where:
- Riders and drivers arrive following Poisson processes
- Locations are represented in polar coordinates
- Matching follows First-Come-First-Served (FCFS) policy
"""

from .Baseline_Model import (
    SimulationBaselineModel,
    Event,
    RiderArrival,
    RiderDeparture,
    DriverArrival,
    Rider,
    Driver,
    uniform_distributed_in_disk,
    distance_of_polar_coordinates,
    calculate_time_period,
    generate_uniformly_points,
    determine_warmup_period,
    grid_search,
)

__version__ = "1.0.0"
__author__ = "Weixuan Jiang"

__all__ = [
    "SimulationBaselineModel",
    "Event",
    "RiderArrival",
    "RiderDeparture",
    "DriverArrival",
    "Rider",
    "Driver",
    "uniform_distributed_in_disk",
    "distance_of_polar_coordinates",
    "calculate_time_period",
    "generate_uniformly_points",
    "determine_warmup_period",
    "grid_search",
]
