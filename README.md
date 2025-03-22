# Ride-Hailing Simulation

This project is a discrete-event simulation of a ride-hailing system using polar coordinates. The simulation models the random arrival of riders and drivers in a unit disk and simulates the matching process between riders and drivers using a First-Come-First-Serve (FCFS) strategy.

## Features

- **Event-Driven Simulation**:  
  Implements a simulation framework where events (rider arrival, driver arrival, and rider departure) are scheduled and processed using a priority queue.
  
- **Uniform Spatial Distribution**:  
  Riders and drivers are assigned random positions uniformly distributed within a disk. A helper function (`generate_uniformly_points`) is provided to visualize the uniform distribution of points.
  
- **Dynamic Warmup Period Determination**:  
  The simulation includes a `determine_warmup_period` function that determines a warmup period to ensure that data is collected only when the system reaches a steady-state.
  
- **Basic Grid Search** :  
  A simple grid search function (`grid_search`) is applied to explore different combinations of rider arrival rates and driver-to-rider ratios.

## Project Structure

- `Event` and its subclasses (`RiderArrival`, `DriverArrival`, `RiderDeparture`):  
  Define the events in the simulation.
  
- `Rider` and `Driver` classes:  
  Represent the participants in the ride-hailing systema.
  
- `SimulationBaselineModel` class:  
  The core simulation model that runs the simulation, processes events, and collects performance metrics.
  
- Helper functions:  
  - `uniform_distributed_in_disk`: Generate a point uniformly in a disk using polar coordinates.
  - `distance_of_polar_coordinates`: Compute the polar coordinates distance between two points.
  - `calculate_time_period`: Convert time into discrete time periods.
  - `generate_uniformly_points`: Generate and visualize uniformly distributed points.
  - `determine_warmup_period`: Determine a warmup period by analyzing queue length stability.
  - `grid_search`: Explore different parameter combinations to identify optimal simulation settings.

- `main` function:  
  Coordinates the grid search and runs the simulation module.

## How to Run

1. **Prerequisites**:  
   - Python 3.x  
   - Required packages: `matplotlib`, `numpy`

2. **Run the Simulation**:  
   Execute the main script:
   ```bash
   python <Baseline_Model>.py
