import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import random
import heapq
import uuid
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from Base_Line_Model.Baseline_Model import (
    SimulationBaselineModel,
    Rider,
    Driver,
    RiderArrival,
    DriverArrival,
    RiderDeparture,
    distance_of_polar_coordinates,
    uniform_distributed_in_disk,
    calculate_time_period
)

class SimulationTemporalPooling(SimulationBaselineModel):
    """
    A simulation model that extends the baseline model with temporal pooling.
    This model only allows drivers to match with riders when the queue length
    reaches a certain threshold.
    
    Attributes:
        m_threshold (int): Minimum number of riders required in queue before matching.
        lost_drivers (int): Number of drivers who left without matching.
    """
    def __init__(self, lambda_rate, mu_rate, sim_duration=5000, warmup_period=500, 
                 period_length=100, radius=1, m_threshold=1):
        """
        Initialize the temporal pooling model.
        
        Parameters:
            lambda_rate (float): Arrival rate of riders.
            mu_rate (float): Arrival rate of drivers.
            sim_duration (float): Total duration of the simulation.
            warmup_period (float): Warm-up period.
            period_length (float): Time length for each period.
            radius (float): Radius of the disk.
            m_threshold (int): Minimum number of riders required in queue before matching.
        """
        super().__init__(lambda_rate, mu_rate, sim_duration, warmup_period, period_length, radius)
        self.m_threshold = m_threshold
        self.lost_drivers = 0
        self.queue_times = []
        self.pickup_times = []
        self.travel_times = []

    def _when_driver_arrival(self, event):
        """
        Handle driver arrival event with temporal pooling.
        Only matches drivers with riders when queue length reaches threshold.
        
        Parameters:
            event (DriverArrival): The driver arrival event.
        """
        if self.is_warmed_up:
            self.total_drivers += 1
            self.driver_arrival_times.append(event.time)
            
        driver = Driver(event.driver_id, event.time, event.origin, event.period)
        
        if len(self.rider_queue) < self.m_threshold:
            if self.is_warmed_up:
                self.lost_drivers += 1
            return
            
        min_distance = float('inf')
        closest_rider = None
        closest_index = -1
        
        rider_list = list(self.rider_queue)
        
        for i, rider in enumerate(rider_list):
            distance = distance_of_polar_coordinates(driver.origin, rider.origin)
            if distance < min_distance:
                min_distance = distance
                closest_rider = rider
                closest_index = i
                
        if closest_rider is not None:
            self.rider_queue.remove(closest_rider)
            
            closest_rider.matched_time = event.time
            pickup_distance = min_distance
            closest_rider.pickup_distance = pickup_distance
            trip_distance = distance_of_polar_coordinates(closest_rider.origin, closest_rider.destination)
            closest_rider.trip_distance = trip_distance
            
            pickup_time = event.time + pickup_distance
            departure_time = pickup_time + trip_distance
            
            closest_rider.pickup_time = pickup_time
            closest_rider.departure_time = departure_time
            
            closest_rider.queue_time = closest_rider.matched_time - closest_rider.arrival_time
            closest_rider.pickup_wait_time = pickup_distance
            closest_rider.travel_time = trip_distance
            closest_rider.total_time = closest_rider.queue_time + closest_rider.pickup_wait_time + closest_rider.travel_time
            
            departure_event = RiderDeparture(departure_time, closest_rider)
            heapq.heappush(self.event_queue, departure_event)
            
            if self.is_warmed_up:
                self.matched_drivers += 1

    def _process_rider_departure(self, event):
        """
        Process rider departure event and track time components.
        
        Parameters:
            event (RiderDeparture): The rider departure event.
        """

        super()._process_rider_departure(event)
        
        if self.is_warmed_up:
            rider = event.rider
            self.queue_times.append(rider.queue_time)
            self.pickup_times.append(rider.pickup_wait_time)
            self.travel_times.append(rider.travel_time)
            
    def get_time_components(self):
        """
        Get the average time components for completed riders.
        
        Returns:
            dict: Dictionary with average queue time, pickup time, and travel time.
        """
        result = {}
        
        if self.queue_times:
            result["avg_queue_time"] = np.mean(self.queue_times)
        else:
            result["avg_queue_time"] = 0
            
        if self.pickup_times:
            result["avg_pickup_time"] = np.mean(self.pickup_times)
        else:
            result["avg_pickup_time"] = 0
            
        if self.travel_times:
            result["avg_travel_time"] = np.mean(self.travel_times)
        else:
            result["avg_travel_time"] = 0
            
        return result

    def calculate_metrics(self):
        """
        Calculate and return various metrics for the simulation.
        
        Returns:
            dict: Dictionary containing various metrics including:
                - total_riders: Total number of riders
                - total_drivers: Total number of drivers
                - matched_drivers: Number of successfully matched drivers
                - lost_drivers: Number of drivers who left without matching
                - match_rate: Ratio of matched drivers to total drivers
                - loss_rate: Ratio of lost drivers to total drivers
                - avg_rider_time: Average time riders spend in the system
                - variance_rider_time: Variance in rider times
        """
        metrics = {
            "total_riders": self.total_riders,
            "total_drivers": self.total_drivers,
            "matched_drivers": self.matched_drivers,
            "lost_drivers": self.lost_drivers
        }
        
        if self.total_drivers > 0:
            metrics["match_rate"] = self.matched_drivers / self.total_drivers
            metrics["loss_rate"] = self.lost_drivers / self.total_drivers
        else:
            metrics["match_rate"] = 0
            metrics["loss_rate"] = 0
            
        if self.rider_times:
            metrics["avg_rider_time"] = np.mean(self.rider_times)
            metrics["variance_rider_time"] = np.var(self.rider_times)
        else:
            metrics["avg_rider_time"] = 0
            metrics["variance_rider_time"] = 0
            
        return metrics

if __name__ == "__main__":
    base_lambda_rate = 5.0
    sim_duration = 5000
    warmup_period = 500
    period_length = 100
    radius = 1
    
    utilization_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    m_thresholds = range(1, 11)
    
    results_by_util = {}
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
        
    for i, util_rate in enumerate(utilization_rates):
        print(f"\nTesting system utilization = {util_rate:.1f}")
        
        mu_rate = base_lambda_rate / util_rate
        
        util_results = []
        
        for m in m_thresholds:
            print(f"  Testing m_threshold = {m}")
            
            sim = SimulationTemporalPooling(
                lambda_rate=base_lambda_rate,
                mu_rate=mu_rate,
                sim_duration=sim_duration,
                warmup_period=warmup_period,
                period_length=period_length,
                radius=radius,
                m_threshold=m
            )
            
            sim.run()
            metrics = sim.calculate_metrics()
            time_components = sim.get_time_components()
            
            util_results.append({
                "m_threshold": m,
                "avg_queue_time": time_components["avg_queue_time"],
                "avg_pickup_time": time_components["avg_pickup_time"],
                "avg_travel_time": time_components["avg_travel_time"],
                "total_time": metrics["avg_rider_time"]
            })
            
            print(f"  Queue time: {time_components['avg_queue_time']:.3f}, "
                  f"Pickup time: {time_components['avg_pickup_time']:.3f}, "
                  f"Travel time: {time_components['avg_travel_time']:.3f}")
            print(f"  Total rider time: {metrics['avg_rider_time']:.3f}")
        
        results_by_util[util_rate] = util_results
    
    