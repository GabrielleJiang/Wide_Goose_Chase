import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Base_Line_Model.Baseline_Model import (
    SimulationBaselineModel, 
    distance_of_polar_coordinates,
    RiderDeparture,
    Driver
)


class SimulationLocalizationModel(SimulationBaselineModel):
    """
    A simulation model that extends the baseline model with geographic distance constraints.
    Localization model only allows drivers to match with riders within a certain distance.
    
    Attributes:
        l (float): Maximum allowed distance between driver and rider.
        completed_riders (list): List to store all completed rider trips.
        matched_rider_times (list): List to store times of successfully matched riders.
    """
    def __init__(self, lambda_rate, mu_rate, l=0.4, sim_duration=5000, warmup_period=500, period_length=100, radius=1):
        """
        Initialize the localization model with distance constraint.
        
        Parameters:
            lambda_rate (float): Arrival rate of riders.
            mu_rate (float): Arrival rate of drivers.
            l (float): Maximum allowed distance between driver and rider.
            sim_duration (float): Total duration of the simulation.
            warmup_period (float): Warm-up period.
            period_length (float): Time length for each period.
            radius (float): Radius of the disk.
        """
        super().__init__(lambda_rate, mu_rate, sim_duration, warmup_period, period_length, radius)
        self.l = l
        self.completed_riders = []
        self.matched_rider_times = []
        

    def run(self):
        """
        Run the simulation and return performance metrics.
        """
        self.completed_riders = []
        self.matched_rider_times = []
        return super().run()


    def _when_driver_arrival(self, event):
        """
        Handle driver arrival events with distance constraints.
        Only match drivers with riders within the maximum allowed distance.
        
        Parameters:
            event (DriverArrival): The driver arrival event.
        """
        if self.is_warmed_up:
            self.total_drivers += 1
            self.driver_arrival_times.append(event.time)
            
        driver = Driver(event.driver_id, event.time, event.origin, event.period)
        
        if self.rider_queue:
            rider = self.rider_queue[0]
            
            distance = distance_of_polar_coordinates(driver.origin, rider.origin)
            
            if distance <= self.l:
                self.rider_queue.popleft()
                
                rider.matched_time = event.time
                
                pickup_distance = distance
                rider.pickup_distance = pickup_distance
                trip_distance = distance_of_polar_coordinates(rider.origin, rider.destination)
                rider.trip_distance = trip_distance
                
                pickup_time = event.time + pickup_distance
                departure_time = pickup_time + trip_distance
                
                rider.pickup_time = pickup_time
                rider.departure_time = departure_time
                
                rider.queue_time = rider.matched_time - rider.arrival_time
                rider.pickup_wait_time = pickup_distance
                rider.travel_time = trip_distance
                rider.total_time = rider.queue_time + rider.pickup_wait_time + rider.travel_time
                
                departure_event = RiderDeparture(departure_time, rider)
                heapq.heappush(self.event_queue, departure_event)
                
                if self.is_warmed_up:
                    self.matched_drivers += 1
                    self.matched_rider_times.append(rider.total_time)


    def _when_rider_departure(self, event):
        """
        Process rider departure events.
        Add successfully matched riders to the completed_riders list.
        
        Parameters:
            event (RiderDeparture): The rider departure event.
        """
        if self.is_warmed_up and hasattr(event.rider, 'matched_time'):
            self.completed_riders.append(event.rider)
        super()._when_rider_departure(event)


def run_distance_comparison():
    """
    Run simulations with different distance constraints and arrival rate ratios.
    
    Returns:
        dict: Results for different parameter combinations.
    """
    sim_duration = 5000
    warmup_period = 500
    
    distance_limits = [0.2, 0.3, 0.4, 0.5]
    mu_rate = 100
    rho_values = [0.005, 0.01, 0.015, 0.02, 0.025]
    
    results = {}
    
    for rho in rho_values:
        lambda_rate = rho * mu_rate 
        print(f"\nAnalyzing with λ = {lambda_rate:.1f}, μ = {mu_rate} (ρ = {rho:.3f})")
        print("=" * 50)
        
        rho_results = {}
        
        for l in distance_limits:
            print(f"Running simulation with distance limit l = {l}")
            model = SimulationLocalizationModel(
                lambda_rate=lambda_rate,
                mu_rate=mu_rate,
                l=l,
                sim_duration=sim_duration,
                warmup_period=warmup_period
            )
            
            metrics = model.run()
            
            rho_results[l] = {
                "avg_rider_time": metrics["avg_rider_time"],
                "driver_matching_rate": metrics["driver_matching_rate"],
                "total_riders": metrics["total_riders"],
                "total_drivers": metrics["total_drivers"],
                "matched_drivers": metrics["matched_drivers"]
            }
            
            print(f"Average rider time: {metrics['avg_rider_time']:.2f} seconds")
            print(f"Driver matching rate: {metrics['driver_matching_rate']:.2f}")
            print("-----------------------------------")
        
        results[rho] = rho_results
    
    plot_combined_distance_comparison(results, mu_rate)
    plot_scenario_comparison(results)
    
    return results


def plot_combined_distance_comparison(results, mu_rate):
    """
    Plot the comparison of different distance constraints with all rho values in the same plot.
    
    Parameters:
        results (dict): Results from different simulations.
        mu_rate (float): Fixed driver arrival rate.
    """
    distance_limits = sorted(list(list(results.values())[0].keys()))
    rho_values = sorted(list(results.keys()))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(rho_values)))
    
    for rho, color in zip(rho_values, colors):
        lambda_rate = rho * mu_rate
        avg_rider_times = [results[rho][l]["avg_rider_time"] for l in distance_limits]
        driver_matching_rates = [results[rho][l]["driver_matching_rate"] for l in distance_limits]
        
        ax1.plot(distance_limits, avg_rider_times, 'o-', color=color, 
                label=f'ρ = {rho:.3f} (λ = {lambda_rate:.1f})')
        ax2.plot(distance_limits, driver_matching_rates, 'o-', color=color, 
                label=f'ρ = {rho:.3f} (λ = {lambda_rate:.1f})')
    
    ax1.set_xlabel('Distance Limit (l)')
    ax1.set_ylabel('Average Rider Time (seconds)')
    ax1.set_title('Average Rider Time vs Distance Limit')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('Distance Limit (l)')
    ax2.set_ylabel('Driver Matching Rate')
    ax2.set_title('Driver Matching Rate vs Distance Limit')
    ax2.grid(True)
    ax2.legend()
    
    plt.suptitle(f'System Performance Analysis (μ = {mu_rate})', fontsize=14)
    plt.tight_layout()
    plt.savefig('combined_distance_comparison.png')
    plt.show()
    plt.close()

def plot_scenario_comparison(results):
    """
    Plot comparison across different rho values.
    
    Parameters:
        results (dict): Results from different rho values and distance limits.
    """
    rho_values = sorted(list(results.keys()))
    distance_limits = sorted(list(results[rho_values[0]].keys()))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(distance_limits)))
    
    for l, color in zip(distance_limits, colors):
        avg_rider_times = [results[rho][l]["avg_rider_time"] for rho in rho_values]
        driver_matching_rates = [results[rho][l]["driver_matching_rate"] for rho in rho_values]
        
        ax1.plot(rho_values, avg_rider_times, 'o-', color=color, label=f'l = {l}')
        ax2.plot(rho_values, driver_matching_rates, 'o-', color=color, label=f'l = {l}')
    
    ax1.set_xlabel('ρ (λ/μ)')
    ax1.set_ylabel('Average Rider Time (seconds)')
    ax1.set_title('Average Rider Time vs ρ')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('ρ (λ/μ)')
    ax2.set_ylabel('Driver Matching Rate')
    ax2.set_title('Driver Matching Rate vs ρ')
    ax2.grid(True)
    ax2.legend()
    
    plt.suptitle(f'System Performance Analysis (μ = 100)', fontsize=14)
    plt.tight_layout()
    plt.savefig('rho_comparison.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    results = run_distance_comparison()
    
    print("\nSummary of Results:")
    print("-------------------")
    for rho, rho_results in results.items():
        lambda_rate = rho * 100
        print(f"\nρ = {rho:.3f} (λ = {lambda_rate:.1f}, μ = 100)")
        print("-" * 30)
        for l, metrics in rho_results.items():
            print(f"Distance Limit (l) = {l}:")
            print(f"  Average Rider Time: {metrics['avg_rider_time']:.2f} seconds")
            print(f"  Driver Matching Rate: {metrics['driver_matching_rate']:.2f}")
            print(f"  Total Riders: {metrics['total_riders']}")
            print(f"  Matched Drivers: {metrics['matched_drivers']} out of {metrics['total_drivers']}")
            print()