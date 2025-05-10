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
        
    def _when_driver_arrival(self, event):
        """
        Use to handle driver arrival events with distance constraints.
        
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


def run_distance_comparison():
    """
    Run simulations with different distance constraints and arrival rate ratios to compare their performance.
    
    Returns:
        dict: Results for different parameter combinations.
    """

    sim_duration = 5000
    warmup_period = 500
    
    distance_limits = [0.4, 0.6, 0.8, 1.0]
    
    scenarios = [
        {"name": "Equal rates", "lambda": 5, "mu": 5},
        {"name": "More drivers", "lambda": 5, "mu": 7},
        {"name": "More riders", "lambda": 7, "mu": 5},
        {"name": "Low volume", "lambda": 3, "mu": 3},
        {"name": "High volume", "lambda": 10, "mu": 10}
    ]
    
    results = {}
    
    for scenario in scenarios:
        scenario_name = scenario["name"]
        lambda_rate = scenario["lambda"]
        mu_rate = scenario["mu"]
        
        print(f"\nScenario: {scenario_name} (λ={lambda_rate}, μ={mu_rate})")
        print("=" * 50)
        
        scenario_results = {}
        
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
            
            scenario_results[l] = {
                "avg_rider_time": metrics["avg_rider_time"],
                "driver_matching_rate": metrics["driver_matching_rate"],
                "total_riders": metrics["total_riders"],
                "total_drivers": metrics["total_drivers"],
                "matched_drivers": metrics["matched_drivers"],
                "final_queue_length": len(model.rider_queue)
            }
            
            print(f"Average rider time: {metrics['avg_rider_time']:.2f} seconds")
            print(f"Driver matching rate: {metrics['driver_matching_rate']:.2f}")
            print(f"Final queue length: {len(model.rider_queue)}")
            print("-----------------------------------")
        
        results[scenario_name] = scenario_results
        
        plot_distance_comparison(scenario_results, title=f"Results for {scenario_name} (λ={lambda_rate}, μ={mu_rate})")
    
    plot_scenario_comparison(results)
    
    return results


def plot_distance_comparison(results, title="Distance Limit Comparison"):
    """
    Plot the comparison of different distance constraints.
    
    Parameters:
        results (dict): Results from different simulations with varying distance constraints.
        title (str): Title for the plot.
    """
    distance_limits = list(results.keys())
    avg_rider_times = [results[l]["avg_rider_time"] for l in distance_limits]
    driver_matching_rates = [results[l]["driver_matching_rate"] for l in distance_limits]
    final_queue_lengths = [results[l]["final_queue_length"] for l in distance_limits]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.plot(distance_limits, avg_rider_times, 'o-', color='blue')
    ax1.set_xlabel('Distance Limit (l)')
    ax1.set_ylabel('Average Rider Time (seconds)')
    ax1.set_title('Average Rider Time vs Distance Limit')
    ax1.grid(True)
    
    ax2.plot(distance_limits, driver_matching_rates, 'o-', color='green')
    ax2.set_xlabel('Distance Limit (l)')
    ax2.set_ylabel('Driver Matching Rate')
    ax2.set_title('Driver Matching Rate vs Distance Limit')
    ax2.grid(True)
    
    ax3.plot(distance_limits, final_queue_lengths, 'o-', color='red')
    ax3.set_xlabel('Distance Limit (l)')
    ax3.set_ylabel('Final Queue Length')
    ax3.set_title('Final Queue Length vs Distance Limit')
    ax3.grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "_") + ".png"
    plt.savefig(filename)
    plt.show()


def plot_scenario_comparison(results):
    """
    Plot comparison across different scenarios.
    
    Parameters:
        results (dict): Results from different scenarios and distance limits.
    """
    scenarios = list(results.keys())
    distance_limits = list(results[scenarios[0]].keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for l in distance_limits:
        avg_rider_times = [results[scenario][l]["avg_rider_time"] for scenario in scenarios]
        driver_matching_rates = [results[scenario][l]["driver_matching_rate"] for scenario in scenarios]
        
        ax1.plot(scenarios, avg_rider_times, 'o-', label=f'l = {l}')
        ax2.plot(scenarios, driver_matching_rates, 'o-', label=f'l = {l}')
    
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Average Rider Time (seconds)')
    ax1.set_title('Average Rider Time Across Scenarios')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xticklabels(scenarios, rotation=45)
    
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Driver Matching Rate')
    ax2.set_title('Driver Matching Rate Across Scenarios')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xticklabels(scenarios, rotation=45)
    
    plt.suptitle('Comparison Across Different Scenarios', fontsize=16)
    plt.tight_layout()
    plt.savefig('scenario_comparison.png')
    plt.show()


if __name__ == "__main__":
    results = run_distance_comparison()
    
    print("\nSummary of Results:")
    print("-------------------")
    for scenario, scenario_results in results.items():
        print(f"\nScenario: {scenario}")
        print("-" * 30)
        for l, metrics in scenario_results.items():
            print(f"Distance Limit (l) = {l}:")
            print(f"  Average Rider Time: {metrics['avg_rider_time']:.2f} seconds")
            print(f"  Driver Matching Rate: {metrics['driver_matching_rate']:.2f}")
            print(f"  Total Riders: {metrics['total_riders']}")
            print(f"  Matched Drivers: {metrics['matched_drivers']} out of {metrics['total_drivers']}")
            print(f"  Final Queue Length: {metrics['final_queue_length']}")
            print()
