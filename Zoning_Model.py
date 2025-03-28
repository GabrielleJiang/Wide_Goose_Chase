import matplotlib.pyplot as plt
import math
import random
import heapq
import uuid
import numpy as np
from collections import deque
from Baseline_Model import (
    SimulationBaselineModel, 
    Event, 
    RiderArrival, 
    RiderDeparture, 
    DriverArrival, 
    Rider, 
    Driver, 
    uniform_distributed_in_disk, 
    distance_of_polar_coordinates, 
    calculate_time_period
)


class SimulationZoningModel(SimulationBaselineModel):
    """
    Zoning model for ride-hailing simulation where the circular area is divided into zones.
    Riders and drivers are matched only within the same zone.
    
    Attributes:
        n_zones (int): Number of zones to divide the circular area into.
        zone_queues (dict): Dictionary of queues for each zone.
        zone_stats (dict): Statistics for each zone.
        wait_times (list): List of all waiting times across all zones.
        pickup_times (list): List of all pickup times across all zones.
        travel_times (list): List of all travel times across all zones.
    """
    
    def __init__(self, lambda_rate, mu_rate, n_zones = 4, sim_duration = 5000, 
                 warmup_period = 500, period_length = 100, radius = 1):
        """
        Initialize the zoning model.
        
        Parameters:
            lambda_rate (float): Arrival rate of riders.
            mu_rate (float): Arrival rate of drivers.
            n_zones (int): Number of zones.
            sim_duration (float): Number of simulation.
            warmup_period (float): Warm-up period.
            period_length (float): Length of each time period.
            radius (float): Radius of the service area.
        """
        super().__init__(lambda_rate, mu_rate, sim_duration, warmup_period, period_length, radius)
        
        self.n_zones = n_zones
        
        self.zone_queues = {}
        for i in range(n_zones):
            self.zone_queues[i] = deque()
        
        self.zone_stats = {}
        for i in range(n_zones):
            self.zone_stats[i] = {
                "total_riders": 0,
                "total_drivers": 0,
                "matched_drivers": 0,
                "rider_times": [],
                "queue_times": [],
                "pickup_wait_times": [],
                "travel_times": []
            }
        
        self.wait_times = []
        self.pickup_times = []
        self.travel_times = []
        self.rider_queue = deque()

    def _get_zone_id(self, coordinates):
        """
        Determine which zone a point belongs to based on its polar coordinates.
        
        Parameters:
            coordinates (tuple): Polar coordinates (r, θ) of the point.
        """
        _, theta = coordinates
        if theta < 0:
            theta += 2 * math.pi   
        zone_width = 2 * math.pi / self.n_zones
        zone_id = int(theta / zone_width)
    
        return min(zone_id, self.n_zones - 1)


    def _when_rider_arrival(self, event):
        """
        Handle rider arrival event.
        
        Parameters:
            event (RiderArrival): The rider arrival event.
        """
        rider = Rider(event.rider_id, event.time, event.origin, event.destination, event.period)
        zone_id = self._get_zone_id(event.origin)
        self.zone_queues[zone_id].append(rider)
        if self.is_warmed_up:
            self.total_riders += 1
            self.rider_arrival_times.append(event.time)
            self.zone_stats[zone_id]["total_riders"] += 1
        
        self._when_next_rider_arrival()


    def _when_driver_arrival(self, event):
        """
        Handle driver arrival event.
        
        Parameters:
            event (DriverArrival): The driver arrival event.
        """
        driver = Driver(event.driver_id, event.time, event.origin, event.period)
        zone_id = self._get_zone_id(event.origin)
        
        if self.is_warmed_up:
            self.total_drivers += 1
            self.driver_arrival_times.append(event.time)
            self.zone_stats[zone_id]["total_drivers"] += 1
        
        if self.zone_queues[zone_id]:
            rider = self.zone_queues[zone_id].popleft()
            rider.matched_time = event.time
            pickup_distance = distance_of_polar_coordinates(driver.origin, rider.origin)
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
                self.zone_stats[zone_id]["matched_drivers"] += 1
        

    def _process_rider_departure(self, event):
        """
        Process rider departure event.
        
        Parameters:
            event (RiderDeparture): The rider departure event.
        """
        if self.is_warmed_up:
            rider = event.rider
            zone_id = self._get_zone_id(rider.origin)
            self.rider_times.append(rider.total_time)
            self.zone_stats[zone_id]["rider_times"].append(rider.total_time)
            self.zone_stats[zone_id]["queue_times"].append(rider.queue_time)
            self.zone_stats[zone_id]["pickup_wait_times"].append(rider.pickup_wait_time)
            self.zone_stats[zone_id]["travel_times"].append(rider.travel_time)
            self.wait_times.append(rider.queue_time)
            self.pickup_times.append(rider.pickup_wait_time)
            self.travel_times.append(rider.travel_time)


    def run(self):
        """
        Run the simulation and return the results.
        
        Returns:
            dict: Simulation results including global and per-zone statistics.
        """
        self._when_next_rider_arrival()
        self._when_next_driver_arrival()
        self.wait_times = []
        self.pickup_times = []
        self.travel_times = []
        
        while self.event_queue and self.current_time < self.sim_duration:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            if int(self.current_time) % 100 == 0:
                total_queue_length = sum(len(queue) for queue in self.zone_queues.values())
                self.queue_length_history.append(total_queue_length)
                self.time_points.append(self.current_time)
            if not self.is_warmed_up and self.current_time >= self.warmup_period:
                self.is_warmed_up = True
                self.total_riders = 0
                self.total_drivers = 0
                self.matched_drivers = 0
                self.rider_times = []
                self.rider_arrival_times = []
                self.driver_arrival_times = []
                self.wait_times = []
                self.pickup_times = []
                self.travel_times = []
                for zone_id in range(self.n_zones):
                    self.zone_stats[zone_id] = {
                        "total_riders": 0,
                        "total_drivers": 0,
                        "matched_drivers": 0,
                        "rider_times": [],
                        "queue_times": [],
                        "pickup_wait_times": [],
                        "travel_times": []
                    }
            
            if event.event_type == "rider_arrival":
                self._when_rider_arrival(event)
            elif event.event_type == "driver_arrival":
                self._when_driver_arrival(event)
                self._when_next_driver_arrival()
            elif event.event_type == "rider_departure":
                self._process_rider_departure(event)
        
        # calculate average time of riders
        if len(self.rider_times) > 0:
            avg_rider_time = np.mean(self.rider_times)
        else:
            avg_rider_time = 0
        
        # calculate variance of riders
        if len(self.rider_times) > 0:
            variance_rider_time = np.var(self.rider_times)
        else:
            variance_rider_time = 0
        
        # calculate matching rate of drivers
        if self.total_drivers > 0:
            driver_matching_rate = self.matched_drivers / self.total_drivers
        else:
            driver_matching_rate = 0
        
        # calculate average wait time
        if len(self.wait_times) > 0:
            avg_wait_time = np.mean(self.wait_times)
        else:
            avg_wait_time = 0
            
        # calculate average pickup time
        if len(self.pickup_times) > 0:
            avg_pickup_time = np.mean(self.pickup_times)
        else:
            avg_pickup_time = 0
            
        # calculate average travel time
        if len(self.travel_times) > 0:
            avg_travel_time = np.mean(self.travel_times)
        else:
            avg_travel_time = 0

        result = {
            "avg_rider_time": avg_rider_time,
            "variance_rider_time": variance_rider_time,
            "driver_matching_rate": driver_matching_rate,
            "total_riders": self.total_riders,
            "total_drivers": self.total_drivers,
            "matched_drivers": self.matched_drivers,
            "avg_wait_time": avg_wait_time,
            "avg_pickup_time": avg_pickup_time,
            "avg_travel_time": avg_travel_time
        }
        
        #the metrics in zone
        zone_results = {}
        for zone_id in range(self.n_zones):
            stats = self.zone_stats[zone_id]
            
            # calculate average rider time for zone
            if len(stats["rider_times"]) > 0:
                avg_rider_time = np.mean(stats["rider_times"])
            else:
                avg_rider_time = 0
                
            # calculate average queue time for zone
            if len(stats["queue_times"]) > 0:
                avg_queue_time = np.mean(stats["queue_times"])
            else:
                avg_queue_time = 0
                
            # calculate average pickup time for zone
            if len(stats["pickup_wait_times"]) > 0:
                avg_pickup_time = np.mean(stats["pickup_wait_times"])
            else:
                avg_pickup_time = 0
                
            # calculate average travel time for zone
            if len(stats["travel_times"]) > 0:
                avg_travel_time = np.mean(stats["travel_times"])
            else:
                avg_travel_time = 0
            
            # calculate variance in rider time for zone
            if len(stats["rider_times"]) > 0:
                var_rider_time = np.var(stats["rider_times"])
            else:
                var_rider_time = 0
            
            # calculate driver matching rate for zone
            if stats["total_drivers"] > 0:
                driver_matching_rate = stats["matched_drivers"] / stats["total_drivers"]
            else:
                driver_matching_rate = 0
            
            zone_results[zone_id] = {
                "avg_rider_time": avg_rider_time,
                "avg_queue_time": avg_queue_time,
                "avg_pickup_time": avg_pickup_time,
                "avg_travel_time": avg_travel_time,
                "variance_rider_time": var_rider_time,
                "driver_matching_rate": driver_matching_rate,
                "total_riders": stats["total_riders"],
                "total_drivers": stats["total_drivers"],
                "matched_drivers": stats["matched_drivers"]
            }
        
        result["zone_results"] = zone_results
        
        return result
    
    def run_zone_comparison(self, zone_numbers = [2, 4, 6, 8], sim_duration = 5000, num_trials = 1):
        """
        Run simulations with different numbers of zones and compare metrics.
        
        Parameters:
            zone_numbers (list): List of zone numbers to compare.
            sim_duration (float): Duration of each simulation.
            num_trials (int): Number of trials to run for each zone configuration.
        """
        results = {}
        
        for n_zones in zone_numbers:
            print(f"Running simulation with {n_zones} zones...")
            
            avg_rider_times = []
            driver_match_rates = []
            
            for trial in range(num_trials):
                if num_trials > 1:
                    print(f"  Trial {trial+1}/{num_trials}")
                sim = SimulationZoningModel(
                    lambda_rate = self.lambda_rate,
                    mu_rate = self.mu_rate,
                    n_zones = n_zones,
                    sim_duration = sim_duration,
                    warmup_period = self.warmup_period
                )
                
                result = sim.run()
                avg_rider_times.append(result["avg_rider_time"])
                driver_match_rates.append(result["driver_matching_rate"])
            
            avg_time = np.mean(avg_rider_times)
            
            if num_trials > 1:
                std_time = np.std(avg_rider_times)
            else:
                std_time = 0
            
            avg_match_rate = np.mean(driver_match_rates)

            if num_trials > 1:
                std_match_rate = np.std(driver_match_rates)
            else:
                std_match_rate = 0
            
            results[n_zones] = {
                "avg_rider_time": avg_time,
                "std_rider_time": std_time,
                "driver_matching_rate": avg_match_rate,
                "std_driver_matching_rate": std_match_rate,
                "all_rider_times": avg_rider_times,
                "all_match_rates": driver_match_rates
            }
        
        return results


    def run_utilization_comparison(self, zone_numbers = [2, 4, 6, 8], utilization_rates = [0.6, 0.7, 0.8, 0.9], 
                               sim_duration = 5000, num_trials = 3):
        """
        Run simulations with different utilization rates and zone numbers to compare metrics.
        
        Parameters:
            zone_numbers (list): List of zone numbers to compare.
            utilization_rates (list): List of utilization rates.
            sim_duration (float): Duration of each simulation.
            num_trials (int): Number of trials.
        """
        results = {}
        base_mu_rate = 5.0
        
        for util_rate in utilization_rates:
            print(f"\nRunning simulations with utilization rate = {util_rate}...")
            
            lambda_rate = util_rate * base_mu_rate
            results[util_rate] = {}
            
            for n_zones in zone_numbers:
                print(f"  Number of zones: {n_zones}")
                avg_rider_times = []
                driver_match_rates = []
                std_devs = []
                for trial in range(num_trials):
                    if num_trials > 1:
                        print(f"    Trial {trial+1}/{num_trials}")
                    sim = SimulationZoningModel(
                        lambda_rate = lambda_rate,
                        mu_rate = base_mu_rate,
                        n_zones = n_zones,
                        sim_duration = sim_duration,
                        warmup_period = self.warmup_period
                    )
                    result = sim.run()
                    avg_rider_times.append(result["avg_rider_time"])
                    driver_match_rates.append(result["driver_matching_rate"])

                avg_time = np.mean(avg_rider_times)
                if num_trials > 1:
                    std_time = np.std(avg_rider_times)
                else:
                    std_time = 0
                avg_match_rate = np.mean(driver_match_rates)
                if num_trials > 1:
                    std_match_rate = np.std(driver_match_rates)
                else:
                    std_match_rate = 0
                
                results[util_rate][n_zones] = {
                    "avg_rider_time": avg_time,
                    "std_rider_time": std_time,
                    "driver_matching_rate": avg_match_rate,
                    "std_driver_matching_rate": std_match_rate,
                    "all_rider_times": avg_rider_times,
                    "all_match_rates": driver_match_rates
                }
        
        return results


def grid_search_zones(lambda_values, mu_ratios, zone_numbers, sim_duration = 5000):
    """
    Simple grid search to find the best combination.

    Parameters:
        lambda_values (list): Different rider arrival rates
        mu_ratios (list): Different rider/driver ratios
        zone_numbers (list): Different numbers of zones
        sim_duration (float): How long to run 
    """
    best_params = None
    best_score = float('inf')
    all_results = []
    
    warmup_period = 500
    
    for lambda_rate in lambda_values:
        for ratio in mu_ratios:
            mu_rate = lambda_rate / ratio
            for n_zones in zone_numbers:
                print(f"Testing: λ = {lambda_rate:.1f}, μ = {mu_rate:.1f}, ratio λ/μ = {ratio:.2f}, zones = {n_zones}")
                
                sim = SimulationZoningModel(
                    lambda_rate = lambda_rate, 
                    mu_rate = mu_rate,
                    n_zones = n_zones,
                    sim_duration = sim_duration,
                    warmup_period = warmup_period
                )
                
                results = sim.run()
                avg_rider_time = results["avg_rider_time"]
                driver_match_rate = results["driver_matching_rate"]
                score = avg_rider_time - 2 * driver_match_rate
                result_entry = {
                    "lambda_rate": lambda_rate,
                    "mu_rate": mu_rate,
                    "ratio": ratio,
                    "n_zones": n_zones,
                    "avg_rider_time": avg_rider_time,
                    "driver_match_rate": driver_match_rate,
                    "score": score
                }
                all_results.append(result_entry)
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        "lambda_rate": lambda_rate,
                        "mu_rate": mu_rate,
                        "ratio": ratio,
                        "n_zones": n_zones
                    }
    
    return best_params, all_results


def plot_grid_search_results(all_results, filename = "grid_search_results.png"):
    """
    Plot the results of the grid search to visualize how different parameters affect performance.
    
    Parameters:
        all_results (list): List of dictionaries
        filename (str): Name of file
    """
    zones = set()
    ratios = set()
    
    for result in all_results:
        zones.add(result["n_zones"])
        ratios.add(result["ratio"])
    
    zones = sorted(zones)
    ratios = sorted(ratios)
    plt.figure(figsize = (12, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for i, ratio in enumerate(ratios):
        x_points = []
        y_points = []
        for zone in zones:
            matching_results = []
            for result in all_results:
                if result["ratio"] == ratio and result["n_zones"] == zone:
                    matching_results.append(result)
            if matching_results:
                total_avg_time = 0
                for r in matching_results:
                    total_avg_time += r["avg_rider_time"]
                avg_time = total_avg_time / len(matching_results)
                x_points.append(zone)
                y_points.append(avg_time)
        
        plt.plot(
            x_points, y_points,
            marker = 'o', 
            linestyle = '-', 
            color = colors[i % len(colors)],
            label = f'λ/μ = {ratio:.2f}'
        )
    
    plt.title("Average Total Time vs Number of Zones", fontsize = 14)
    plt.xlabel("Number of Zones", fontsize = 12)
    plt.ylabel("Average Total Time (wait + pickup + travel)", fontsize = 12)
    plt.grid(True)
    plt.legend()
    
    plt.savefig(filename)
    plt.close()
    print(f"Grid search results plot saved as '{filename}'")


def extended_main():
    """
    Run an extended grid search to find the best combination of parameters for the zoning model.
    
    This function:
    1. Tests different rider and driver arrival rates
    2. Tests different numbers of zones
    3. Finds the best combination
    4. Shows the final results including:
       - How many riders and drivers we had
       - How many matches we made
       - How long riders waited on average
       - How often drivers found rides
    5. Creates plots to visualize the results
    """
    lambda_values = [4.0, 5.0, 6.0]
    mu_ratios = [0.7, 0.8, 0.9, 0.95]
    zone_numbers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    print("Starting grid search to find optimal parameters:")
    best_params, all_results = grid_search_zones(lambda_values, mu_ratios, zone_numbers)

    print("\nBest parameters found:")
    print(f"Rider arrival rate (λ): {best_params['lambda_rate']:.3f}")
    print(f"Rider/Driver ratio (λ/μ): {best_params['ratio']:.3f}")
    print(f"Driver arrival rate (μ): {best_params['mu_rate']:.3f}")
    print(f"Number of zones: {best_params['n_zones']}")
    
    print("\nRunning final simulation with best parameters...")
    final_sim = SimulationZoningModel(
        lambda_rate = best_params['lambda_rate'],
        mu_rate = best_params['mu_rate'],
        n_zones = best_params['n_zones'],
        sim_duration = 5000,
        warmup_period = 500
    )
    
    results = final_sim.run()
    
    print("\nFinal simulation results:")
    print(f"Total riders: {results['total_riders']}")
    print(f"Total drivers: {results['total_drivers']}")
    print(f"Matched drivers: {results['matched_drivers']}")
    print(f"Average rider time: {results['avg_rider_time']:.3f}")
    print(f"Variance in rider time: {results['variance_rider_time']:.3f}")
    print(f"Driver matching rate: {results['driver_matching_rate']:.3f}")
    
    print("\nCreating result plots...")
    plot_grid_search_results(all_results)
    
    print("\nPer-zone statistics from best configuration:")
    for zone_id, zone_stats in results["zone_results"].items():
        print(f"\nZone {zone_id}:")
        print(f"  Total riders: {zone_stats['total_riders']}")
        print(f"  Total drivers: {zone_stats['total_drivers']}")
        print(f"  Matched drivers: {zone_stats['matched_drivers']}")
        print(f"  Average rider time: {zone_stats['avg_rider_time']:.3f}")
        print(f"  Driver matching rate: {zone_stats['driver_matching_rate']:.3f}")


def main():
    """
    Run the zoning model simulation with different configurations to analyze performance.
    """
    lambda_rate = 5.0
    mu_rate = 6.0
    n_zones = 4
    sim_duration = 5000
    warmup_period = 500
    
    sim = SimulationZoningModel(
        lambda_rate = lambda_rate,
        mu_rate = mu_rate,
        n_zones = n_zones,
        sim_duration = sim_duration,
        warmup_period = warmup_period
    )
    
    print("Running single zone simulation...")
    results = sim.run()
    
    print("\nOverall results:")
    print(f"Total riders: {results['total_riders']}")
    print(f"Total drivers: {results['total_drivers']}")
    print(f"Matched drivers: {results['matched_drivers']}")
    print(f"Average wait time: {results['avg_wait_time']:.3f}")
    print(f"Average pickup time: {results['avg_pickup_time']:.3f}")
    print(f"Average travel time: {results['avg_travel_time']:.3f}")
    print(f"Average total time: {results['avg_rider_time']:.3f}")
    print(f"Driver matching rate: {results['driver_matching_rate']:.3f}")
    
    print("\nRunning zone number comparison...")
    zone_numbers = [2, 4, 6, 8, 10]
    results = sim.run_zone_comparison(zone_numbers, sim_duration, num_trials = 3)
    
    print("\nZone comparison results:")
    print("------------------------")
    print("Zone Number | Avg. Rider Time | Driver Matching Rate")
    print("------------------------")
    for n_zones in sorted(results.keys()):
        print(f"     {n_zones}     |      {results[n_zones]['avg_rider_time']:.3f}     |        {results[n_zones]['driver_matching_rate']*100:.1f}%")
    
    print("\nRunning utilization rate comparison...")
    utilization_rates = [0.6, 0.7, 0.8, 0.9]
    results = sim.run_utilization_comparison(zone_numbers, utilization_rates, sim_duration, num_trials = 3)
    
    print("\nSummary of optimal zone numbers:")
    print("------------------------------------")
    print("Utilization Rate | Optimal for Time | Optimal for Matching")
    print("------------------------------------")
    
    for util_rate in utilization_rates:
        util_results = results[util_rate]
        
        zones = sorted(util_results.keys())
        
        # Find best zone for time
        avg_times = []
        for z in zones:
            avg_times.append(util_results[z]["avg_rider_time"])
        
        min_time_index = np.argmin(avg_times)
        best_time_zone = zones[min_time_index]
        
        # Find best zone for matching
        match_rates = []
        for z in zones:
            match_rates.append(util_results[z]["driver_matching_rate"])
        
        max_rate_index = np.argmax(match_rates)
        best_match_zone = zones[max_rate_index]
        
    print("\nSimulation complete.")


if __name__ == "__main__":
    use_grid_search = True
    
    if use_grid_search:
        extended_main()
    else:
        main()