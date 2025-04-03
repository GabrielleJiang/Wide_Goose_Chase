import math
import numpy as np
import heapq
import matplotlib.pyplot as plt
from collections import deque
from Baseline_Model import(SimulationBaselineModel, 
                           Rider, 
                           RiderDeparture, 
                           distance_of_polar_coordinates)


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
    def __init__(self, lambda_rate, mu_rate, n_zones = 4, sim_duration = 5000, warmup_period = 500, period_length = 100, radius = 1):
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
        for zone_id in range(n_zones):
            self.zone_queues[zone_id] = deque()
        
        self.zone_stats = {}
        for zone_id in range(n_zones):
            self.zone_stats[zone_id] = {
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
        if self.is_warmed_up:
            self.total_riders += 1
            self.rider_arrival_times.append(event.time)
            self.zone_stats[zone_id]["total_riders"] += 1
        
        self.zone_queues[zone_id].append(rider)
        self._when_next_rider_arrival()
    

    def _when_driver_arrival(self, event):
        """
        Handle driver arrival event.
        
        Parameters:
            event (DriverArrival): The driver arrival event.
        """
        driver_zone_id = self._get_zone_id(event.origin)
        
        if self.is_warmed_up:
            self.total_drivers += 1
            self.driver_arrival_times.append(event.time)
            self.zone_stats[driver_zone_id]["total_drivers"] += 1
        
        if self.zone_queues[driver_zone_id]:
            rider = self.zone_queues[driver_zone_id].popleft()
            rider.matched_time = event.time
            pickup_distance = distance_of_polar_coordinates(event.origin, rider.origin)
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
                self.zone_stats[driver_zone_id]["matched_drivers"] += 1
                
        self._when_next_driver_arrival()
    

    def _process_rider_departure(self, event):
        """
        Process rider departure event.
        
        Parameters:
            event (RiderDeparture): The rider departure event.
        """
        if self.is_warmed_up:
            rider = event.rider
            origin_zone_id = self._get_zone_id(rider.origin)           
            self.rider_times.append(rider.total_time)
            self.wait_times.append(rider.queue_time)
            self.pickup_times.append(rider.pickup_wait_time)
            self.travel_times.append(rider.travel_time)           
            self.zone_stats[origin_zone_id]["rider_times"].append(rider.total_time)
            self.zone_stats[origin_zone_id]["queue_times"].append(rider.queue_time)
            self.zone_stats[origin_zone_id]["pickup_wait_times"].append(rider.pickup_wait_time)
            self.zone_stats[origin_zone_id]["travel_times"].append(rider.travel_time)
    

    def run(self):
        """
        Run the simulation and return the results.
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
            elif event.event_type == "rider_departure":
                self._process_rider_departure(event)
        
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
        
        if self.wait_times:
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
        
        zone_results = {}
        for zone_id in range(self.n_zones):
            stats = self.zone_stats[zone_id]
            
            if stats["rider_times"]:
                zone_avg_time = np.mean(stats["rider_times"])
            else:
                zone_avg_time = 0
                
            if stats["rider_times"]:
                zone_variance_time = np.var(stats["rider_times"])
            else:
                zone_variance_time = 0
            
            if stats["queue_times"]:
                zone_avg_queue_time = np.mean(stats["queue_times"])
            else:
                zone_avg_queue_time = 0
            
            if stats["pickup_wait_times"]:
                zone_avg_pickup_time = np.mean(stats["pickup_wait_times"])
            else:
                zone_avg_pickup_time = 0
            
            if stats["travel_times"]:
                zone_avg_travel_time = np.mean(stats["travel_times"])
            else:
                zone_avg_travel_time = 0
            
            zone_total_drivers = stats["total_drivers"]
            zone_matched_drivers = stats["matched_drivers"]
            
            if zone_total_drivers > 0:
                zone_matching_rate = zone_matched_drivers / zone_total_drivers
            else:
                zone_matching_rate = 0
            
            zone_results[zone_id] = {
                "avg_rider_time": zone_avg_time,
                "avg_queue_time": zone_avg_queue_time,
                "avg_pickup_time": zone_avg_pickup_time,
                "avg_travel_time": zone_avg_travel_time,
                "variance_rider_time": zone_variance_time,
                "driver_matching_rate": zone_matching_rate,
                "total_riders": stats["total_riders"],
                "total_drivers": zone_total_drivers,
                "matched_drivers": zone_matched_drivers
            }
        
        return {
            "avg_rider_time": avg_rider_time,
            "variance_rider_time": variance_rider_time,
            "driver_matching_rate": driver_matching_rate,
            "total_riders": self.total_riders,
            "total_drivers": self.total_drivers,
            "matched_drivers": self.matched_drivers,
            "avg_wait_time": avg_wait_time,
            "avg_pickup_time": avg_pickup_time,
            "avg_travel_time": avg_travel_time,
            "zone_results": zone_results
        }
    
    def run_zone_comparison(self, zone_numbers=[1, 2, 4, 6, 8], sim_duration=5000, num_trials=1):
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
            combined_travel_times = []
            
            for trial in range(num_trials):
                if num_trials > 1:
                    print(f"  Trial {trial+1}/{num_trials}")
                
                sim = SimulationZoningModel(
                    lambda_rate=self.lambda_rate,
                    mu_rate=self.mu_rate,
                    n_zones=n_zones,
                    sim_duration=sim_duration,
                    warmup_period=self.warmup_period
                )
                
                result = sim.run()
                avg_rider_times.append(result["avg_rider_time"])
                driver_match_rates.append(result["driver_matching_rate"])
                
                combined_travel_time = result["avg_pickup_time"] + result["avg_travel_time"]
                combined_travel_times.append(combined_travel_time)
            
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
                
            avg_combined_travel_time = np.mean(combined_travel_times)
            
            if num_trials > 1:
                std_combined_travel_time = np.std(combined_travel_times)
            else:
                std_combined_travel_time = 0
            
            results[n_zones] = {
                "avg_rider_time": avg_time,
                "std_rider_time": std_time,
                "driver_matching_rate": avg_match_rate,
                "std_driver_matching_rate": std_match_rate,
                "avg_combined_travel_time": avg_combined_travel_time,
                "std_combined_travel_time": std_combined_travel_time,
                "all_rider_times": avg_rider_times,
                "all_match_rates": driver_match_rates,
                "all_combined_travel_times": combined_travel_times
            }
        
        return results
    
    def run_utilization_comparison(self, zone_numbers = [1, 2, 4, 6, 8], 
                                  utilization_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
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
                
                for trial in range(num_trials):
                    if num_trials > 1:
                        print(f"    Trial {trial+1}/{num_trials}")                   
                    sim = SimulationZoningModel(
                        lambda_rate=lambda_rate,
                        mu_rate=base_mu_rate,
                        n_zones=n_zones,
                        sim_duration=sim_duration,
                        warmup_period=self.warmup_period
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


if __name__ == "__main__":
    zoning_model = SimulationZoningModel(
        lambda_rate=5.0, 
        mu_rate=6.0, 
        n_zones=4,
        sim_duration=5000,
        warmup_period=500
    )
    
    results = zoning_model.run()
    
    print("Overall Results:")
    print(f"Average Rider Time: {results['avg_rider_time']:.2f}")
    print(f"Average Wait Time: {results['avg_wait_time']:.2f}")
    print(f"Average Pickup Time: {results['avg_pickup_time']:.2f}")
    print(f"Average Travel Time: {results['avg_travel_time']:.2f}")
    print(f"Driver Matching Rate: {results['driver_matching_rate']:.2f}")
    print(f"Total Riders: {results['total_riders']}")
    print(f"Total Drivers: {results['total_drivers']}")
    print(f"Matched Drivers: {results['matched_drivers']}")
    
    print("\nZone Results:")
    for zone_id, zone_data in results["zone_results"].items():
        print(f"\nZone {zone_id}:")
        print(f"  Average Rider Time: {zone_data['avg_rider_time']:.2f}")
        print(f"  Average Queue Time: {zone_data['avg_queue_time']:.2f}")
        print(f"  Average Pickup Time: {zone_data['avg_pickup_time']:.2f}")
        print(f"  Average Travel Time: {zone_data['avg_travel_time']:.2f}")
        print(f"  Driver Matching Rate: {zone_data['driver_matching_rate']:.2f}")
        print(f"  Total Riders: {zone_data['total_riders']}")
        print(f"  Total Drivers: {zone_data['total_drivers']}")
        print(f"  Matched Drivers: {zone_data['matched_drivers']}")
    
    zone_comparison_results = zoning_model.run_zone_comparison(
        zone_numbers=[1, 2, 4, 6, 8, 10, 12], 
        num_trials=3
    )
    
    utilization_results = zoning_model.run_utilization_comparison(
        zone_numbers=[1, 2, 4, 6, 8, 10, 12],
        utilization_rates=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    zone_numbers = list(zone_comparison_results.keys())
    
    avg_times = []
    std_times = []
    for zone in zone_numbers:
        avg_time = zone_comparison_results[zone]['avg_rider_time']
        std_time = zone_comparison_results[zone]['std_rider_time']
        avg_times.append(avg_time)
        std_times.append(std_time)
    
    plt.errorbar(zone_numbers, avg_times, yerr=std_times, fmt='o-', capsize=5)
    plt.xlabel('Number of Zones')
    plt.ylabel('Average Rider Time')
    plt.title('Average Rider Time vs Number of Zones')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    
    match_rates = []
    std_rates = []
    for zone in zone_numbers:
        match_rate = zone_comparison_results[zone]['driver_matching_rate']
        std_rate = zone_comparison_results[zone]['std_driver_matching_rate']
        match_rates.append(match_rate)
        std_rates.append(std_rate)
    
    plt.errorbar(zone_numbers, match_rates, yerr=std_rates, fmt='o-', capsize=5)
    plt.xlabel('Number of Zones')
    plt.ylabel('Driver Matching Rate')
    plt.title('Driver Matching Rate vs Number of Zones')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('zone_comparison_results.png', dpi=300)
    plt.show()
    
    plt.figure(figsize=(8, 5))
    
    travel_times = []
    travel_time_stds = []
    for zone in zone_numbers:
        travel_time = zone_comparison_results[zone]['avg_combined_travel_time']
        travel_time_std = zone_comparison_results[zone]['std_combined_travel_time']
        travel_times.append(travel_time)
        travel_time_stds.append(travel_time_std)
    
    plt.errorbar(
        x=zone_numbers, 
        y=travel_times, 
        yerr=travel_time_stds, 
        fmt='o-', 
        capsize=5, 
        color='green'
    )
    plt.xlabel('Number of Zones')
    plt.ylabel('Average Travel Time (Pickup + Enroute)')
    plt.title('Average Travel Time (Pickup + Enroute) vs Number of Zones')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('travel_time_vs_zones.png', dpi=300)
    plt.show()
    
    plt.figure(figsize=(10, 5))
    
    zone_list = [1, 2, 4, 6, 8, 10, 12]
    
    for n_zones in zone_list:
        times = []
        stds = []
        
        util_rates = list(utilization_results.keys())
        
        for util_rate in util_rates:
            time_value = utilization_results[util_rate][n_zones]['avg_rider_time']
            std_value = utilization_results[util_rate][n_zones]['std_rider_time']
            times.append(time_value)
            stds.append(std_value)
        
        plt.errorbar(util_rates, times, yerr = stds, fmt = 'o-', label = f'{n_zones} zones', capsize = 5)
    
    plt.xlabel('Utilization Rate')
    plt.ylabel('Average Rider Time')
    plt.title('Average Rider Time vs Utilization Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('utilization_comparison_results.png', dpi = 300)
    plt.show()