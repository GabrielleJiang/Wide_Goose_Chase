import matplotlib.pyplot as plt
import math
import random
import heapq
import uuid
import numpy as np
from collections import deque


class Event:
    """
    Class of the event in the simulation.

    Attribute:
        time (float): Time when the event happens.
    """
    def __init__(self, time):
        self.time = time
    
    def __lt__(self, other):
        """
        Compare events in terms of time.

        Parameters: 
            other (Event): Other event to compare with.
        """
        return self.time < other.time
    

class RiderArrival(Event):
    """
    A event for rider arrival.

    Attribute:
        time (float): Arrival time of the rider.
        rider_id (int): Unique identification of the rider.
        origin (tuple): (r, θ) coordinates of the origin of riders.
        destination (tuple): (r, θ) coordinates of the destination of riders.
        period (int): Time period when the rider arrives.
    """
    def __init__(self, time, rider_id, origin, destination, period):
        super().__init__(time)
        self.rider_id = rider_id
        self.origin = origin
        self.destination = destination
        self.period = period
        self.event_type = "rider_arrival"


class RiderDeparture(Event):
    """
    A event of the departure of rider, which mean this rider leave the system.
    
    Attributes:
        time (float): Departure time.
        rider (Rider): The rider who depart.
    """
    def __init__(self, time, rider):
        """
        Initialize a RiderDeparture event.
        
        Parameters:
            time (float): The departure time.
            rider (Rider): The rider associated with this event.
        """
        super().__init__(time)
        self.rider = rider
        self.event_type = "rider_departure"


class DriverArrival(Event):
    """
    A event for driver arrival.

    Attribute:
        time (float): The arrival time of the driver.
        driver_id (int): Unique identification of the driver.
        origin (tuole): (r, θ) coordinates of the origin of drivers.
        period (int): Time period when the driver arrives.
    """
    def __init__(self, time, origin, driver_id, period):
        """
        Initialize a DriverDeparture event.
        
        Parameters:
            time (float): The arrival time of the driver.
            driver_id (int): Unique identification of the driver.
            origin (tuole): (r, θ) coordinates of the origin of drivers.
            period (int): Time period when the driver arrives.
        """
        super().__init__(time)
        self.origin = origin
        self.driver_id = driver_id
        self.period = period
        self.event_type = "driver_arrival"



class Rider:
    """
    A class of the rider in the simulation

    Attribute:
        id (int): The unique identification of rider.
        arrival_time (float): Time when the rider arrives.
        origin (tuple): The origin of the rider.
        destination (tuple): The destination of the rider.
        period (int): The time period when the rider arrives.
        matched_time (float or None): Time when the rider is matched with a driver.
        pickup_time (float or None): Time when the driver reaches the rider's origin.
        departure_time (float or None): Time when the rider reaches the destination.
        queue_time (float or None): Waiting time in the queue from arrival to matching.
        pickup__wait_time (float or None): Waiting time for pickup from matching to pickup.
        travel_time (float or None): Travel time from pickup to destination.
        total_time (float or None): Total time from arrival to destination.
        pickup_distance (float or None): Distance from driver to rider at pickup.
        trip_distance (float or None): Distance from pickup to destination.
    """
    def __init__(self, rider_id, arrival_time, origin, destination, period):
        self.rider_id = rider_id
        self.arrival_time = arrival_time
        self.origin = origin
        self.destination = destination
        self.period = period
        
        self.matched_time = None
        self.pickup_time = None
        self.departure_time = None
        self.queue_time = None
        self.pickup_wait_time = None
        self.travel_time = None
        self.total_time = None
        
        self.pickup_distance = None
        self.trip_distance = None

class Driver:
    """
    Driver class.

    Attributes:
        driver_id (int): The unique identification of driver.
        arrival_time (float): Time the driver arrives.
        origin (tuple): The polar coordinates (r, θ) of the origin of the driver.
        period (int): The time period when the driver arrives.
        matched (bool): Indicates if the driver is matched with a rider.
        matched_rider_id (int or None): ID of the rider that this driver is matched with.
    """
    def __init__(self, driver_id, arrival_time, origin, period):
        self.driver_id = driver_id
        self.arrival_time = arrival_time
        self.origin = origin
        self.period = period
        self.matched = False
        self.matched_rider_id = None


def uniform_distributed_in_disk(radius_of_disk = 1):
    """
    Randomly generate a uniform distributed point within the disk with the radius of 1 in polar coordinates.

    Parameters:
        radiis_of_disk (folat): Radius of the disk is 1 in this baseline model.
    """
    theta = 2 * math.pi * random.random()
    valueofradius = radius_of_disk * math.sqrt(random.random())
    return (valueofradius, theta)
    

def distance_of_polar_coordinates(point_1, point_2):
    """
    Calculate the distance between two polar coordinates with in the disk.

    Parameters:
        point_1 (tuple): (r, θ) coordinates of the first point.
        point_2 (tuple): (r, θ) coordinates of the second point.
    """

    radius_1, theta_1 = point_1
    radius_2, theta_2 = point_2
    
    angle_diffience = abs(theta_1 - theta_2)
    if angle_diffience > math.pi:
        angle_diffience = 2 * math.pi - angle_diffience
    
    distance = math.sqrt(radius_1 ** 2 + radius_2 ** 2 - 2 * radius_1 * radius_2 * math.cos(angle_diffience))
    return distance


def calculate_time_period(time, len_period):
    """
    Calculate the time period.

    Parameters:
        time (float): Time.
        len_period (float): the time length.
    """
    return int(time / len_period)


def generate_uniformly_points(num_points = 1000, radius = 1, show_plot = True, save_path = None):
    """
    Generate and visualize points uniformly distributed in a disk.
    
    Parameters:
        num_points (int): Number of points to generate.
        radius (float): Radius of the disk.
        show_plot (bool): Whether to display the plot.
        save_path (str): Path to save the figure. If None, the figure won't be saved.
        
    Returns:
        tuple: A tuple containing x and y coordinates of the generated points.
    """
    points_polar = []
    for i in range(num_points):
        point = uniform_distributed_in_disk(radius)
        points_polar.append(point)
    
    points_cartesian = []
    for point in points_polar:
        r = point[0]
        theta = point[1]
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points_cartesian.append((x, y))
    
    x_coords = []
    y_coords = []
    for coord in points_cartesian:
        x_coords.append(coord[0])
        y_coords.append(coord[1])
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, s=10, alpha=0.6)
    plt.title(f"Uniform Distribution in a Disk (r={radius})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    circle = plt.Circle((0, 0), radius, fill=False, color='red', linestyle='--')
    plt.gca().add_patch(circle)
    
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return x_coords, y_coords


class SimulationBaselineModel:
    """
    A class to do the simulation of the ride-hailing system by polar coordinates.
    
    Attributes:
        lamada_rate (float): The arrival rate of riders.
        mu_rate (float): The arrival rate of drivers.
        radius_of_disk (float): Radius of the service area.
        warmup_period (float): Warm-up period during which statistics are not collected.
        period_length (float): Time length for each period.
        simula_duration (float): Total duration of the simulation.
        event_queue (list): Priority queue of events.
        rider_queue (deque): Queue of waiting riders.
        rider_times (list): Total time each rider spend.
        current_time (float): Current simulation time.
        is_warmed_up (bool): Whether is warm up or not.
        current_time (float): Currnt time.
        rider_arrival_times (list): Times when riders arrive.
        driver_arrival_times (list): Times when drivers arrive.
    """
    def __init__(self, lambda_rate, mu_rate, sim_duration=5000, warmup_period=500, period_length=100, radius=1):
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.sim_duration = sim_duration
        self.warmup_period = warmup_period
        self.period_length = period_length
        self.radius = radius
        
        self.event_queue = []
        self.rider_queue = deque()
        self.rider_times = []
        self.queue_length_history = []
        self.time_points = []

        self.current_time = 0
        self.rider_time = []
        self.total_riders = 0
        self.total_drivers = 0
        self.matched_drivers = 0
        
        self.is_warmed_up = False
        self.rider_arrival_times = []
        self.driver_arrival_times = []
        

    def run(self):
        """
        Run the simulation of the ride-hailing.
        """

        self._when_next_rider_arrival()
        self._when_next_driver_arrival()
        
        while self.event_queue and self.current_time < self.sim_duration:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if int(self.current_time) % 100 == 0:
                self.queue_length_history.append(len(self.rider_queue))
                self.time_points.append(self.current_time)

            if not self.is_warmed_up and self.current_time >= self.warmup_period:
                self.is_warmed_up = True
                self.total_riders = 0
                self.total_drivers = 0
                self.matched_drivers = 0
                self.rider_times = []
                self.rider_arrival_times = []
                self.driver_arrival_times = []
                
            if event.event_type == "rider_arrival":
                self._when_rider_arrival(event)
            elif event.event_type == "driver_arrival":
                self._when_driver_arrival(event)
                self._when_next_driver_arrival()
            elif event.event_type == "rider_departure":
                self._process_rider_departure(event)
        
        avg_rider_time = np.mean(self.rider_times) if self.rider_times else 0
        variance_rider_time = np.var(self.rider_times) if self.rider_times else 0
        driver_matching_rate = self.matched_drivers / self.total_drivers if self.total_drivers > 0 else 0

        return {
            "avg_rider_time": avg_rider_time,
            "variance_rider_time": variance_rider_time,
            "driver_matching_rate": driver_matching_rate,
            "total_riders": self.total_riders,
            "total_drivers": self.total_drivers,
            "matched_drivers": self.matched_drivers
        }


    def _when_rider_arrival(self, event):
        """
        Deal with the event of rider arrival.

        Parameters:
        event (RiderArrival): The event when the rider arrives.
        """
        rider = Rider(event.rider_id, event.time, event.origin, event.destination, event.period)
        if self.is_warmed_up:
            self.total_riders += 1
            self.rider_arrival_times.append(event.time)
        self.rider_queue.append(rider)
        self._when_next_rider_arrival()


    def _when_driver_arrival(self, event):
        """
        Deal with the event of driver arrival. Match the driver with the rider and follow FCFS 
        if the queue is not empty. Otherwise, the dirver will leave the system right away.

        Parameters:
            event (DriverArrival): The event when the driver arrives.
        """
        if self.is_warmed_up:
            self.total_drivers +=1
            self.driver_arrival_times.append(event.time)
        driver = Driver(event.driver_id, event.time, event.origin, event.period)
        
        if self.rider_queue:
            rider = self.rider_queue.popleft()
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


    def _process_rider_departure(self, event):
        """
        Process the rider departure event.
        
        Parameters:
            event (RiderDeparture): The rider departure event.
        """
        if self.is_warmed_up:
            self.rider_times.append(event.rider.total_time)


    def _when_next_rider_arrival(self):
        """
        Generate the next rider arrival event which follow the Poisson distribution.
        """
        interval_rider_time = np.random.exponential(1 / self.lambda_rate)
        event_time = self.current_time + interval_rider_time
        rider_id = str(uuid.uuid4())
        origin = uniform_distributed_in_disk(self.radius)
        destination = uniform_distributed_in_disk(self.radius)
        period = calculate_time_period(event_time, self.period_length)
        event = RiderArrival(event_time, rider_id, origin, destination, period)
        heapq.heappush(self.event_queue, event)


    def _when_next_driver_arrival(self):
        """
        Generate the next driver arrival event which follow the Poisson distribution.
        """
        interval_driver_time = np.random.exponential(1.0 / self.mu_rate)
        event_time = self.current_time + interval_driver_time
        driver_id = str(uuid.uuid4())
        origin = uniform_distributed_in_disk(self.radius)
        period = calculate_time_period(event_time, self.period_length)
        event = DriverArrival(event_time, origin, driver_id, period)
        heapq.heappush(self.event_queue, event)


    def calculate_metrics(self):
        """
        Calculate the metrics and results for the simulation module.
        """
        metrics_dic = {}
        metrics_dic["total_riders"] = self.total_riders
        metrics_dic["total_drivers"] = self.total_drivers
        metrics_dic["matched_drivers"] = self.matched_drivers
        if self.total_drivers > 0:
            metrics_dic["match_rate"] = self.matched_drivers / self.total_drivers
        else:
            metrics_dic["match_rate"] = 0
        # calculate the average time of the rider
        if self.rider_times:
            average = sum(self.rider_times) / len(self.rider_times)
            metrics_dic["average_rider_time"] = average
        # calculate the variance time of the rider
        variance = np.var(self.rider_times)
        metrics_dic["variance_rider_time"] = variance
        # calculate the matching rate of the driver
        if self.total_drivers > 0:
            matching_rate = self.matched_drivers / self.total_drivers
        else:
            matching_rate = 0
        metrics_dic["match_rate"] = matching_rate

    def plot_queue_length(self, filename="queue_length.png"):
        """
        Plot the queue length over time to visualize system stability.

        Parameters:
            filename: the name of the file
        """
        if not self.queue_length_history or not self.time_points:
            print("No queue length data available")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.queue_length_history, 'b-', linewidth=2)
        
        # Find data points after warmup period
        steady_state_points = []
        steady_state_times = []
        
        for i, time in enumerate(self.time_points):
            if time >= self.warmup_period:
                steady_state_points.append(self.queue_length_history[i])
                steady_state_times.append(time)
        
        # Calculate average queue length after warmup
        if steady_state_points:
            avg_queue_length = sum(steady_state_points) / len(steady_state_points)
            print(f"Average queue length after warm-up: {avg_queue_length:.2f}")
            
            # Add average line to the plot
            plt.axhline(y=avg_queue_length, color='g', linestyle='--')
            plt.text(self.time_points[-1] * 0.5, avg_queue_length + 0.5, 
                     f"Avg Queue Length: {avg_queue_length:.2f}", color='g')
        else:
            print("No data after warm-up period")
        
        # Add labels and save
        plt.title("Rider Queue Length Over Time", fontsize=14)
        plt.xlabel("Simulation Time", fontsize=12)
        plt.ylabel("Number of Riders in Queue", fontsize=12)
        plt.grid(True)
        plt.figtext(0.02, 0.02, 
                   f"λ = {self.lambda_rate:.2f}, μ = {self.mu_rate:.2f}, ρ = {self.lambda_rate/self.mu_rate:.2f}",
                   fontsize=10)
        plt.savefig(filename)
        plt.close()
        print(f"Queue length plot saved as '{filename}'")

    def plot_arrivals_histogram(self, bin_width = 200, filename = "arrivals_histogram.png"):
        """
        Plot a histogram showing the number of rider and driver arrivals over time.
        
        Parameters:
            bin_width (float): Width of each time bin
            filename (str): Name of the file
        """
        if not self.rider_arrival_times or not self.driver_arrival_times:
            print("No Data")
            return
        
        max_time = max(max(self.rider_arrival_times), max(self.driver_arrival_times))
        bins = np.arange(self.warmup_period, max_time + bin_width, bin_width)
        plt.figure(figsize=(12, 6))
        
        # rider arrivals
        rider_counts, rider_bins, _ = plt.hist(
            self.rider_arrival_times, 
            bins = bins, 
            alpha = 0.5, 
            label = 'Riders', 
            color = 'blue'
        )
        
        # driver arrivals
        driver_counts, driver_bins, _ = plt.hist(
            self.driver_arrival_times, 
            bins = bins, 
            alpha = 0.5, 
            label = 'Drivers', 
            color = 'green'
        )
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        plt.plot(bin_centers, rider_counts, 'b-', linewidth = 2, label='Rider Trend')
        plt.plot(bin_centers, driver_counts, 'g-', linewidth = 2, label='Driver Trend')
        plt.title(f"Arrivals Distribution Over Time (bin width = {bin_width})", fontsize = 14)
        plt.xlabel("Simulation Time", fontsize = 12)
        plt.ylabel("Number of Arrivals", fontsize = 12)
        plt.grid(True, alpha = 0.3)
        plt.legend()
        plt.figtext(0.02, 0.02, 
                   f"λ = {self.lambda_rate:.2f}, μ = {self.mu_rate:.2f}, ρ = {self.lambda_rate/self.mu_rate:.2f}",
                   fontsize=10)
        plt.savefig(filename)
        plt.close()
        print(f"Arrivals histogram saved as '{filename}'")

        return {
            "bin_centers": bin_centers,
            "rider_counts": rider_counts,
            "driver_counts": driver_counts
        }


def determine_warmup_period(lambda_rate, mu_rate, test_duration=2000):
    """
    Determine an appropriate warm-up period by checking when queue length stabilizes.
    
    Args:
        lambda_rate (float): How often riders arrive
        mu_rate (float): How often drivers arrive
        test_duration (float): How long to run the test
    
    Returns:
        float: The time when the system becomes stable
    """
    simulation = SimulationBaselineModel(
        lambda_rate=lambda_rate,
        mu_rate=mu_rate,
        sim_duration=test_duration,
        warmup_period=0
    )
    
    simulation.run()
    
    queue_lengths = simulation.queue_length_history
    time_points = simulation.time_points
    
    if len(queue_lengths) < 10:
        return test_duration * 0.1
    
    moving_avg = []
    window_size = 3
    
    for i in range(len(queue_lengths) - window_size + 1):
        window_avg = sum(queue_lengths[i:i+window_size]) / window_size
        moving_avg.append(window_avg)
    
    stability_threshold = 0.2
    
    for i in range(1, len(moving_avg)):
        if moving_avg[i-1] < 0.1:
            continue
            
        percent_change = abs(moving_avg[i] - moving_avg[i-1]) / moving_avg[i-1]
        
        if percent_change < stability_threshold:
            warmup_time = time_points[i + window_size - 2]
            return warmup_time
    
    return test_duration * 0.2


def grid_search(lambda_values, mu_ratios, sim_duration=5000):
    """
    Simple grid search to find the best combination of rider and driver arrival rates.

    Args:
        lambda_values (list): Different rider arrival rates to test
        mu_ratios (list): Different driver/rider ratios to test
        sim_duration (float): How long to run each test
    """
    best_params = None
    best_score = float('inf')
    all_results = []
    
    warmup_period = determine_warmup_period(lambda_values[0], lambda_values[0] * mu_ratios[0])
    
    for lambda_rate in lambda_values:
        for ratio in mu_ratios:
            mu_rate = lambda_rate * ratio
            
            sim = SimulationBaselineModel(
                lambda_rate=lambda_rate, 
                mu_rate=mu_rate,
                sim_duration=sim_duration,
                warmup_period=warmup_period
            )
            
            results = sim.run()
            
            avg_rider_time = results["avg_rider_time"]
            driver_match_rate = results["driver_matching_rate"]
            
            score = avg_rider_time - 2 * driver_match_rate
            
            result_entry = {
                "lambda_rate": lambda_rate,
                "mu_rate": mu_rate,
                "ratio": ratio,
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
                    "ratio": ratio
                }
    
    return best_params, all_results


def run_comparison_cases():
    """
    Run simulations with different arrival rates and collect results.
    
    This function tests multiple combinations where driver arrival rate (μ) is 
    always greater than or equal to passenger arrival rate (λ). This ensures 
    queue stability, which in the steady-state.
    """
    # pick values to see comparison cases
    lambda_values = [3.0, 4.0, 5.0, 6.0, 7.0]
    utilization_values = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    results = []
    warmup_period = 500
    
    print("Starting comparison...")
    for lambda_rate in lambda_values:
        for utilization in utilization_values:
            # ensures stability of the system so μ >= λ 
            mu_rate = lambda_rate / utilization
            
            print(f"Testing: λ={lambda_rate:.1f}, μ={mu_rate:.1f}, utilization={utilization:.2f}")
        
            sim = SimulationBaselineModel(
                lambda_rate = lambda_rate,
                mu_rate = mu_rate,
                sim_duration = 5000,
                warmup_period = warmup_period
            )
            
            results_dict = sim.run()
            results.append({
                "lambda_rate": lambda_rate,
                "mu_rate": mu_rate,
                "utilization": utilization,
                "avg_rider_time": results_dict["avg_rider_time"]  # This is total time which includes wait + pickup + trip
            })
    
    plot_performance_comparison(results)
    return results


def plot_performance_comparison(results):
    """
    Plot how average total time riders spend in the system changes with utilization.
    
    Parameter:
        results (list): A list with results
    """
    for result in results:
        if "utilization" not in result:
            result["utilization"] = result["lambda_rate"] / result["mu_rate"]
    results.sort(key=lambda r: r["utilization"])
    
    lambda_values = []
    for result in results:
        if result["lambda_rate"] not in lambda_values:
            lambda_values.append(result["lambda_rate"])
    lambda_values.sort()
    
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    for i, lambda_val in enumerate(lambda_values):
        x_points = []  # this is utilization values
        y_points = []  # this is average total times
        
        for result in results:
            if result["lambda_rate"] == lambda_val:
                x_points.append(result["utilization"])
                y_points.append(result["avg_rider_time"])
        
        color = colors[i % len(colors)]
        plt.plot(
            x_points, y_points,
            marker='o', 
            linestyle='-', 
            color=color,
            label=f'λ = {lambda_val}'
        )
    
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    plt.text(1.02, plt.ylim()[1] * 0.5, "Stability Threshold", color='red', rotation=90)
    
    plt.title("Average Total Time in System vs Utilization", fontsize=14)
    plt.xlabel("System Utilization (λ/μ)", fontsize=12)
    plt.ylabel("Average Total Time (wait + pickup + trip)", fontsize=12)
    plt.grid(True)
    plt.legend()
    
    plt.savefig("performance_comparison.png")
    plt.close()
    print("Plot saved as 'performance_comparison.png'")


def main():
    """
    Run the ride-sharing simulation to find the best setup.
    
    This program:
    1. Tests different rider and driver arrival rates
    2. Finds the best combination
    3. Shows the final results including:
       - How many riders and drivers we had
       - How many matches we made
       - How long riders waited (average and variance)
       - How often drivers found rides
    4. Creates a plot showing queue length over time to analyze system stability
    """
    # pick some values and ratios for ploting the visualization
    lambda_values = [3.0, 4.0, 5.0]
    mu_ratios = [0.8, 1.0, 1.2]
    
    print("Starting grid search to find optimal parameters:")
    best_params, all_results = grid_search(lambda_values, mu_ratios)

    print("\nBest parameters found:")
    print(f"Rider arrival rate (λ): {best_params['lambda_rate']:.3f}")
    print(f"Driver/Rider ratio: {best_params['ratio']:.3f}")
    print(f"Driver arrival rate (μ): {best_params['mu_rate']:.3f}")
    
    print("\nRunning final simulation with best parameters...")
    final_sim = SimulationBaselineModel(
        lambda_rate=best_params['lambda_rate'],
        mu_rate=best_params['mu_rate'],
        sim_duration=5000,
        warmup_period=500
    )
    
    results = final_sim.run()
    
    print("\nFinal simulation results:")
    print(f"Total riders: {results['total_riders']}")
    print(f"Total drivers: {results['total_drivers']}")
    print(f"Matched drivers: {results['matched_drivers']}")
    print(f"Average rider wait time: {results['avg_rider_time']:.3f}")
    print(f"Variance in rider wait time: {results['variance_rider_time']:.3f}")
    print(f"Driver matching rate: {results['driver_matching_rate']:.3f}")
    
    # Plot queue length with best parameters that found by grid search
    print("\nCreating queue length plot with best parameters...")
    final_sim.plot_queue_length("best_params_queue_length.png")
    
    # Plot histogram of driver and rider arrival
    print("Creating arrivals histogram...")
    final_sim.plot_arrivals_histogram(bin_width=200, filename="arrivals_histogram.png")
    
    run_comparison_cases()
    print("\nSimulation complete")


if __name__ == "__main__":
    generate_uniformly_points(num_points=1000, radius=1, save_path="uniform_points.png")
    main()