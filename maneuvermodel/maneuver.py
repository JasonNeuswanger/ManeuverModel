import numpy as np
from numba import jit, float64, int8, jitclass, optional
from .maneuveringfish import ManeuveringFish
from .path import ManeuverPath
from .dynamics import ManeuverDynamics
from .finalstraight import CONVERGENCE_FAILURE_COST

# Create numba types for the imported jitclasses, so I can use jitclasses as attributes of other jitclasses.
dynamics_type = ManeuverDynamics.class_type.instance_type
path_type = ManeuverPath.class_type.instance_type
fish_type = ManeuveringFish.class_type.instance_type

@jit(float64(float64, float64, float64), nopython=True) # Needs to go above Maneuver definition because it's used in there
def proportion_of_range(value, range_min, range_max):
    range_size = range_max - range_min
    if range_size > 0.0:
        proportional_value = (value - range_min) / range_size
        if 0.0 <= proportional_value <= 1.0:
            return proportional_value
        else:
            print("Value", value, "was not within its required range from", range_min, "to", range_max, ".")
            raise ValueError("Value not within requested range in proportion_of_range.")
    elif range_size == 0.0:
        return 0.0
    else:
        print("Asked for", value, "as a proportion of invalid range with min", range_min, "greater than max", range_max,".")
        raise ValueError("Asked for proportion of range with with range_min < range_max.")

maneuver_spec = [
    ('fish', fish_type),
    ('prey_velocity', float64),
    ('mean_water_velocity', float64),
    ('r1', float64),
    ('r2', float64),
    ('r3', float64),
    ('final_turn_x', float64),
    ('det_x', float64),                        # detection position x coord
    ('det_y', float64),                        # detection position y coord
    ('wait_time',float64),                     # time fish waits before beginning pursuit
    ('thrusts', float64[:]),                   # thrust values in m/s corresponding to asymptotic speed at a given thrust
    ('pthrusts', float64[:]),                  # proportional versions of the above corresponding to their place within their individual bounds
    ('final_thrust_a', float64),               # first thrust of the final straight, constrained differently than the rest
    ('final_pthrust_a', float64),              # proportional version of the first thrust of the final straight
    ('final_duration_a_proportional', float64),
    ('fitness', float64),                     
    ('duration', float64),  
    ('pursuit_duration', optional(float64)), # not used internally, just in the final summary for reference elsewhere
    ('energy_cost', optional(float64)),      # not used internally, just in the final summary for reference elsewhere
    ('mean_swimming_speed', optional(float64)),  
    ('mean_swimming_speed_bodylengths', optional(float64)),  
    ('swimming_cost_per_second', optional(float64)),  
    ('mean_metabolic_rate', optional(float64)),   
    ('mean_metabolic_rate_SMRs', optional(float64)),       
    ('origin', int8), # 1 for random, 2 for crossover (mating), 3 for mutant
    ('dynamics',optional(dynamics_type)),
    ('path',optional(path_type)),
    ('matrix_3Dfrom2D',optional(float64[:,:])),  # these two values are both used to convert results back into
    ('ysign',optional(float64))                  # 3-D from the positive side of the 2-D x-y plane in which calculations are done
]

@jitclass(maneuver_spec)
class Maneuver(object):
    
    def __init__(self, fish, prey_velocity, r1, r2, r3, final_turn_x, pthrusts, wait_time, det_x, det_y, ftap, fdap, origin):
        self.fish = fish
        self.prey_velocity = prey_velocity
        self.mean_water_velocity = (prey_velocity + fish.focal_velocity) / 2.0
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.final_turn_x = final_turn_x
        self.det_x = det_x
        self.det_y = det_y
        self.origin = origin
        self.wait_time = wait_time
        self.pthrusts = pthrusts
        self.final_pthrust_a = ftap
        self.thrusts = np.zeros(pthrusts.size, dtype=float64)     # just a placeholder of the right type
        self.final_thrust_a = ftap                                # just a placeholder of the right type
        self.final_duration_a_proportional = fdap
        self.dynamics = None
        self.path = None
        self.calculate_fitness()  # compute the path and dynamics

    def print_inputs(self):
        print("INPUTS BELOW: ")
        print("Prey velocity: ", self.prey_velocity)
        print("r1, r2, r3: ", self.r1, ", ", self.r2, ", ", self.r3)
        print("final_turn_x: ", self.final_turn_x)
        print("pthrusts: ")
        for pthrust in self.pthrusts:
            print(pthrust)
        print("final_pthrust_a: ", self.final_pthrust_a)
        print("final_duration_a_proportional: ", self.final_duration_a_proportional)
        print("det_x, det_y: ", self.det_x, ", ", self.det_y)
        print("wait_time: ", self.wait_time)

    def to_3D(self, value):
        '''Converts a model result expressed as a 2-vector in the positive-y side of the x-y plane back into 3-D coordinates on the original side of the fish'''
        if (self.matrix_3Dfrom2D is None or self.ysign is None):
            return None
        else:
            value_3vec = np.array([value[0], value[1], 0.0])
            result = np.dot(self.matrix_3Dfrom2D, value_3vec)  # todo this line generates a silly, irrelevant numba performance warning
            result[1] *= self.ysign
            return result
            
    @property
    def capture_point_3D_groundcoords(self):
        if (self.matrix_3Dfrom2D is None or self.ysign is None):
            return None
        else:
            result = self.to_3D(self.path.capture_point)                                # In water coordinates
            result[0] += self.dynamics.pursuit_duration * self.mean_water_velocity      # Converted into ground coordinates
            return result        
                    
    def calculate_fitness(self):
        self.fish.fEvals += 1
        self.path = ManeuverPath(self)
        # Most path constraints are handled in validate_for_fish() above, but compatibility of r2 and r3 depends on 
        # the distance between points J and K which isn't know until the path is partly calculated. So its initialization
        # function checks that and returns prematurely with an attribute indicating failure if they're incompatible.
        # Most failures are from r3 being too big, so we first try to shrink it, or else move turn 3 farther left.
        while not self.path.creation_succeeded: # first try shrinking r3
            shrunk_r3 = 0.8 * self.r3
            if shrunk_r3 >= self.fish.min_turn_radius:
                self.r3 = shrunk_r3
                try:
                    self.path = ManeuverPath(self)
                except Exception:
                    print("Exception creating path in maneuver.py calculate_fitness() while shrinking r3 to build path.")
            else: # if we shrunk r3 as far as it can go and it's still too big, start moving circle 3 to the left
                self.final_turn_x -= self.fish.min_turn_radius
                try:
                    self.path = ManeuverPath(self)
                except Exception:
                    print("Exception creating path in maneuver.py calculate_fitness() while moving circle 3 to the left to build path.")

        # Once a valid path has been created, calculate its dynamics
        try:
            self.dynamics = ManeuverDynamics(self)
        except Exception:
            print("Exception creating dynamics in maneuver.py calculate_fitness().")
        # The "fitness" for the optimization algorithm is the negative cost, so maximizing fitness = minimizing cost
        self.fitness = -self.dynamics.total_cost if self.fish.use_total_cost else -self.dynamics.energy_cost
        self.fitness = self.fitness * (1 + self.dynamics.bad_thrust_b_penalty)
        
    def calculate_summary_metrics(self):
        """ Calculates attributes of the solution that only need to be requested if it was the optimal solution to a given maneuver, not for each intermediate solution. Separated from
            calculate_fitness for computational efficiency, since these are usually only needed for the final optimal solution."""
        oq = 14.05834 # oxycalorific equivalent
        self.duration = self.dynamics.duration                       # saving duration for convenient input to plotting functions later
        self.pursuit_duration = self.dynamics.pursuit_duration
        self.energy_cost = self.dynamics.energy_cost
        if self.energy_cost != CONVERGENCE_FAILURE_COST:
            self.mean_swimming_speed = self.path.total_length / self.dynamics.moving_duration
            self.mean_swimming_speed_bodylengths = self.mean_swimming_speed / self.fish.total_length
            self.swimming_cost_per_second = self.dynamics.energy_cost / self.dynamics.moving_duration
            self.mean_metabolic_rate = self.swimming_cost_per_second / ((1/3600.0) * (self.fish.base_mass/1000.0) * oq) # convert from J/s back to mg O2/kg/hr to get mean metabolic rate while moving on the maneuver
            self.mean_metabolic_rate_SMRs = 1 + self.mean_metabolic_rate / self.fish.SMR if self.fish.SMR > 0 else np.nan # in case zero is passed as SMR for calculations limited to locomotion costs

    def predicted_capture_point_3D_groundcoords(self):
        prey_speed = self.mean_water_velocity # remember, it's passed the focal + prey speed average average as water velocity
        (turn_1, straight_1, turn_2, straight_2, turn_3, straight_3) = self.dynamics.plottable_segments(self)
        gc_capture_point = [0.0, 0.0]
        gc_capture_point[0] = self.path.capture_point[0] + prey_speed * (turn_1.duration + straight_1.duration) # convert to ground coords
        gc_capture_point[1] = self.path.capture_point[1] # copying element values 1 by 1 to avoid modifying the path
        return self.to_3D(gc_capture_point)
        
    def proportions(self): # Returns a representation of solution's current state as a numpy vector of proportions of the allowed range
        p = np.empty(12, dtype=float64)
        # Set wait time within bounds, if setting it at all
        if not (self.fish.disable_wait_time or self.det_x > 0): # make this the last element of the solution vector, absent if not needed
            max_x_for_wait = 0.0 if self.det_y > 2 * self.fish.min_turn_radius else -np.sqrt(2*self.fish.min_turn_radius*self.det_y - self.det_y**2)
            max_wait_time = (max_x_for_wait - self.det_x) / self.mean_water_velocity if self.det_x < max_x_for_wait else 0
        else:
            max_wait_time = 0
        p[11] = proportion_of_range(self.wait_time, 0, max_wait_time)
        # Set the capture point, given the wait time
        yc = self.det_y
        xc = self.det_x + self.mean_water_velocity * self.wait_time
        # Set all the thrusts
        for i in range(5): 
            p[i] = self.pthrusts[i]
        # Set turn radii
        max_turn_radius = 15 * self.fish.min_turn_radius # ARBITRARY GUESS, CONSTRAIN BASED ON SOLUTIONS
        max_r1 = min(0.5*(xc**2/yc + yc) - 0.01, max_turn_radius) # subtract 0.01 cm (0.1 mm) so pursuit turn cannot encompass the capture point and cause divide-by-zero errors
        min_r1 = min(self.fish.min_turn_radius, max_r1)           # allow r1 to shrink beyond the min turn radius if necessary to not collide with the capture point
        p[5] = proportion_of_range(self.r1, min_r1, max_r1)
        p[6] = proportion_of_range(self.r2, self.fish.min_turn_radius, max_turn_radius)
        p[7] = proportion_of_range(self.r3, self.fish.min_turn_radius, max_turn_radius)
        # Set characteristics of the final straight to catch up to the focal point / velocity
        min_final_turn_x = xc - 2 * (self.r2 + self.r3 + 2 * self.fish.fork_length) # need to encompass any value to which this might be pushed to the left to guarantee convergence
        p[8] = self.final_pthrust_a
        p[9] = proportion_of_range(self.final_turn_x, min_final_turn_x, xc + 3 * self.fish.fork_length)
        p[10] = self.final_duration_a_proportional
        return p
    
maneuver_type = Maneuver.class_type.instance_type

@jit(float64(float64, float64, float64, float64), nopython=True)
def value_from_proportion(p, min_value, max_value, min_weight):
    # Pass a nonzero min_weight (between 0 and 1) to represent a proportion of time that values should be forced to 0.
    # For example, pass min_weight of 0.3 for wait_time to make sure solutions consider no wait time 30 % of the time.
    if min_value < max_value:
        if min_weight == 0.0:
            return min_value + p * (max_value - min_value)
        else:
            if p > min_weight: # If using min_weight, rescale remaining proportion to fill the rest of the range
                return min_value + ((p - min_weight) / (1 - min_weight)) * (max_value - min_value)
            else:
                return min_value
    elif min_value == max_value:
        return min_value
    else:
        print("Min value was",min_value,"and max value was",max_value)
        raise ValueError('In solution.py, value_from_proportion got min_value > max_value.')

@jit(maneuver_type(fish_type, float64, float64, float64, float64[:]), nopython=True)
def maneuver_from_proportions(fish, prey_velocity, xd, yd, p):
    """ Create a valid solution from just a numpy vector p of [0, 1) of values in proportion to their allowed range. This should make it easier to 
        test out different optimization algorithms. There are a total of 18 parameters per maneuver."""
    no_min_weight = 0.0
    mean_velocity = (prey_velocity + fish.focal_velocity) / 2
    # Set wait time within bounds, if setting it at all
    if not (fish.disable_wait_time or xd > 0): # make this the last element of the solution vector, absent if not needed
        max_x_for_wait = 0.0 if yd > 2 * fish.min_turn_radius else -np.sqrt(2*fish.min_turn_radius*yd - yd**2)
        max_wait_time = (max_x_for_wait - xd) / mean_velocity if xd < max_x_for_wait else 0
        wait_time = value_from_proportion(p[11], 0, max_wait_time, 0.1)
    else:
        wait_time = 0
    # Set the capture point, given the wait time
    yc = yd
    xc = xd + mean_velocity * wait_time
    # Set all the proportional thrusts
    pthrusts = p[:6]
    # Set turn radii
    max_turn_radius = 15 * fish.min_turn_radius # ARBITRARY GUESS, CONSTRAIN BASED ON SOLUTIONS
    max_r1 = min(0.5 * (xc ** 2 / yc + yc) - 0.01, max_turn_radius)  # subtract 0.01 cm (0.1 mm) so pursuit turn cannot encompass the capture point and cause divide-by-zero errors
    min_r1 = min(fish.min_turn_radius, max_r1)  # allow r1 to shrink beyond the min turn radius if necessary to not collide with the capture point
    r1 = value_from_proportion(p[5], min_r1, max_r1, no_min_weight) # use max_r1 as r1 for head-snap maneuvers
    r2 = value_from_proportion(p[6], fish.min_turn_radius, max_turn_radius, no_min_weight)
    r3 = value_from_proportion(p[7], fish.min_turn_radius, max_turn_radius, no_min_weight)
    # Set characteristics of the final straight to catch up to the focal point / velocity
    final_pthrust_a = p[8] # must be higher than the water velocity to catch up to the focal point (by at least 2 % to improve convergence)
    min_final_turn_x = xc - 2 * (r2 + r3 + 2 * fish.fork_length) # need to encompass any value to which this might be pushed to the left to guarantee convergence
    final_turn_x = value_from_proportion(p[9], min_final_turn_x, xc + 3 * fish.fork_length, no_min_weight) # ARBITRARY GUESS, MIGHT CONSTRAIN MORE BASED ON SOLUTIONS
    final_duration_a_proportional = p[10]
    # Creation solution object and return
    return Maneuver(fish, prey_velocity, r1, r2, r3, final_turn_x, pthrusts, wait_time, xd, yd, final_pthrust_a, final_duration_a_proportional, 1)

@jit(maneuver_type(fish_type, float64, float64, float64), nopython=True)
def random_maneuver(fish, prey_velocity, xd, yd):
    p = np.random.rand(12)
    return maneuver_from_proportions(fish, prey_velocity, xd, yd, p)
    
@jit(maneuver_type(fish_type, float64, float64, float64), nopython=True)
def convergent_random_maneuver(fish, prey_velocity, xd, yd): # quick helper to get a random solution that converges
    maneuver_converged = False
    while not maneuver_converged:
        maneuver = random_maneuver(fish, prey_velocity, xd, yd)
        maneuver_converged = maneuver.dynamics.energy_cost < CONVERGENCE_FAILURE_COST
    return maneuver

@jit(maneuver_type(fish_type, float64), nopython=True)
def test_maneuver(fish, prey_velocity): # Returns the same test solution every time, for repeatable analysis
    mean_velocity = (fish.focal_velocity + prey_velocity) / 2
    r1 = 1.5 * fish.min_turn_radius
    xd = -2 * r1
    yd = 2 * r1
    r2 = 1.5 * r1
    r3 = 1.25 * r1
    final_turn_x = -4 * r1
    thrusts = 1.2 * mean_velocity * np.ones(5)
    wait_time = 0
    final_thrust_a = 1.5 * mean_velocity
    final_duration_a_proportional = 0.5
    return Maneuver(fish, prey_velocity, r1, r2, r3, final_turn_x, thrusts, wait_time, xd, yd, final_thrust_a, final_duration_a_proportional, 1)
