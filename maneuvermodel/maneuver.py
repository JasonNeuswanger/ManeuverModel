import numpy as np
from numba import jit, float64, uint64, optional, boolean, types
from numba.experimental import jitclass
from .maneuveringfish import ManeuveringFish
from .path import ManeuverPath
from .dynamics import ManeuverDynamics
from .constants import *

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

@jit(float64(float64, float64, float64), nopython=True)
def value_from_proportion(p, min_value, max_value):
    if min_value < max_value:
        return min_value + p * (max_value - min_value)
    elif min_value == max_value:
        return min_value
    else:
        print("Min value was", min_value, "and max value was", max_value)
        raise ValueError('In solution.py, value_from_proportion got min_value > max_value.')

maneuver_spec = [
    ('fish', fish_type),
    ('mean_water_velocity', float64),
    ('r1', float64),
    ('r2', float64),
    ('r3', float64),
    ('r3_proportional', float64),
    ('max_r3', float64),                       # for reconstructing proportion of range
    ('final_turn_x', float64),
    ('det_x', float64),                        # detection position x coord
    ('det_y', float64),                        # detection position y coord
    ('wait_time',float64),                     # time fish waits before beginning pursuit
    ('thrusts', float64[:]),                   # thrust values in m/s corresponding to asymptotic speed at a given thrust
    ('pthrusts', float64[:]),                  # proportional versions of the above corresponding to their place within their individual bounds
    ('final_thrust_a', float64),               # first thrust of the final straight, constrained differently than the rest
    ('final_pthrust_a', float64),              # proportional version of the first thrust of the final straight
    ('fitness', float64),                      # internal metric to be maximized to find the optimal maneuver, often just -1*energy_cost
    ('had_final_turn_adjusted', boolean),      # tracks whether this maneuver had final_turn_x adjusted to avoid overshooting the focal point
    ('duration', float64),  
    ('pursuit_duration', optional(float64)), # not used internally, just in the final summary for reference elsewhere
    ('return_duration', optional(float64)),  # not used internally, just in the final summary for reference elsewhere
    ('activity_cost', optional(float64)),      # not used internally, just in the final summary for reference elsewhere
    ('energy_cost_including_SMR', optional(float64)), # not used internally, just in the final summary for reference elsewhere
    ('capture_x', optional(float64)),        # not used internally, just in the final summary for reference elsewhere
    ('mean_swimming_speed', optional(float64)),  
    ('mean_swimming_speed_bodylengths', optional(float64)),  
    ('swimming_cost_per_second', optional(float64)),  
    ('mean_metabolic_rate', optional(float64)),   
    ('mean_metabolic_rate_SMRs', optional(float64)),
    ('convergence_failure_code', types.string),
    ('dynamics',optional(dynamics_type)),
    ('path',optional(path_type)),
    ('matrix_3Dfrom2D',optional(float64[:,::1])),  # these two values are both used to convert results back into
    ('ysign',optional(float64)),                   # 3-D from the positive side of the 2-D x-y plane in which calculations are done
    ('objective_function_evaluations', optional(uint64))
]

@jitclass(maneuver_spec)
class Maneuver(object):
    
    def __init__(self, fish, r1, r2, r3_proportional, final_turn_x, pthrusts, wait_time, det_x, det_y, ftap):
        self.fish = fish
        self.mean_water_velocity = fish.mean_water_velocity
        self.r1 = r1
        self.r2 = r2
        self.r3_proportional = r3_proportional
        self.r3 = 0 # placeholder, will be filled in when calculating path
        self.max_r3 = 0 # placeholder, same reasoning as above
        self.final_turn_x = final_turn_x
        self.det_x = det_x
        self.det_y = det_y
        self.wait_time = wait_time
        self.pthrusts = pthrusts
        self.final_pthrust_a = ftap
        self.thrusts = np.zeros(pthrusts.size, dtype=float64)     # just a placeholder of the right type
        self.final_thrust_a = ftap  # also just a placeholder of the right type; real value set in dynamics.build_segments()
        self.had_final_turn_adjusted = False
        self.dynamics = None
        self.path = None
        self.convergence_failure_code = ''
        self.calculate_fitness()  # compute the path and dynamics

    def print_inputs(self):
        print("INPUTS BELOW: ")
        print("r1, r2, r3_proportional: ", self.r1, ", ", self.r2, ", ", self.r3_proportional)
        print("final_turn_x: ", self.final_turn_x)
        print("pthrusts: ")
        for pthrust in self.pthrusts:
            print(pthrust)
        print("final_pthrust_a: ", self.final_pthrust_a)
        print("det_x, det_y: ", self.det_x, ", ", self.det_y)
        print("wait_time: ", self.wait_time)

    def to_3D(self, value):
        """ Converts a model result expressed as a 2-vector in the positive-y side of the x-y plane back into 3-D coordinates on the original side of the fish """
        if (self.matrix_3Dfrom2D is None or self.ysign is None):
            return None
        else:
            value_3vec = np.array([value[0], value[1], 0.0])
            result = np.dot(self.matrix_3Dfrom2D, value_3vec)
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
        self.path = ManeuverPath(self)
        self.dynamics = ManeuverDynamics(self)
        while self.dynamics.needs_to_back_up_by > 0.0:  # move the final turn downstream to allow room to converge on the focal point from below
            self.final_turn_x += self.dynamics.needs_to_back_up_by  # back up enough to be behind the focal point approximately
            # if self.final_turn_x > self.capture_x + MAX_FINAL_TURN_X_LENGTH_MULTIPLE * self.fish.fork_length:
            #     print("backing up by", self.dynamics.needs_to_back_up_by,"put final_turn_x", self.final_turn_x,"> boundary",self.capture_x + MAX_FINAL_TURN_X_LENGTH_MULTIPLE * self.fish.fork_length)
            self.path = ManeuverPath(self)
            self.dynamics = ManeuverDynamics(self)
            self.had_final_turn_adjusted = True
        # The "fitness" for the optimization algorithm is the negative cost, so maximizing fitness = minimizing cost
        self.fitness = -self.dynamics.total_cost if self.fish.use_total_cost else -self.dynamics.activity_cost
        self.duration = self.dynamics.duration                       # saving duration for convenient input to plotting functions later
        self.pursuit_duration = self.dynamics.pursuit_duration
        self.return_duration = self.dynamics.return_duration
        self.activity_cost = self.dynamics.activity_cost
        self.energy_cost_including_SMR = self.dynamics.activity_cost + self.fish.SMR_J_per_s * self.duration
        self.capture_x = self.path.capture_point[0]

    def calculate_summary_metrics(self):
        """ Calculates attributes of the solution that only need to be requested if it was the optimal solution to a given maneuver, not for each intermediate solution. Separated from
            calculate_fitness for computational efficiency, since these are usually only needed for the final optimal solution."""
        oq = 14.05834 # oxycalorific equivalent
        if self.activity_cost != CONVERGENCE_FAILURE_COST:
            self.mean_swimming_speed = self.path.total_length / self.dynamics.moving_duration
            self.mean_swimming_speed_bodylengths = self.mean_swimming_speed / self.fish.total_length
            self.swimming_cost_per_second = self.dynamics.activity_cost / self.dynamics.moving_duration
            self.mean_metabolic_rate = self.fish.SMR + self.swimming_cost_per_second / ((1/3600.0) * (self.fish.base_mass/1000.0) * oq) # convert from J/s back to mg O2/kg/hr to get mean metabolic rate while moving on the maneuver
            self.mean_metabolic_rate_SMRs = 1 + self.mean_metabolic_rate / self.fish.SMR if self.fish.SMR > 0 else np.nan # in case zero is passed as SMR for calculations limited to locomotion costs

    def predicted_capture_point_3D_groundcoords(self):
        prey_speed = self.mean_water_velocity # remember, it's passed the focal + prey speed average average as water velocity
        (turn_1, straight_1, turn_2, straight_2, turn_3, straight_3) = self.dynamics.plottable_segments(self)
        gc_capture_point = [0.0, 0.0]
        gc_capture_point[0] = self.path.capture_point[0] + prey_speed * (turn_1.duration + straight_1.duration) # convert to ground coords
        gc_capture_point[1] = self.path.capture_point[1] # copying element values 1 by 1 to avoid modifying the path
        return self.to_3D(gc_capture_point)
        
    def proportions(self): # Returns a representation of solution's current state as a numpy vector of proportions of the allowed range
        p = np.empty(11, dtype=float64)
        # Set wait time within bounds, if setting it at all
        if not (self.fish.disable_wait_time or self.det_x > 0): # make this the last element of the solution vector, absent if not needed
            max_x_for_wait = 0.0 if self.det_y > 2 * self.fish.min_turn_radius else -np.sqrt(2*self.fish.min_turn_radius*self.det_y - self.det_y**2)
            max_wait_time = (max_x_for_wait - self.det_x) / self.mean_water_velocity if self.det_x < max_x_for_wait else 0
        else:
            max_wait_time = 0
        p[10] = proportion_of_range(self.wait_time, 0, max_wait_time)
        # Set the capture point, given the wait time
        yc = self.det_y
        xc = self.det_x + self.mean_water_velocity * self.wait_time
        # Set all the thrusts
        for i in range(5):
            p[i] = self.pthrusts[i]
        # Set turn radii
        max_turn_radius = MAX_TURN_RADIUS_MULTIPLE * self.fish.min_turn_radius # arbitrary maximum to constrain the solution space
        max_r1 = min(0.5*(xc**2/yc + yc) - 0.0001, max_turn_radius)  # subtract 0.0001 cm (0.001 mm) so pursuit turn cannot encompass the capture point and cause divide-by-zero errors
        min_r1 = min(self.fish.min_turn_radius, max_r1)           # allow r1 to shrink beyond the min turn radius if necessary to not collide with the capture point
        p[5] = proportion_of_range(self.r1, min_r1, max_r1)
        p[6] = proportion_of_range(self.r2, self.fish.min_turn_radius, max_turn_radius)
        p[7] = proportion_of_range(self.r3, self.fish.min_turn_radius, self.max_r3)
        # Set characteristics of the final straight to catch up to the focal point / velocity
        p[8] = self.final_pthrust_a
        min_final_turn_x = xc - MAX_FINAL_TURN_X_LENGTH_MULTIPLE * self.fish.fork_length
        max_final_turn_x = max(MAX_FINAL_TURN_X_LENGTH_MULTIPLE * self.fish.fork_length,
                               xc + MAX_FINAL_TURN_X_LENGTH_MULTIPLE * self.fish.fork_length) + self.r1 + 2*self.r2
        p[9] = proportion_of_range(self.final_turn_x, min_final_turn_x, max_final_turn_x)
        return p
    
maneuver_type = Maneuver.class_type.instance_type

@jit(maneuver_type(fish_type, float64, float64, float64[:]), nopython=True)
def maneuver_from_proportions(fish, xd, yd, p):
    """ Create a valid solution from just a numpy vector p of [0, 1) of values in proportion to their allowed range. Allowing the solutions being
    explored when optimizing a maneuver to be expressed as proportions allows for easier testing of different optimization algorithms. """
    # Set wait time within bounds, if setting it at all
    if not (fish.disable_wait_time or xd > 0): # make this the last element of the solution vector, absent if not needed
        max_x_for_wait = 0.0 if yd > 2 * fish.min_turn_radius else -np.sqrt(2*fish.min_turn_radius*yd - yd**2)
        max_wait_time = (max_x_for_wait - xd) / fish.mean_water_velocity if xd < max_x_for_wait else 0
        wait_time = value_from_proportion(p[10], 0, max_wait_time)
    else:
        wait_time = 0
    # Set the capture point, given the wait time
    yc = yd
    xc = xd + fish.mean_water_velocity * wait_time
    # Set all the proportional thrusts
    pthrusts = p[:5]
    # Set turn radii
    max_turn_radius = MAX_TURN_RADIUS_MULTIPLE * fish.min_turn_radius
    max_r1 = min(0.5 * (xc ** 2 / yc + yc) - 0.0001, max_turn_radius)  # subtract 0.0001 cm (0.001 mm) so pursuit turn cannot encompass the capture point and cause divide-by-zero errors
    min_r1 = min(fish.min_turn_radius, max_r1)  # allow r1 to shrink beyond the min turn radius if necessary to not collide with the capture point
    r1 = value_from_proportion(p[5], min_r1, max_r1) # use max_r1 as r1 for head-snap maneuvers
    r2 = value_from_proportion(p[6], fish.min_turn_radius, max_turn_radius)
    r3_proportional = p[7] # = value_from_proportion(p[7], fish.min_turn_radius, max_turn_radius)
    # Set characteristics of the final straight to catch up to the focal point / velocity
    final_pthrust_a = p[8] # must be higher than the water velocity to catch up to the focal point (by at least 2 % to improve convergence)
    # maximum value of final_turn_x is governed by an arbitrary multiple of fish.fork_length, either to the -x side of
    # the capture point or to the +x side of whichever is greater, the capture point or the origin
    min_final_turn_x = xc - MAX_FINAL_TURN_X_LENGTH_MULTIPLE * fish.fork_length
    max_final_turn_x = max(MAX_FINAL_TURN_X_LENGTH_MULTIPLE * fish.fork_length, xc + MAX_FINAL_TURN_X_LENGTH_MULTIPLE * fish.fork_length) + r1 + 2*r2
    final_turn_x = value_from_proportion(p[9], min_final_turn_x, max_final_turn_x)
    # Creation solution object and return
    return Maneuver(fish, r1, r2, r3_proportional, final_turn_x, pthrusts, wait_time, xd, yd, final_pthrust_a)

@jit(maneuver_type(fish_type, float64, float64), nopython=True)
def random_maneuver(fish, xd, yd):
    p = np.random.rand(11)
    return maneuver_from_proportions(fish, xd, yd, p)
    
@jit(maneuver_type(fish_type, float64, float64), nopython=True)
def convergent_random_maneuver(fish, xd, yd): # quick helper to get a random solution that converges
    maneuver_converged = False
    while not maneuver_converged:
        maneuver = random_maneuver(fish, xd, yd)
        maneuver_converged = maneuver.dynamics.activity_cost < CONVERGENCE_FAILURE_COST
    return maneuver
