import numpy as np
from numba import float64
from numba.experimental import jitclass
from .segment import ManeuverSegment, Cd
from .finalstraight import ManeuverFinalStraight, thrust_a_adjusted_for_tailbeats
from .constants import *

segment_type = ManeuverSegment.class_type.instance_type
final_straight_type = ManeuverFinalStraight.class_type.instance_type

maneuver_dynamics_spec = [
    ('penultimate_duration', float64),
    ('wait_duration', float64),
    ('pursuit_duration', float64),
    ('return_duration', float64),
    ('moving_duration', float64),
    ('duration', float64),
    ('mean_speed', float64),
    ('activity_cost', float64),
    ('activity_cost_pursuit', float64),
    ('activity_cost_return', float64),
    ('opportunity_cost', float64),
    ('total_cost', float64),
    ('turn_1', segment_type),
    ('turn_2', segment_type),
    ('turn_3', segment_type),
    ('straight_1', segment_type),
    ('straight_2', segment_type),
    ('straight_3', final_straight_type),
    ('fish_mass', float64),
    ('fish_rho', float64),
    ('fish_webb_factor', float64),
    ('fish_area', float64),
    ('fish_volume', float64),
    ('fish_waterlambda', float64),
    ('fish_total_length', float64),
    ('fish_min_thrust', float64),
    ('fish_max_thrust', float64),
    ('needs_to_back_up_by', float64),
    ('v', float64) # shortcut to maneuver.mean_water_velocity
]

@jitclass(maneuver_dynamics_spec)
class ManeuverDynamics(object):

    def __init__(self, maneuver):
        # Relevant attributes of the fish attribute are duplicated to avoid circular dependency among jitclasses
        fish = maneuver.fish
        self.fish_mass = fish.mass
        self.fish_rho = fish.rho
        self.fish_webb_factor = fish.webb_factor
        self.fish_area = fish.area
        self.fish_total_length = fish.total_length
        self.fish_waterlambda = fish.waterlambda
        self.fish_volume = fish.volume
        self.fish_min_thrust = fish.min_thrust
        self.fish_max_thrust = fish.max_thrust
        self.v = maneuver.mean_water_velocity
        # Create the initial maneuver segments and calculate their dynamics
        self.needs_to_back_up_by = 0.0  # Gets set to some number > 0 if the fish needs to slow down to let the focal point get in front of it by the end of turn 3
        try:
            self.build_segments(maneuver)    # This function will set the above flags to slow down or speed up if needed
        except Exception:
            print("Exception caught when building maneuver segments as initially called from ManeuverDynamics.__init__.")
        self.needs_to_back_up_by = self.check_if_needs_to_back_up(maneuver)
        if self.needs_to_back_up_by > 0:
            return
        # Now that the end of turn 3 is somewhere downstream of the focal point, calculate the final straight to catch up to it
        try:
            x_p, t_p = self.penultimate_point(maneuver)
            self.straight_3 = ManeuverFinalStraight(fish, t_p, x_p, self.turn_3.final_speed, maneuver.final_thrust_a, False)
            if not self.straight_3.creation_succeeded:
                maneuver.convergence_failure_code = self.straight_3.convergence_failure_code
                self.activity_cost = CONVERGENCE_FAILURE_COST
                self.total_cost = CONVERGENCE_FAILURE_COST
                return
        except Exception:
            print("Exception caught when creating ManeuverFinalStraight in ManeuverDynamics.__init__.")
        maneuver.path.update_with_straight_3_length(self.straight_3.length)
        # The commented-out line below was a condition specifically to prevent bad figure-8s, but it is covered by single the condition below
        # if maneuver.path.turn_2_length > np.pi * maneuver.r2 and maneuver.path.turn_3_length > np.pi * maneuver.r3 and maneuver.path.turn_2_direction != maneuver.path.turn_3_direction:

        # Prevent rare suboptimal solutions that can happen if the third turn is allowed to be more than 180 degrees. It is not
        # intended to ever be that large, but solutions where it is very large can end up being powerful local optima that trap
        # the optimization in solutions with 3-5X the energy cost of the true optimum. These include solutions that involve
        # doing an unnecessary circular loop backward to capture prey or doing an unnecessary figure-8 (in water coordinates).
        if maneuver.path.turn_3_length > np.pi*maneuver.r3:
            maneuver.convergence_failure_code = '8' # avoided return turn being more than 180 degrees
            self.activity_cost = CONVERGENCE_FAILURE_COST
            self.total_cost = CONVERGENCE_FAILURE_COST
            return
        # Save final attributes
        self.wait_duration = maneuver.path.wait_length / maneuver.mean_water_velocity
        self.pursuit_duration = self.turn_1.duration + self.straight_1.duration
        self.return_duration = self.turn_2.duration + self.straight_2.duration + self.turn_3.duration + self.straight_3.duration
        self.moving_duration = self.pursuit_duration + self.return_duration
        self.duration = self.wait_duration + self.pursuit_duration + self.return_duration # duration of the full maneuver in seconds
        self.mean_speed = maneuver.path.total_length / self.moving_duration
        self.activity_cost = self.turn_1.cost + self.straight_1.cost + self.turn_2.cost + self.straight_2.cost + self.turn_3.cost + self.straight_3.cost # energy cost of the full maneuver in Joules
        self.activity_cost_pursuit = self.turn_1.cost + self.straight_1.cost
        self.activity_cost_return = self.activity_cost - self.activity_cost_pursuit
        self.opportunity_cost = fish.NREI * (self.wait_duration + self.pursuit_duration)                                 # cost of not searching during the wait/pursuit
        self.total_cost = self.activity_cost + self.opportunity_cost

    def next_thrust_from_proportion(self, proportional_next_thrust, is_not_final_a):
        min_next_thrust = self.fish_min_thrust if is_not_final_a else 1.02 * self.v
        max_next_thrust = self.fish_max_thrust
        next_thrust = min_next_thrust + (max_next_thrust - min_next_thrust) * proportional_next_thrust
        return next_thrust

    def build_segments(self, maneuver):
        """ Here we create each segment, starting with the proportional representations of the thrusts, creating each actual thrust (m/s) value
            based on the speed at the end of the preceding segment, then plugging it the resulting
            actual thrust (based on asymptotic speed in m/s) in to create the next segment."""

        maneuver.thrusts[0] = self.next_thrust_from_proportion(maneuver.pthrusts[0], True)
        self.turn_1 = ManeuverSegment(maneuver.fish, maneuver.path.turn_1_length, self.v, maneuver.thrusts[0], True, maneuver.path.turn_1_radius, False)

        maneuver.thrusts[1] = self.next_thrust_from_proportion(maneuver.pthrusts[1], True)
        self.straight_1 = ManeuverSegment(maneuver.fish, maneuver.path.straight_1_length, self.turn_1.final_speed, maneuver.thrusts[1], False, 0.0, False)

        maneuver.thrusts[2] = self.next_thrust_from_proportion(maneuver.pthrusts[2], True)
        self.turn_2 = ManeuverSegment(maneuver.fish, maneuver.path.turn_2_length, self.straight_1.final_speed, maneuver.thrusts[2], True, maneuver.path.turn_2_radius, False)

        maneuver.thrusts[3] = self.next_thrust_from_proportion(maneuver.pthrusts[3], True)
        self.straight_2 = ManeuverSegment(maneuver.fish, maneuver.path.straight_2_length, self.turn_2.final_speed, maneuver.thrusts[3], False, 0.0, False)

        maneuver.thrusts[4] = self.next_thrust_from_proportion(maneuver.pthrusts[4], True)
        self.turn_3 = ManeuverSegment(maneuver.fish, maneuver.path.turn_3_length, self.straight_2.final_speed, maneuver.thrusts[4], True, maneuver.path.turn_3_radius, False)

        maneuver.final_thrust_a = self.next_thrust_from_proportion(maneuver.final_pthrust_a, False)

    def check_if_needs_to_back_up(self, maneuver):
        """Now we check how long the arbitrary-thrust portions of the maneuver took, and make sure they left the fish at the end of turn 3 downstream of
           the focal point (higher x coordinate) and with enough time and space to slown down (if needed) to the focal velocity without overshooting the focal point.
           If that condition isn't met, we gradually slow down the rest of the maneuver by reducing thrusts until it is met."""
        x_p, t_p = self.penultimate_point(maneuver)
        u_a = thrust_a_adjusted_for_tailbeats(maneuver.final_thrust_a, t_p, x_p, self.turn_3.final_speed, self.v, self.fish_total_length)
        tau_times_thrust = self.fish_mass / (self.fish_rho * self.fish_webb_factor * ALPHA * self.fish_area * Cd(self.fish_total_length, u_a))
        if self.turn_3.final_speed > self.v:
            max_possible_final_speed_a = max(self.turn_3.final_speed, maneuver.final_thrust_a)
            t_0 = 2 * tau_times_thrust * (1/self.v - 1/max_possible_final_speed_a)   # min time to slow down to v (at thrust 0)
            s_0 = 2 * tau_times_thrust * np.log(max_possible_final_speed_a / self.v) # min distance to slow down to v (at thrust 0)
        else:
            t_0 = (tau_times_thrust*np.log(((u_a - self.turn_3.final_speed)*(u_a + self.v))/((u_a + self.turn_3.final_speed)*(u_a - self.v))))/u_a   # min time to speed up down to v (at u_a)
            s_0 = tau_times_thrust*np.log((u_a**2 - self.turn_3.final_speed**2)/(u_a**2 - self.v**2)) # min distance to speed up to v (at thrust u_a)
        fish_x_at_critical_point = (x_p - s_0) # critical point being the earliest point at which fish speed can match v
        focal_point_x_at_critical_point = -self.v * (t_p + t_0)
        if focal_point_x_at_critical_point > fish_x_at_critical_point:
            amount_to_back_up = max(focal_point_x_at_critical_point - fish_x_at_critical_point, 0.001) # back up at least 0.01 cm to prevent lengthy cycling close to convergence but not quite there
            return amount_to_back_up # turn 3 needs to back up and let focal point get in front
        else:
            return 0 # turn 3 is fine

    def penultimate_point(self, maneuver):
        x_p = maneuver.path.tangent_point_F[0]
        t_p = self.turn_1.duration + self.straight_1.duration + self.turn_2.duration + self.straight_2.duration + self.turn_3.duration
        return x_p, t_p

    def plottable_segments(self, maneuver):
        '''Since there doesn't seem to be a way to store a jitclass as an attribute of another jitclass yet, I just recalculate all the segment information
           when it's actually requested elsewhere (from plotting functions, etc). That seems inefficient, but it doesn't matter when I'm just plotting.'''
        plottable = True
        fish = maneuver.fish
        turn_1 = ManeuverSegment(fish, maneuver.path.turn_1_length, maneuver.mean_water_velocity, maneuver.thrusts[0], True, maneuver.path.turn_1_radius, plottable)
        straight_1 = ManeuverSegment(fish, maneuver.path.straight_1_length, turn_1.final_speed, maneuver.thrusts[1], False, 0.0, plottable)
        turn_2 = ManeuverSegment(fish, maneuver.path.turn_2_length, straight_1.final_speed, maneuver.thrusts[2], True, maneuver.path.turn_2_radius, plottable)
        straight_2 = ManeuverSegment(fish, maneuver.path.straight_2_length, turn_2.final_speed, maneuver.thrusts[3], False, 0.0, plottable)
        turn_3 = ManeuverSegment(fish, maneuver.path.turn_3_length, straight_2.final_speed, maneuver.thrusts[4], True, maneuver.path.turn_3_radius, plottable)
        x_p, t_p = self.penultimate_point(maneuver)
        straight_3 = ManeuverFinalStraight(fish, t_p, x_p, turn_3.final_speed, maneuver.final_thrust_a, plottable)
        return (turn_1, straight_1, turn_2, straight_2, turn_3, straight_3)

    def thrust_durations(self):
        return ((self.turn_1.thrust, self.turn_1.duration),
                (self.straight_1.thrust, self.straight_1.duration),
                (self.turn_2.thrust, self.turn_2.duration),
                (self.straight_2.thrust, self.straight_2.duration),
                (self.turn_3.thrust, self.turn_3.duration),
                (self.straight_3.thrust_a_experienced, self.straight_3.duration_a),
                (self.straight_3.thrust_b, self.straight_3.duration_b))