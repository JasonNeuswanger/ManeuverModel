import numpy as np
from numba import float64, boolean
from numba.experimental import jitclass
from .segment import ManeuverSegment, Cd
from .finalstraight import ManeuverFinalStraight
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
    ('energy_cost', float64),
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
    ('needs_to_slow_down', boolean),
    ('needs_to_speed_up', boolean),
    ('bad_thrust_b_penalty', float64),
    ('violates_acceleration_limit_penalty', float64),
    ('mean_of_pthrusts_except_last', float64),
    ('v', float64) # shortcut to maneuver.mean_water_velocity
]

@jitclass(maneuver_dynamics_spec)
class ManeuverDynamics(object):
    
    def __init__(self, maneuver):
        # Relevant attributes of the fish attribute are duplicated to avoid circular dependency among jitclasses
        #print("\nINITIALIZING DYNAMICS")
        #print("Initial pthrusts are: ", maneuver.pthrusts[0], " -- ", maneuver.pthrusts[1], " -- ", maneuver.pthrusts[2], " -- ", maneuver.pthrusts[3], " -- ", maneuver.pthrusts[4],".")

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
        self.needs_to_slow_down = False  # Gets set to True if the fish needs to slow down to let the focal point get in front of it by the end of turn 3
        self.needs_to_speed_up = False   # Gets set to True if the fish is going too slow at the end of turn 3 to accelerate to thrust > v without violating thrust limit
        self.mean_of_pthrusts_except_last = np.mean(maneuver.pthrusts[:-1])
        self.violates_acceleration_limit_penalty = 0.0
        try:
            self.build_segments(maneuver)    # This function will set the above flags to slow down or speed up if needed
        except Exception:
            print("Exception caught when building maneuver segments as initially called from ManeuverDynamics.__init__.")
        loop_count = 0
        while self.needs_to_slow_down or self.needs_to_speed_up:
            speed_change_increment = 0.03
            if self.needs_to_slow_down:
                for i in range(5):
                    slowdown_multiplier = 1 - speed_change_increment * (5 - i)
                    maneuver.pthrusts[i] *= slowdown_multiplier
            if self.needs_to_speed_up:
                for i in range(5):
                    speedup_amount = speed_change_increment * (1 + i)
                    maneuver.pthrusts[i] = min(maneuver.pthrusts[i] + speedup_amount, 1.0)
            self.mean_of_pthrusts_except_last = np.mean(maneuver.pthrusts[:-1])
            try:
                self.build_segments(maneuver)
            except Exception:
                print("Exception caught when looping through building of maneuver segments in ManeuverDynamics.__init__.")
            loop_count += 1
            # print("Finished change loop with pthrusts: ", maneuver.pthrusts[0], " -- ", maneuver.pthrusts[1], " -- ", maneuver.pthrusts[2], " -- ", maneuver.pthrusts[3], " -- ", maneuver.pthrusts[4], ".")
            if loop_count > 1/speed_change_increment:
                if self.needs_to_speed_up:
                    maneuver.convergence_failure_code = 1
                elif self.needs_to_slow_down:
                    maneuver.convergence_failure_code = 2
                else:
                    maneuver.convergence_failure_code = 3
                self.energy_cost = CONVERGENCE_FAILURE_COST
                self.total_cost = CONVERGENCE_FAILURE_COST
                return

        # -------------------------
        # COULD WE SKIP A LOT OF THIS IF I JUST DO THE MATH FOR MAKING THE FISH SLOW DOWN FROM ABOVE THE FOCAL POINT IN THE FINAL STRAIGHT??
        # If the fish ends in front of the focal point, it can always slow down and drift downstream toward the focal point until it hits it, then up the thrust to v.
        # In some maneuvers this might be energetically advantageous, although it's probably fairly rare as it requires extra upstream progress and takes less advantage of lower swim costs at the focal speed.
        # It would seem to save some efficiency rebuilding segments, but it might also lead to a vast majority of random maneuvers going too fast to be realistic,
        # and the optimization algorithm having trouble finding its way into the better part of the solution space.
        # -------------------------

        # Now that the end of turn 3 is somewhere downstream of the focal point, calculate the final straight to catch up to it
        try:
            x_p, t_p = self.penultimate_point(maneuver)
            self.straight_3 = ManeuverFinalStraight(fish, self.v, t_p, x_p, self.turn_3.final_speed, maneuver.final_thrust_a, maneuver.final_duration_a_proportional, False)
            if not self.straight_3.creation_succeeded:
                maneuver.convergence_failure_code = 4
                self.energy_cost = CONVERGENCE_FAILURE_COST
                self.total_cost = CONVERGENCE_FAILURE_COST
                return
        except Exception:
            print("Exception caught when creating ManeuverFinalStraight in ManeuverDynamics.__init__.")

        # Compute a penalty on the solution if acceleration at the start of segment B of the final straight is over the limit, because the way it's calculated
        # makes it computationally infeasible to restrict this a priori. This way we let the optimization algorithm do the heavy lifting for this one constraint
        try:
            threshold = MAX_ACCELERATION
            tau_times_thrust = self.fish_mass / (self.fish_rho * self.fish_webb_factor * ALPHA * self.fish_area * Cd(self.fish_total_length, (self.straight_3.final_speed_a + self.v) / 2))
            min_thrust_b = max(0.0, np.sqrt(-2 * threshold * tau_times_thrust + self.straight_3.final_speed_a ** 2) if threshold < self.straight_3.final_speed_a ** 2 / (2. * tau_times_thrust) else 0.0)
            min_penalty = 2.0 # at least 200 % cost penalty for having bad thrust b, so it's never chosen if a solution with an allowable thrust can be found
            if self.straight_3.thrust_b < min_thrust_b:
                self.bad_thrust_b_penalty = min_penalty + (min_thrust_b - self.straight_3.thrust_b) # beyond the min, use a penalty proportional to the difference to help gide the algorithm
            else:
                self.bad_thrust_b_penalty = 0.0
        except Exception:
            print("Exception caught in ManeuverDynamics.__init__ when computing penalties for excessive acceleration at the start of segment B.")

        # if self.straight_3.thrust_b < min_thrust_b or self.straight_3.thrust_b > max_thrust_b:
        #     print("Straight 3 thrust b is",self.straight_3.thrust_b,"which is outside allowed interval (",min_thrust_b,",",max_thrust_b,")")

        # When thrust b is lower than the min allowed, that means the fish was going too fast coming out of segment a
        # When thrust b is higher than the max allowed, that means the fish was going too slow coming out of segment a
        # We can't tweak thrust_a here because it's already restricted previously
        # So the only knob to tweak is duration_a
        # If u_a > ut3 (fish is speeding up during a), then increasing duration_a should increase final_speed_a
        # If u_a < ut3 (fish is slowing down during a), then increasing duration_a should decrease final_speed_a

        # I've now calculated a new, more restrictive max_t_a that leaves room to reach v using a min thrust based on the acceleration threshold rather than 0.
        # This SHOULD automatically constrain the thrust in segment B, since it is determined by t_a.

        maneuver.path.update_with_straight_3_length(self.straight_3.length)

        self.wait_duration = maneuver.path.wait_length / maneuver.mean_water_velocity
        self.pursuit_duration = self.turn_1.duration + self.straight_1.duration
        self.return_duration = self.turn_2.duration + self.straight_2.duration + self.turn_3.duration + self.straight_3.duration
        self.moving_duration = self.pursuit_duration + self.return_duration
        self.duration = self.wait_duration + self.pursuit_duration + self.return_duration # duration of the full maneuver in seconds
        self.mean_speed = maneuver.path.total_length / self.moving_duration
        self.energy_cost = self.turn_1.cost + self.straight_1.cost + self.turn_2.cost + self.straight_2.cost + self.turn_3.cost + self.straight_3.cost # energy cost of the full maneuver in Joules
        self.opportunity_cost = fish.NREI * (self.wait_duration + self.pursuit_duration)                                 # cost of not searching during the wait/pursuit
        self.total_cost = self.energy_cost + self.opportunity_cost

    def min_next_thrust(self, previous_speed, is_not_final_a):
        threshold = MAX_ACCELERATION
        tau_times_thrust = self.fish_mass / (self.fish_rho * self.fish_webb_factor * ALPHA * self.fish_area * Cd(self.fish_total_length, previous_speed))
        min_next_thrust = np.sqrt(-2*threshold*tau_times_thrust + previous_speed**2) if threshold < previous_speed**2/(2*tau_times_thrust) else 0.0
        regular_min = self.fish_min_thrust if is_not_final_a else 1.02 * self.v
        min_next_thrust = max(regular_min, min_next_thrust) # enforce constraints from the fish's thrust limits
        return min_next_thrust

    def max_next_thrust(self, previous_speed, is_not_final_a):
        """ Normally, the maximum next thrust is constrained by the acceleration limit. However, in very rare cases mostly
            involving extremely short maneuvers in implausibly fast water, this can prevent the algorithm from converging altogether.
            Even with all pthrusts up through segment A of the final straight being 1.0, the fish is still too far behind to
            catch up to its focal point without violating the acceleration limit at the start of segment A in the final straight
            in order to reach or exceed the water velocity and make progress upstream. My solution to this is to check whether
            the fish at the start of segment A has already maxed its pthrusts (through the 'speeding up' iteration above), and
            still needs to speed up for that segment; if it does, its thrust is capped at the fish's maximum rather than
            constrained by acceleration limits. However, next_thrust_from_proportion() calculates acceleration directly and applies a
            fitness penalty in the optimizaton algorithm to any solutions exceeding the acceleration limit, so that solutions
            which use this mechanism to facilitate convergence are not favored over those which respect the acceleration limit
            in all stages, if any of those are available. This entire mechanism affects only a tiny fraction of edge cases and
            not realistic maneuvers, but those funny convergence failures were causing NaN values in the interpolation tables,
            which unaccepably prevent use of fast interpolation methods. This method lets us plausibly fill in those blanks."""
        tau_times_thrust = self.fish_mass / (self.fish_rho * self.fish_webb_factor * ALPHA * self.fish_area * Cd(self.fish_total_length, previous_speed))
        if self.mean_of_pthrusts_except_last == 1.0 and previous_speed < self.v and not is_not_final_a:
            max_next_thrust = self.fish_max_thrust
        else:
            max_next_thrust = np.sqrt(2 * MAX_ACCELERATION * tau_times_thrust + previous_speed**2)
        max_next_thrust = min(self.fish_max_thrust, max_next_thrust)
        return max_next_thrust

    def next_thrust_from_proportion(self, previous_speed, proportional_next_thrust, is_not_final_a, is_turn, turn_radius):
        """ Pass turn_radius = np.nan and is_turn = False for straights. Otherwise pass true and the actual radius.
            This function returns the thrust exerted by the fish.
        """
        # Calculate the minimum and maximum for the next thrust based on the acceleration limit, not accounting for the turn factor.
        min_next_thrust = self.min_next_thrust(previous_speed, is_not_final_a)
        max_next_thrust = self.max_next_thrust(previous_speed, is_not_final_a)
        # Now adjust these for turns. Because the output of this function is the exerted thrust and not the actual thrust affecting
        # the physics in the case of turns, I basically invert the turn adjustment to thrust (from calculate_dynamics() in segment.py)
        # by solving taking the equation for u_f as a function of u_thrust and solving it for u_thrust with u_f being the bounds above,
        # which lets me adjust the limits here upward so that they fall within the allowable range of acceleration and deceleration after
        # turn adjustment. Before I made this correction, I was getting peak decelerations that exceeded the limit because the fish was
        # using a lower effective thrust (after the turn correction) than the limit allows. One trick to this adjustment is that
        # the turn_factor as used in calculate_dynamics() depends on drag factor Cd, which depends on the average of the initial speed
        # and thrust, but we're still trying to calculate the thrust and don't know it yet at this point. Therefore, I use a different
        # value of Cd here, based only on the initial speed, to calculate the turn factor. That's why I preface the variables with preliminary_.
        # Because Cd is slightly different between here and calculate_dynamics(), this adjustment might not perfectly enforce the acceleration
        # bounds every time, but in practice it's so extremely close it doesn't matter.
        if is_turn:
            preliminary_Cd = Cd(self.fish_total_length, previous_speed) # using u_i for drag factor here instead of average of u_i and u_thrust, because u_thrust isn't known yet
            preliminary_turn_factor = np.sqrt(1 + (2 * self.fish_volume * (1 + self.fish_waterlambda) / (turn_radius * self.fish_area * preliminary_Cd * ALPHA * self.fish_webb_factor)) ** 2)
            if SENSITIVITY_TURN_FACTOR_MULTIPLIER != 1.0:
                preliminary_turn_factor = 1 + SENSITIVITY_TURN_FACTOR_MULTIPLIER * (preliminary_turn_factor - 1)
            min_next_thrust = np.sqrt(preliminary_turn_factor * min_next_thrust**2)
            max_next_thrust = np.sqrt(preliminary_turn_factor * max_next_thrust**2)
        self.needs_to_speed_up = max_next_thrust < min_next_thrust
        if self.needs_to_speed_up:
            # print("Setting needs_to_speed_up=True because max_next_thrust=",max_next_thrust," is < min_next_thrust=",min_next_thrust," for previous speed",previous_speed)
            next_thrust = min_next_thrust # just to prevent errors in the meantime before the maneuver is sped up
        else:
            next_thrust = min_next_thrust + (max_next_thrust - min_next_thrust) * proportional_next_thrust
        # We allow for violating the acceleration limit, but assign a solution fitness penalty proportional to the violation, so
        # it is only used if there is no other choice to allow the solution to converge.
        tau_times_thrust = self.fish_mass / (self.fish_rho * self.fish_webb_factor * ALPHA * self.fish_area * Cd(self.fish_total_length, previous_speed))
        next_max_acceleration = abs(((next_thrust - previous_speed) * (next_thrust + previous_speed)) / (2 * tau_times_thrust))
        if next_max_acceleration > MAX_ACCELERATION:
            # Note that subtracting 1 from the ratio below would allow for penalties of effectively 0 when next_max_acceleration is very slightly above
            # max_acceleration. I really want to avoid solutions that violate MAX_ACCELERATION unless they are the ONLY ones that converge, so I am only
            # subtracting 0.5 instead of 1.0, meaning solutions which don't violate the limit will have at minimum a 50 % fitness advantage over those that do.
            self.violates_acceleration_limit_penalty = (next_max_acceleration / MAX_ACCELERATION) - 0.5
        return next_thrust

    def build_segments(self, maneuver):
        """ Here we create each segment, starting with the proportional representations of the thrusts, creating each actual thrust (m/s) value
            based on the speed at the end of the preceding segment and limits on acceleration/deceleration, then plugging it the resulting
            actual thrust (based on asymptotic speed in m/s) in to create the next segment."""
        fish = maneuver.fish
        path = maneuver.path
        verbose = False

        maneuver.thrusts[0] = self.next_thrust_from_proportion(self.v, maneuver.pthrusts[0], True, True, path.turn_1_radius)
        self.turn_1 = ManeuverSegment(fish, path.turn_1_length, self.v, maneuver.thrusts[0], True, path.turn_1_radius, False)
        if verbose: print("final speed from turn 1 is",self.turn_1.final_speed,"after thrust",maneuver.thrusts[0])

        maneuver.thrusts[1] = self.next_thrust_from_proportion(self.turn_1.final_speed, maneuver.pthrusts[1], True, False, np.nan)
        self.straight_1 = ManeuverSegment(fish, path.straight_1_length, self.turn_1.final_speed, maneuver.thrusts[1], False, 0.0, False)
        if verbose: print("final speed from straight 1 is", self.straight_1.final_speed, "after thrust", maneuver.thrusts[1])

        maneuver.thrusts[2] = self.next_thrust_from_proportion(self.straight_1.final_speed, maneuver.pthrusts[2], True, True, path.turn_2_radius)
        self.turn_2 = ManeuverSegment(fish, path.turn_2_length, self.straight_1.final_speed, maneuver.thrusts[2], True, path.turn_2_radius, False)
        if verbose: print("final speed from turn 2 is", self.turn_2.final_speed, "after thrust", maneuver.thrusts[2])

        maneuver.thrusts[3] = self.next_thrust_from_proportion(self.turn_2.final_speed, maneuver.pthrusts[3], True, False, np.nan)
        self.straight_2 = ManeuverSegment(fish, path.straight_2_length, self.turn_2.final_speed, maneuver.thrusts[3], False, 0.0, False)
        if verbose: print("final speed from straight 2 is", self.straight_2.final_speed, "after thrust", maneuver.thrusts[3])

        maneuver.thrusts[4] = self.next_thrust_from_proportion(self.straight_2.final_speed, maneuver.pthrusts[4], True, True, path.turn_3_radius)
        self.turn_3 = ManeuverSegment(fish, path.turn_3_length, self.straight_2.final_speed, maneuver.thrusts[4], True, path.turn_3_radius, False)
        if verbose: print("final speed from turn 3 is", self.turn_3.final_speed, "after thrust", maneuver.thrusts[4])

        maneuver.final_thrust_a = self.next_thrust_from_proportion(self.turn_3.final_speed, maneuver.final_pthrust_a, False, False, np.nan)
        if verbose: print("final thrust a will be ",maneuver.final_thrust_a)

        self.check_if_needs_to_slow_down(maneuver)

    def check_if_needs_to_slow_down(self, maneuver):
        """Now we check how long the arbitrary-thrust portions of the maneuver took, and make sure they left the fish at the end of turn 3 downstream of
           the focal point (higher x coordinate) and with enough time and space to slown down (if needed) to the focal velocity without overshooting the focal point.
           If that condition isn't met, we gradually slow down the rest of the maneuver by reducing thrusts until it is met. Note that we use the fixed
           value of ALPHA (currently 4) in the drag factor constant part tau_times_thrust, because we want to represent a fish slowing down to the best of
           its ability, i.e. maximal braking, not a fish coasting to reduce drag (even though thrust is 0 in this case)."""
        x_p, t_p = self.penultimate_point(maneuver)
        u_a = maneuver.final_thrust_a
        tau_times_thrust = self.fish_mass / (self.fish_rho * self.fish_webb_factor * ALPHA * self.fish_area * Cd(self.fish_total_length, u_a))
        if self.turn_3.final_speed > self.v:
            max_possible_final_speed_a = max(self.turn_3.final_speed, maneuver.final_thrust_a)
            t_0 = 2 * tau_times_thrust * (1/self.v - 1/max_possible_final_speed_a)   # min time to slow down to v (at thrust 0)
            s_0 = 2 * tau_times_thrust * np.log(max_possible_final_speed_a / self.v) # min distance to slow down to v (at thrust 0)
        else:
            t_0 = (tau_times_thrust*np.log(((u_a - self.turn_3.final_speed)*(u_a + self.v))/((u_a + self.turn_3.final_speed)*(u_a - self.v))))/u_a   # min time to speed up down to v (at u_a)
            s_0 = tau_times_thrust*np.log((u_a**2 - self.turn_3.final_speed**2)/(u_a**2 - self.v**2)) # min distance to speed up to v (at thrust u_a)
        self.needs_to_slow_down = -self.v * (t_p + t_0) > (x_p - s_0)  # returns true if the fish needs to slow down up through turn 3 to make the final straight possible

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
        straight_3 = ManeuverFinalStraight(fish, maneuver.mean_water_velocity, t_p, x_p, turn_3.final_speed, maneuver.final_thrust_a, maneuver.final_duration_a_proportional, plottable)
        return (turn_1, straight_1, turn_2, straight_2, turn_3, straight_3)

    def thrust_durations(self):
        return ((self.turn_1.thrust, self.turn_1.duration),
                (self.straight_1.thrust, self.straight_1.duration),
                (self.turn_2.thrust, self.turn_2.duration),
                (self.straight_2.thrust, self.straight_2.duration),
                (self.turn_3.thrust, self.turn_3.duration),
                (self.straight_3.thrust_a, self.straight_3.duration_a),
                (self.straight_3.thrust_b, self.straight_3.duration_b))