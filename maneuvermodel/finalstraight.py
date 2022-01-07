import numpy as np
from .segment import s_t, u_t, s_u, Cd
from numba import njit, float64, boolean, optional, types
from numba.experimental import jitclass
from .maneuveringfish import swimming_activity_cost
from .constants import *

maneuver_final_straight_spec = [
    ('creation_succeeded', boolean),
    ('initial_speed', float64),
    ('final_speed_a', float64), # could also be called initial_speed_b -- same thing!
    ('initial_t', float64),
    ('initial_x', float64),
    ('can_plot',boolean),
    ('fish_webb_factor', float64),
    ('fish_min_thrust', float64),
    ('fish_base_mass',float64),
    ('fish_u_crit', float64),
    ('fish_total_length', float64),
    ('mean_water_velocity', float64), # main velocity to use for the maneuver, mean of fish & focal
    ('tau_times_thrust',float64),
    ('plottimes',optional(float64[:])),
    ('plotspeeds',optional(float64[:])),
    ('plotthrusts',optional(float64[:])),
    ('plotaccelerations',optional(float64[:])),
    ('plotpositions',optional(float64[:])),
    ('plotcosts',optional(float64[:])),
    ('t_a_range_min',float64),
    ('min_t_a',float64),
    ('max_t_a',float64),
    ('thrust_a_exerted', float64),
    ('thrust_a_experienced',float64),
    ('thrust_b',float64),
    ('adj_thrust_a', float64),
    ('adj_thrust_b', float64),
    ('cost_a',float64),
    ('duration_a',float64),
    ('length_a', float64),
    ('cost_b',float64),
    ('duration_b',float64),
    ('length_b', float64),
    ('cost',float64),
    ('duration',float64),
    ('length', float64),
    ('convergence_failure_code', types.string)
]

@njit
def thrust_a_adjusted_for_tailbeats(thrust_a, tp, xp, ui, v, fish_total_length):
    # Making this a separate function so I can call it in advance in needs_to_slow_down
    uf = thrust_a
    duration_estimate = (tp * v + xp) / (uf - v) # Duration estimate based on thrust as constant speed
    initial_tailbeat_frequency = 0.98 + 2.54 * (ui/fish_total_length) # tailbeat frequency from Webb 1991 eqn 9
    initial_tailbeat_duration = 1 / initial_tailbeat_frequency
    if duration_estimate <= initial_tailbeat_duration:
        estimated_thrust_at_end = ui + (uf - ui) * (duration_estimate / initial_tailbeat_duration)
        mean_thrust = (ui + estimated_thrust_at_end) / 2
    else:
        full_thrust_duration = duration_estimate - initial_tailbeat_duration
        mean_thrust_during_first_tailbeat = (ui + uf) / 2
        mean_thrust = uf * (full_thrust_duration / duration_estimate) + mean_thrust_during_first_tailbeat * (initial_tailbeat_duration / duration_estimate)
    return max(mean_thrust, 1.02 * v)  # don't let the adjustment drop it below 1.02*v

@jitclass(maneuver_final_straight_spec)
class ManeuverFinalStraight(object):
    
    def __init__(self, fish, initial_t, initial_x, initial_speed, thrust_a_exerted, can_plot):
        self.creation_succeeded = False
        self.convergence_failure_code = ''
        if not np.isfinite(initial_t):
            self.initial_t = initial_t  # setting to allow for diagnostics via solution.dynamics.straight_3.initial_t
            self.convergence_failed('5', 0)  # extremely rare edge case
            return
        self.initial_speed = initial_speed
        self.initial_t = initial_t
        self.initial_x = initial_x
        self.can_plot = can_plot # controls whether to (at great computational expensive) calculate plot values -- only to be used for optimal/end solutions
        self.fish_base_mass = fish.base_mass  # the normal, measurable mass of the fish, for use calculating swimming cost
        self.fish_u_crit = fish.u_crit
        self.fish_min_thrust = fish.min_thrust
        self.fish_total_length = fish.total_length
        self.fish_webb_factor = fish.webb_factor
        self.fish_base_mass = fish.base_mass # the apparent mass of the fish for use in equations of motion, i.e. the base mass times 1 + lambda
        self.mean_water_velocity = fish.mean_water_velocity
        self.fish_min_thrust = fish.min_thrust
        # We need a common value of Cd for both segments, or all the math gets way more complicated mixing constants from segments A and B into each.
        # The times when this will be most important are when the fish is using a long segment A to catch up to its focal point. In that case,
        # we would expect thrust_a to be the best approximation of the speed the fish is moving most of the time and the most relevant to the drag factor.
        try:
            self.thrust_a_exerted = thrust_a_exerted
            self.thrust_a_experienced = self.thrust_a_adjusted_for_tailbeats(thrust_a_exerted)
            Cd_a_and_b = Cd(fish.total_length, self.thrust_a_experienced)
            self.tau_times_thrust = fish.mass / (fish.rho * fish.webb_factor * ALPHA * fish.area * Cd_a_and_b) # constant for these segments since not turning
            tau_a = self.tau_times_thrust / self.thrust_a_experienced
            # Compute the bounds on t_a, the t_a_range_min restricting search to where the functions (for both min and max) can converge
            v = self.mean_water_velocity
            ut3 = self.initial_speed
            ua = self.thrust_a_experienced
            self.t_a_range_min = 0.0 if ut3 > ua else max(0.0, (self.tau_times_thrust*np.log(((ua - ut3)*(ua + v))/((ua + ut3)*(ua - v))))/ua)
            self.min_t_a = self.compute_min_t_a() # have to do min before max, then use min to set initial guess for max for best convergence
            self.max_t_a = self.compute_max_t_a()
            self.duration_a = self.min_t_a + 0.5 * (self.max_t_a - self.min_t_a) # locking in the middle of the allowed range for duration_a, because the range is so narrow it would be pointless as an optimization parameter
            # Compute the rest of segment a, then b
            self.length_a = s_t(self.initial_speed, self.thrust_a_experienced, self.duration_a, tau_a)
            self.final_speed_a = u_t(self.initial_speed, self.thrust_a_experienced, self.duration_a, tau_a)
        except Exception:
            print("Exception caught when calculating segment A of ManeuverFinalStraight in finalstraight.py")
            self.convergence_failed('9', 0)
        segment_b_completion_code = self.compute_segment_b()
        if segment_b_completion_code == 0:  # segment b constructed successfully
            self.duration = self.duration_a + self.duration_b
            self.length = self.length_a + self.length_b
            self.compute_costs()
            if can_plot: self.compute_plot_components()
            self.creation_succeeded = True
        else: # If segment B failed to converge (typically because final_speed_a < 1.002 * water_velocity, or required u_b = v) just give this solution high costs, so it will be ignored by the genetic algorithm
            self.convergence_failed('6', segment_b_completion_code)

    def convergence_failed(self, failure_code, failure_subcode):
        """ Failure code will be 6.something if segment B failed to converge above, 5 if initial_t is not finite"""
        self.duration = CONVERGENCE_FAILURE_COST
        self.length = CONVERGENCE_FAILURE_COST
        self.cost = CONVERGENCE_FAILURE_COST
        self.convergence_failure_code = failure_code
        if failure_subcode != 0:
            self.convergence_failure_code += '.' + str(failure_subcode)
        self.creation_succeeded = False

    def thrust_a_adjusted_for_tailbeats(self, thrust_a):
        return thrust_a_adjusted_for_tailbeats(thrust_a, self.initial_t, self.initial_x, self.initial_speed, self.mean_water_velocity, self.fish_total_length)

    def print_root_function_inputs_for_t_a(self):
        print("Root function inputs for t_a: {v=", self.mean_water_velocity, ", tauTimesUf=", self.tau_times_thrust, ", tp=", self.initial_t, ", xp=", self.initial_x, ", ut3=", self.initial_speed, ", ua=", self.thrust_a_experienced, "}")
        
    def min_t_a_root_function(self, t_a):
        v = self.mean_water_velocity
        tauTimesUf = self.tau_times_thrust
        tp = self.initial_t
        xp = self.initial_x
        ut3 = self.initial_speed
        ua = self.thrust_a_experienced
        ta = t_a
        if abs((ta*ua)/tauTimesUf) < EXP_OVERFLOW_THRESHOLD:
            return (ta + tp)*v + xp + tauTimesUf*np.log(16) - tauTimesUf*(2*np.log((ua + (2*ua*(-ua + ut3))/(ua - ut3 + np.exp((ta*ua)/tauTimesUf)*(ua + ut3)) + v)/v) + np.log((4*(ua*np.cosh((ta*ua)/(2.*tauTimesUf)) + ut3*np.sinh((ta*ua)/(2.*tauTimesUf)))**2)/ua**2))
        else:
            return -(ta*ua) + (ta + tp)*v + xp + tauTimesUf*np.log(16) + 2*tauTimesUf*np.log(ua) - 2*tauTimesUf*(np.log(ua + ut3) + np.log((ua + v)/v))
        
    def min_t_a_root_function_derivative(self, t_a):
        v = self.mean_water_velocity
        tauTimesUf = self.tau_times_thrust
        ut3 = self.initial_speed
        ua = self.thrust_a_experienced
        ta = t_a
        if abs((ta*ua)/tauTimesUf) < EXP_OVERFLOW_THRESHOLD:
            return -(((ua - ut3 + np.exp((ta*ua)/tauTimesUf)*(ua + ut3))*(ua - v)*(ua + v))/(-((ua - ut3)*(ua - v)) + np.exp((ta*ua)/tauTimesUf)*(ua + ut3)*(ua + v)))
        else:
            return v - ua

    def min_t_a_initial_guess(self):
        # Make an initial guess based on looking at the root function at a high value of t_a, where it very closely
        # approximates a straight line from which we can easily calculate a root in one step knowing its value and
        # derivative. Sometimes this guess is good enough it's within convergence tolerance, so we don't need to
        # use the normal iterations of Newton's method. If this guess isn't good enough, then we can just use the
        # overall range min for t_a. The min_t_a root function shape is more conducive to convergence from there
        # than the max_t_a root function is, so we don't need the extra fancy step for low values from max_t_a.
        v = self.mean_water_velocity
        tauTimesUf = self.tau_times_thrust
        tp = self.initial_t
        xp = self.initial_x
        ut3 = self.initial_speed
        ua = self.thrust_a_experienced
        main_guess = (tp * v + xp + tauTimesUf * np.log(16) + 2 * tauTimesUf * np.log(ua) - 2 * tauTimesUf * (np.log(ua + ut3) + np.log((ua + v) / v))) / (ua - v)
        return max(main_guess, self.t_a_range_min)

    def compute_min_t_a(self):
        if self.min_t_a_root_function(0.0) < 0:
            return 0.0
        range_min = self.t_a_range_min
        initial_guess = self.min_t_a_initial_guess()
        root_func_value = self.min_t_a_root_function(initial_guess)
        if abs(root_func_value) < CONVERGENCE_TOLERANCE_T_A_BOUNDS:
            return initial_guess
        else:
            converged = False
            t_a = initial_guess
            for i in range(MAX_NEWTON_ITERATIONS):
                t_a = t_a - root_func_value / self.min_t_a_root_function_derivative(t_a)
                if t_a < range_min:
                    t_a = range_min + (range_min - t_a)
                root_func_value = self.min_t_a_root_function(t_a)
                # print("Iterating through min_t_a of", t_a,"with root function value", root_func_value)
                converged = abs(root_func_value) < CONVERGENCE_TOLERANCE_T_A_BOUNDS
                if converged: break
            if not np.isfinite(t_a):
                self.print_root_function_inputs_for_t_a()
                print("Allowed minimum for min_t_a was", range_min, "and initial guess was", initial_guess, ".")
                raise ValueError('Min t_a was not finite for the final segment.')
            elif not converged:
                print("Allowed range min for min t_a was", range_min, "and initial guess was", initial_guess, ".")
                print("The thrust_a was", self.thrust_a_exerted, "exerted", self.thrust_a_experienced, "adjusted", "and water velocity was ", self.mean_water_velocity)
                print("Min t_a was not found within tolerance. Value was min t_a=",t_a," with root function value=",self.min_t_a_root_function(t_a),".")
                self.print_root_function_inputs_for_t_a()
                raise ValueError('Min_t_a failed to converge within tolerance.') # kill the program right here if convergence fails
            return t_a

    def max_t_a_root_function(self, t_a):
        v = self.mean_water_velocity
        tauTimesUf = self.tau_times_thrust
        tp = self.initial_t
        xp = self.initial_x
        ut3 = self.initial_speed
        ua = self.thrust_a_experienced
        ta = t_a
        if abs((ta*ua)/tauTimesUf) < EXP_OVERFLOW_THRESHOLD: # exact formula
            return 2*tauTimesUf + ta*v + tp*v - (2*tauTimesUf*v)/ua + (4*tauTimesUf*(-ua + ut3)*v)/(ua*(-ua + ut3 + np.exp((ta*ua)/tauTimesUf)*(ua + ut3))) + xp - tauTimesUf*np.log((ua*np.cosh((ta*ua)/(2.*tauTimesUf)) + ut3*np.sinh((ta*ua)/(2.*tauTimesUf)))**2/ua**2) - tauTimesUf*np.log((ua**2 + (-ua**4 + ua**2*ut3**2)/(ua*np.cosh((ta*ua)/(2.*tauTimesUf)) + ut3*np.sinh((ta*ua)/(2.*tauTimesUf)))**2)/v**2)
        else: # high-t approximation, a bit faster and maybe more numerically stable
            return 2*tauTimesUf + tp*v - (2*tauTimesUf*v)/ua + (4*tauTimesUf*(-ua + ut3)*v)/(ua*(-ua + ut3 + np.exp((ta*ua)/tauTimesUf)*(ua + ut3))) + ta*(-ua + v) + xp - tauTimesUf*(2*np.log((ua + ut3)/(2.*ua)) + np.log((ua**2 + (-ua**4 + ua**2*ut3**2)/(ua*np.cosh((ta*ua)/(2.*tauTimesUf)) + ut3*np.sinh((ta*ua)/(2.*tauTimesUf)))**2)/v**2))
        
    def max_t_a_root_function_derivative(self, t_a):
        v = self.mean_water_velocity
        tauTimesUf = self.tau_times_thrust
        ut3 = self.initial_speed
        ua = self.thrust_a_experienced
        ta = t_a
        if abs((2*ta*ua)/tauTimesUf) < EXP_OVERFLOW_THRESHOLD: # exact formula
            return (-(np.exp((2*ta*ua)/tauTimesUf)*(ua + ut3)**2*(ua - v)) + 2*np.exp((ta*ua)/tauTimesUf)*(ua - ut3)*(ua + ut3)*v + (ua - ut3)**2*(ua + v))/(-ua + ut3 + np.exp((ta*ua)/tauTimesUf)*(ua + ut3))**2
        else: # high-t approximation, a bit faster and maybe more numerically stable
            return -ua + v + ((ua - ut3)*(ua + ut3)*v)/(ut3*np.cosh((ta*ua)/(2.*tauTimesUf)) + ua*np.sinh((ta*ua)/(2.*tauTimesUf)))**2 - (2*ua*(ua - ut3)*(ua + ut3))/(2*ua*ut3*np.cosh((ta*ua)/tauTimesUf) + (ua**2 + ut3**2)*np.sinh((ta*ua)/tauTimesUf))

    def max_t_a_initial_guess(self):
        # Make the initial guess for max t_a, which has to be done very carefully or Newton's Method will have problems converging.
        # The first guess is based on the fact that the root function approximates a straight line at high t_a values. I used Mathematica
        # to calculate the limit of the zero of this line as t_a -> infinity, which is the first guess for max t_a and, at fairly large
        # values of t_a, is frequently good enough to satisfy the convergence tolerance without a single iteration of Newton's method.
        v = self.mean_water_velocity
        tauTimesUf = self.tau_times_thrust
        ut3 = self.initial_speed
        ua = self.thrust_a_experienced
        tp = self.initial_t
        xp = self.initial_x
        guessLarge = (2*tauTimesUf*ua - 2*tauTimesUf*v + tp*ua*v + ua*xp - tauTimesUf*ua*np.log((ua + ut3)**2/(4.*ua**2)) - tauTimesUf*ua*np.log(ua**2/v**2))/(ua*(ua - v))
        # However, if t_a is very small then its tricky curvature becomes a major problem for Newton's method and leads to infinite
        # cycling unless an initial value is chosen pretty close to the zero of the root function. The zero in this case is pretty
        # well predicted by starting from the inflection point in the root function (where its second derivative is zero), which can
        # be calculated analytically and should keep the algorithm away from the flat high-t_a region where infinite cycles happen.
        guessSmallLogArgument = ((ua - ut3) * (ua + 2 * v)) / ((ua + ut3) * (ua - 2 * v))
        guessSmall = (tauTimesUf * np.log(guessSmallLogArgument)) / ua if guessSmallLogArgument > 0 else 0
        return max(guessLarge, guessSmall)

    def compute_max_t_a(self):
        range_min = self.t_a_range_min
        initial_guess = self.max_t_a_initial_guess()
        # Now with the initial guess found, proceed to Newton's Method (unless the initial guess is good enough, which often it is)
        root_func_value = self.max_t_a_root_function(initial_guess)
        if abs(root_func_value) < CONVERGENCE_TOLERANCE_T_A_BOUNDS:
            return initial_guess
        else:
            converged = False
            t_a = initial_guess
            for i in range(MAX_NEWTON_ITERATIONS):
                t_a = t_a - root_func_value / self.max_t_a_root_function_derivative(t_a)
                if t_a < range_min:
                    t_a = range_min + (range_min - t_a)
                root_func_value = self.max_t_a_root_function(t_a)
                # print("Iterating through max_t_a of", t_a,"with root function value", root_func_value)
                converged = abs(root_func_value) < CONVERGENCE_TOLERANCE_T_A_BOUNDS
                if converged: break
            if not np.isfinite(t_a):
                self.print_root_function_inputs_for_t_a()
                print("Allowed minimum for max_t_a was", range_min, "and initial guess was", initial_guess, ".")
                print("Earlier calculated min_t_a of",self.min_t_a,"with root function value",self.min_t_a_root_function(self.min_t_a))
                raise ValueError('Max t_a was not finite for the final segment.')
            elif not converged:
                print("Allowed range min for max t_a was", range_min, "and initial guess was", initial_guess, ".")
                print("Earlier calculated min_t_a of",self.min_t_a,"with root function value",self.min_t_a_root_function(self.min_t_a))
                print("The thrust_a was", self.thrust_a_exerted, "exerted", self.thrust_a_experienced, "adjusted", "and water velocity was ", self.mean_water_velocity)
                print("Max t_a was not found within tolerance. Value was max t_a=",t_a," with root function value=",self.max_t_a_root_function(t_a),".")
                self.print_root_function_inputs_for_t_a()
                raise ValueError('Max_t_a failed to converge within tolerance.') # kill the program right here if convergence fails
            return t_a

    def segment_b_root_function_f1(self, ub_tb):
        ub = ub_tb[0]
        tb = ub_tb[1]
        v = self.mean_water_velocity
        tauTimesUf = self.tau_times_thrust
        utA = self.final_speed_a
        if abs((tb*ub)/tauTimesUf) < EXP_OVERFLOW_THRESHOLD:
            return ub + (2*ub*(-ub + utA))/(ub - utA + np.exp((tb*ub)/tauTimesUf)*(ub + utA)) - v 
        else:
            return ub - v

    def segment_b_root_function_f2(self, ub_tb):
        ub = ub_tb[0]
        tb = ub_tb[1]
        v = self.mean_water_velocity
        tp = self.initial_t
        xp = self.initial_x
        tauTimesUf = self.tau_times_thrust
        utA = self.final_speed_a
        stA = self.length_a
        ta = self.duration_a
        if abs((tb*ub)/tauTimesUf) < EXP_OVERFLOW_THRESHOLD:
            return -stA + (ta + tb + tp)*v + xp - tauTimesUf*np.log((ub*np.cosh((tb*ub)/(2.*tauTimesUf)) + utA*np.sinh((tb*ub)/(2.*tauTimesUf)))**2/ub**2)
        else: # use effectively exact high-t*u approximation to avoid double overflows in the above formula at large t
            return -stA - tb*ub + ta*v + (tb + tp)*v + xp - 2*tauTimesUf*np.log((ub + utA)/(2.*ub))
        
    def segment_b_jacobian_inverse(self, ub_tb):
        # Note that checks for invalid values (both float64 overflows and non-invertability) happen upstream of this function and aren't included here.
        ub = ub_tb[0]
        tb = ub_tb[1]
        v = self.mean_water_velocity
        tauTimesUf = self.tau_times_thrust
        utA = self.final_speed_a
        df1dub = (-(tauTimesUf * (ub - utA) ** 2) + np.exp((2 * tb * ub) / tauTimesUf) * tauTimesUf * (ub + utA) ** 2 + 2 * np.exp((tb * ub) / tauTimesUf) * ub * (-2 * tauTimesUf * utA + tb * (ub - utA) * (ub + utA))) / (tauTimesUf * (ub - utA + np.exp((tb * ub) / tauTimesUf) * (ub + utA)) ** 2)
        df1dtb = (2 * np.exp((tb * ub) / tauTimesUf) * ub ** 2 * (ub - utA) * (ub + utA)) / (tauTimesUf * (ub - utA + np.exp((tb * ub) / tauTimesUf) * (ub + utA)) ** 2)
        df2dub = (tb*ub*(ub - utA) - 2*tauTimesUf*utA + np.exp((tb*ub)/tauTimesUf)*(2*tauTimesUf*utA - tb*ub*(ub + utA)))/(ub*(ub - utA + np.exp((tb*ub)/tauTimesUf)*(ub + utA)))
        df2dtb = -ub + (2*ub*(ub - utA))/(ub - utA + np.exp((tb*ub)/tauTimesUf)*(ub + utA)) + v
        determinant_denominator = df1dub * df2dtb - df1dtb * df2dub
        if determinant_denominator != 0.0:
            det = 1 / (df1dub * df2dtb - df1dtb * df2dub)
            return np.array(((det*df2dtb,-det*df1dtb),(-det*df2dub,det*df1dub)))  # Numba chokes trying to multiply the determinant times the full array object
        else:
            return np.array(((np.nan,np.nan),(np.nan,np.nan)))

    def print_space_for_segment_b(self):
        """ This will print the fish's anticipated position at the end of segment A if it uses min_t_a or max_t_a, as compared to teh focal point position"""
        print("-------------------------------------------------------------------------------------------------------------")
        fish_position_with_min_t_a = self.initial_x  - s_t(self.initial_speed, self.thrust_a_experienced, self.min_t_a, self.tau_times_thrust / self.thrust_a_experienced)
        focal_position_with_min_t_a = (-self.mean_water_velocity * (self.initial_t + self.min_t_a))
        diff_min_t_a = fish_position_with_min_t_a - focal_position_with_min_t_a
        print("At end of segment A if it used min_t_a =", self.min_t_a, ": fish x=", fish_position_with_min_t_a, "vs focal x=", focal_position_with_min_t_a, "diff=",diff_min_t_a)
        fish_position_with_max_t_a = self.initial_x  - s_t(self.initial_speed, self.thrust_a_experienced, self.max_t_a, self.tau_times_thrust / self.thrust_a_experienced)
        focal_position_with_max_t_a = (-self.mean_water_velocity * (self.initial_t + self.max_t_a))
        diff_max_t_a = fish_position_with_max_t_a - focal_position_with_max_t_a
        print("At end of segment A if it used max_t_a =", self.max_t_a, ": fish x=", fish_position_with_max_t_a, "vs focal x=", focal_position_with_max_t_a, "diff=",diff_max_t_a)
        fish_position_with_duration_a = self.initial_x  - s_t(self.initial_speed, self.thrust_a_experienced, self.duration_a, self.tau_times_thrust / self.thrust_a_experienced)
        focal_position_with_duration_a = (-self.mean_water_velocity * (self.initial_t + self.duration_a))
        diff_duration_a = fish_position_with_duration_a - focal_position_with_duration_a
        print("At end of segment A if it used duration_a =", self.duration_a, ": fish x=", fish_position_with_duration_a, "vs focal x=", focal_position_with_duration_a, "diff=",diff_duration_a)
        print("Length_a is", self.length_a,"but should be", s_t(self.initial_speed, self.thrust_a_experienced, self.duration_a, self.tau_times_thrust / self.thrust_a_experienced))

    def print_segment_b_inputs(self):
        print("-------------------------------------------------------------------------------------------------------------")
        print("Inputs to segment B root function: {ut3=", self.initial_speed,",ua=", self.thrust_a_experienced, ",v=",
              self.mean_water_velocity,",tauTimesUf=", self.tau_times_thrust,",tp=", self.initial_t,",xp=",
              self.initial_x,",utA=", self.final_speed_a,",stA=", self.length_a,",ta=", self.duration_a,"}")
        initial_guess = self.segment_b_initial_guess()
        print("Initial guess is {ub=", initial_guess[0],",tb=", initial_guess[1],"}")
        x_space_left =  (self.initial_x - self.length_a) - (-self.mean_water_velocity * (self.initial_t + self.duration_a))
        print("Space remaining for segment B was ", x_space_left," cm to change speed from ", self.final_speed_a," to ", self.mean_water_velocity,"cm/s.")
        print("Segment A length was", self.length_a, "cm, duration", self.duration_a, "s, which was middle of range from", self.min_t_a, "to", self.max_t_a)
        u_f = 0.0001 # effectively zero thrust
        tau = self.tau_times_thrust / u_f
        min_distance_required_to_match_speed = s_u(self.final_speed_a, u_f, self.mean_water_velocity, tau)
        print("Minimum space required for segment B to match the focal velocity is ", min_distance_required_to_match_speed," cm.")

    def verify_convergence(self, verbose=False):
        # Checks to see if the fish ends up holding its position at its focal point at the end of the maneuver.
        # Note that this means checking against the mean water velocity and not actual focal velocity, since that's the velocity assumed for the whole maneuver.
        # This function is just for diagnostic exploration / verification and isn't used in the main modeling process.
        focal_point_final_position = -self.mean_water_velocity * (self.initial_t + self.duration_a + self.duration_b)
        fish_final_position = self.initial_x - (self.length_a + self.length_b)
        fish_final_velocity = u_t(self.final_speed_a, self.thrust_b, self.duration_b, self.tau_times_thrust / self.thrust_b)
        position_difference = fish_final_position - focal_point_final_position
        velocity_difference = fish_final_velocity - self.mean_water_velocity
        if position_difference > 0.1 or velocity_difference > 0.1 or verbose:
            print("Converged with position difference ", position_difference," cm and velocity difference ", velocity_difference," cm/s.")
            if position_difference > 0.1 or velocity_difference > 0.1:
                print("Final position was ", fish_final_position," cm for the fish and ", focal_point_final_position," cm for the focal point.")
                print("Final velocity was ", fish_final_velocity," cm for the fish and ", self.mean_water_velocity," cm for the water-speed 'focal' point.")

    def segment_b_initial_guess(self):
        # The initial guess for segment B is based on taking a first-order Taylor expansion of the root functions around the point (v, 0).
        # That system of 2 polynomial equations for 2 unknowns has an analytical solution, given here, which seems to work really well
        # as an approximation to the true solution, facilitating convergence.
        v = self.mean_water_velocity
        tp = self.initial_t
        xp = self.initial_x
        tauTimesUf = self.tau_times_thrust
        utA = self.final_speed_a
        stA = self.length_a
        ta = self.duration_a
        guess_ub = (utA ** 2 + v ** 2 - (2 * tauTimesUf * (utA - v) ** 2) / (-stA + (ta + tp) * v + xp)) / (2. * v)
        guess_tb = (-stA + (ta + tp)*v + xp)/(utA - v)
        return np.array([guess_ub, guess_tb])

    def compute_segment_b(self):
        try:
            v = self.mean_water_velocity
            ub_tb = self.segment_b_initial_guess()
            for i in range(MAX_NEWTON_ITERATIONS):
                if ub_tb[0] == v or abs(2 * (ub_tb[1] * ub_tb[0]) / self.tau_times_thrust) > EXP_OVERFLOW_THRESHOLD: # Will fail to converge with non-invertible Jacobian -- declare maneuver invalid
                    if ub_tb[0] == v:
                        return 1 # failure code 6.1 -- velocity for segment b converged to focal velocity, which should not allow catching up all the way
                    else:
                        return 2 # failure code 6.2 -- Jacobian not invertible
                f1 = self.segment_b_root_function_f1(ub_tb)
                f2 = self.segment_b_root_function_f2(ub_tb)
                ji = self.segment_b_jacobian_inverse(ub_tb)
                ub_tb = np.abs(ub_tb - np.array([f1 * ji[0,0] + f2 * ji[0,1], f1 * ji[1,0] + f2 * ji[1,1]])) # manually write the matrix x vector product since np.matmul() wasn't in Numba earlier
                if np.isnan(ub_tb[0]): # deal with extremely rare times the determinant somehow comes out to 0 for reasons other than overflow or ub == v
                    return 3 # failure code 6.3
                if abs(f1) + abs(f2) <= CONVERGENCE_TOLERANCE_SEGMENT_B:
                    break
            if abs(f1) + abs(f2) > CONVERGENCE_TOLERANCE_SEGMENT_B:
                return 4 # failure code 6.4: reached max_newton_iterations without converging or hitting a known failure mode
            self.thrust_b = max(ub_tb[0], self.fish_min_thrust) # just to prevent divide-by-zero if thrust_b converges toward 0
            self.duration_b = ub_tb[1]
            tau_b = self.tau_times_thrust / self.thrust_b
            self.length_b = s_t(self.final_speed_a, self.thrust_b, self.duration_b, tau_b)
            return 0  # completion code 0 for success
        except Exception:
            print("Exception caught in compute_segment_b() for ManeuverFinalStraight in finalstraight.py")
            return 5 # failure code 6.5: unknown exception caught computing segment B

    def compute_costs(self):
        self.cost_a = self.swimming_cost_rate(self.thrust_a_exerted) * self.duration_a
        self.cost_b = self.swimming_cost_rate(self.thrust_b) * self.duration_b
        self.cost = self.cost_a + self.cost_b

    def swimming_cost_rate(self, u_thrust):
        """ Rate of energy expenditure (J/s) for swimming at a given thrust. """
        u_cost = np.sqrt(self.fish_webb_factor * u_thrust**2)
        return swimming_activity_cost(self.fish_base_mass, u_cost, self.fish_u_crit)
        
    def compute_plot_components(self):
        try:
            tau_a = self.tau_times_thrust / self.thrust_a_experienced
            tau_b = self.tau_times_thrust / self.thrust_b
            plottimes_a = np.linspace(0, self.duration_a, 100)
            plottimes_b = np.linspace(self.duration_a, self.duration, 100)
            plotcosts_a = np.full(plottimes_a.size, (self.cost_a / self.duration_a if self.duration_a > 0 else 0))
            plotcosts_b = np.full(plottimes_b.size, (self.cost_b / self.duration_b if self.duration_b > 0 else 0))
            plotthrusts_a = np.full(plottimes_a.size, np.sqrt(self.fish_webb_factor * self.thrust_a_experienced ** 2))
            plotthrusts_b = np.full(plottimes_b.size, np.sqrt(self.fish_webb_factor*self.thrust_b**2))
            plotspeeds_a = np.empty(100, dtype=np.float64)
            plotpositions_a = np.empty(100, dtype=np.float64)
            for i in range(len(plottimes_a)): # because list comprehensions aren't supported in numba
                plotspeeds_a[i] = u_t(self.initial_speed, self.thrust_a_experienced, plottimes_a[i], tau_a)
                plotpositions_a[i] = s_t(self.initial_speed, self.thrust_a_experienced, plottimes_a[i], tau_a) / self.length
            plotspeeds_b = np.empty(plottimes_b.size, dtype=np.float64)
            plotpositions_b = np.empty(plottimes_b.size, dtype=np.float64)
            for i in range(len(plottimes_b)):
                plotspeeds_b[i] = u_t(self.final_speed_a, self.thrust_b, plottimes_b[i] - self.duration_a, tau_b)
                plotpositions_b[i] = (self.length_a + s_t(self.final_speed_a, self.thrust_b, plottimes_b[i] - self.duration_a, tau_b)) / self.length
            self.plotpositions = np.empty(200, dtype=float64)
            self.plottimes = np.empty(200, dtype=np.float64)
            self.plotspeeds = np.empty(200, dtype=np.float64)
            self.plotcosts = np.empty(200, dtype=np.float64)
            self.plotthrusts = np.empty(200, dtype=np.float64)
            self.plotaccelerations = np.zeros(200, dtype=float64)
            self.plottimes[0:100] = plottimes_a # all this because numba doesn't support np.concatenate
            self.plottimes[100:200] = plottimes_b
            self.plotspeeds[0:100] = plotspeeds_a
            self.plotspeeds[100:200] = plotspeeds_b
            self.plotpositions[0:100] = plotpositions_a
            self.plotpositions[100:200] = plotpositions_b
            self.plotcosts[0:100] = plotcosts_a
            self.plotcosts[100:200] = plotcosts_b
            self.plotthrusts[0:100] = plotthrusts_a
            self.plotthrusts[100:200] = plotthrusts_b
            for i in range(0, self.plottimes.size-2):
                if self.plottimes[i+1]-self.plottimes[i] < 1e-20: # would use 0, but some numerical errors create slightly nonzero values
                    self.plotaccelerations[i] = self.plotaccelerations[i-1] # maintain previous acceleration level when going between steps with 0 time
                else:
                    self.plotaccelerations[i] = (self.plotspeeds[i+1]-self.plotspeeds[i])/(self.plottimes[i+1]-self.plottimes[i])
            self.plotaccelerations[0] = 0.0 # filler, because the first element is removed by the plotting function later
        except Exception:
            print("Exception caught in compute_plot_components() for ManeuverFinalStraight in finalstraight.py")
