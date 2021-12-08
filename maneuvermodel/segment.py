# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, float64, boolean, optional
from numba.experimental import jitclass
from .maneuveringfish import swimming_activity_cost
from .constants import *

# I'm making functions that ManeuverSegment and FinalManeuverSegment have in common be separate jitfunctions outside the
# main jitclasses, so that I'm not duplicating the same thing within the file. Normally I would use subclassing for this,
# but that's not supported in jitclasses.

@jit(float64(float64, float64, float64, float64), nopython=True)
def t_u(u_i, u_f, u, tau): # t_u = time required to attain target speed u after beginning at speed u_i with thrust u_f
    if u_i == u_f: # move at constant speed u_i = u_f
        return 0 if u == u_i else np.inf
    else:
        Q = (u_f + u_i) / (u_f - u_i)
        return tau*np.log((u + u_f)/(Q*(u_f - u)))   

@jit(float64(float64, float64, float64, float64), nopython=True)
def s_t(u_i, u_f, t, tau): # s_t = distance traveled at time t after beginning at speed u_i with thrust u_f
    # This function includes an excellent approximation for large t values that can cause the exponential terms to overflow numpy doubles
    if u_i == u_f: # move at constant speed u_i = u_f
        return t*u_i
    else:
        Q = (u_f + u_i) / (u_f - u_i)
        if (t/tau > EXP_OVERFLOW_THRESHOLD): # extremely accurate approximation
            return u_f*(t + 2*tau*np.log(Q/(1 + Q)))
        else:
            return tau*u_f*np.log( ((Q + np.exp(-t/tau)) * (1 + Q*np.exp(t/tau)))/(Q + 1)**2 ) # exact expression for s_t

@jit(float64(float64, float64, float64, float64), nopython=True)
def u_t(u_i, u_f, t, tau): # u_t = velocity at time t after beginning with speed u_i and thrust u_f
    if u_i == u_f: # move at constant speed u_i = u_f
        return u_i
    else:
        Q = (u_f + u_i) / (u_f - u_i)
        return u_f*(Q - np.exp(-t/tau))/(Q + np.exp(-t/tau))

@jit(float64(float64, float64, float64, float64), nopython=True)
def t_s(u_i, u_f, s, tau): # t_s = time taken to travel distance s after beginning at speed u_i with thrust u_f
    if u_i == u_f: # move at constant speed u_i = u_f
        return s / u_i
    else:
        Q = (u_f + u_i) / (u_f - u_i)
        if s/(tau*u_f) > 200: # approximation to avoid float64 overflow from high values inside the exponent, based on a first-order series expansion about the value s/(tau*u_f) = 200
            return tau*(-200 + s/(tau*u_f)) + tau*np.log(3.6129868840628745e86 + (3.6129868840628745e86 + 7.225973768125749e86*Q)/Q**2 + 1.3440585709080678e43*np.sqrt((-4*Q*(1 + Q)**2 + 7.225973768125749e+86*(1 + Q)**4)/Q**4))
        else: # use exact formula if the approximation isn't needed
            return tau*np.log(0.5*(-((2*Q - np.exp(s/(tau*u_f))*(1 + Q)**2)/Q**2) + np.sqrt(-4/Q**2 + (2*Q - np.exp(s/(tau*u_f))*(1 + Q)**2)**2/Q**4)))

@jit(float64(float64, float64), nopython=True)
def Cd(total_length, mean_speed):
    # Now calculate the drag coefficient Cd.
    nu = 0.01  # kinematic viscosity of water, units cm^-2 * s^-1
    Re = total_length * mean_speed / nu  # Reynolds number
    return 0.072 * Re ** -0.2

@jit()
def s_u(u_i, u_f, u, tau):
    # Distance traveled before speed reaches value u after starting at speed u_i with thrust u_f
    # This is not used in the model itself, but is included for diagnostic/exploratory purposes
    Q = (u_f + u_i) / (u_f - u_i)
    return tau*u_f * np.log((4*Q*u_f**2)/((1 + Q)**2*(-u**2 + u_f**2)))

# Now the main ManeuverSegment class
    
maneuver_segment_spec = [
    ('length', float64),
    ('u_i', float64),
    ('u_thrust',float64),
    ('is_turn',boolean),
    ('radius',float64),
    ('can_plot',boolean),
    ('fish_mass',float64),
    ('fish_base_mass',float64),
    ('fish_total_length',float64),
    ('fish_area',float64),
    ('fish_volume',float64),
    ('Cd',float64),
    ('fish_rho',float64),
    ('fish_waterlambda',float64),
    ('fish_temperature',float64),
    ('fish_SMR',float64),
    ('fish_webb_factor',float64),
    ('fish_u_crit',float64),
    ('turn_factor',float64),
    ('duration',float64),
    ('plottimes',optional(float64[:])),
    ('plotspeeds',optional(float64[:])),
    ('plotthrusts',optional(float64[:])),
    ('plotaccelerations',optional(float64[:])),
    ('plotpositions',optional(float64[:])),
    ('plotcosts',optional(float64[:])),
    ('final_speed',float64),
    ('cost',float64)
]

@jitclass(maneuver_segment_spec)
class ManeuverSegment(object):
    
    def __init__(self, fish, length, u_i, u_thrust, is_turn, radius, can_plot):
        if self.is_turn and self.radius == 0:
            print("Cannot have a turn radius of 0!")
        self.length = length
        self.u_i = u_i
        self.u_thrust = u_thrust
        self.is_turn = is_turn
        self.radius = radius
        self.can_plot = can_plot # controls whether to (at great computational expensive) calculate plot values -- only to be used for optimal/end solutions
        self.fish_mass = fish.mass # the apparent mass of the fish for use in equations of motion, i.e. the base mass times 1 + lambda
        self.fish_base_mass = fish.base_mass # the normal, measurable mass of the fish, for use calculating swimming cost
        self.fish_total_length = fish.total_length  # for calculating drag factor Cd
        self.fish_area = fish.area
        self.fish_volume = fish.volume
        self.fish_rho = fish.rho
        self.fish_waterlambda = fish.waterlambda
        self.fish_temperature = fish.temperature
        self.fish_SMR = fish.SMR
        self.fish_u_crit = fish.u_crit
        self.fish_webb_factor = fish.webb_factor
        self.Cd = Cd(self.fish_total_length, (self.u_i + u_thrust) / 2.0)
        self.turn_factor = 1 if not self.is_turn else np.sqrt(1 + (2 * self.fish_volume * (1 + self.fish_waterlambda) / (self.radius * self.fish_area * self.Cd * ALPHA * self.fish_webb_factor))**2)
        if self.is_turn and SENSITIVITY_TURN_FACTOR_MULTIPLIER != 1.0:
            self.turn_factor = 1 + SENSITIVITY_TURN_FACTOR_MULTIPLIER * (self.turn_factor - 1)
        # Initial values of final stats
        self.duration = 0.0
        self.final_speed = 0.0
        self.cost = 0.0
        # Compute the dynamics once everything is set
        self.calculate_dynamics()

    def tau(self, u_f):
        c = 0.5 * self.fish_rho * self.fish_webb_factor * ALPHA * self.fish_area * self.Cd
        return self.fish_mass / (2 * c * u_f)
            
    def swimming_cost_rate(self, u_thrust):
        """ Rate of energy expenditure (J/s) for swimming at a given thrust. """
        u_cost = np.sqrt(self.turn_factor * self.fish_webb_factor * u_thrust**2)
        return swimming_activity_cost(self.fish_base_mass, u_cost, self.fish_u_crit)

    def calculate_dynamics(self):
        """Calculate the maneuver dynamics.
        Note that u_thrust is the thrust the fish is exerting (not counting the Webb factor w adjustment, which comes when calculating costs). During turns,
        some of that exertion is used to overcome centripetal force and does not contribute to forward motion along the path. Thus the adjusted u_f,
        which equals u_thrust during straights but is smaller during turns, describes the amount of thrust contributing to the equations of motion, but regular
        u_thrust is used for costs."""
        u_f = self.u_thrust if not self.is_turn else np.sqrt(self.u_thrust**2 / self.turn_factor) # The turn-straight difference is built into turn_factor anyway, but the conditional here a little bit of processing time
        tau = self.tau(u_f)
        self.duration = t_s(self.u_i, u_f, self.length, tau)
        self.cost = self.duration * self.swimming_cost_rate(self.u_thrust) # note: not using the turn-adjusted thrust (u_f) for costs, which are based on the force the fish EXERTS not force toward forward motion
        self.final_speed = u_t(self.u_i, u_f, self.duration, tau)
        if self.can_plot:
            self.plottimes = np.linspace(0, self.duration, 100)
            self.plotpositions = np.empty(100, dtype=float64)
            self.plotspeeds = np.empty(100, dtype=float64)
            self.plotaccelerations = np.zeros(self.plottimes.size, dtype=float64)
            self.plotcosts = np.full(self.plottimes.size, self.swimming_cost_rate(self.u_thrust))
            self.plotthrusts = np.full(self.plottimes.size, np.sqrt(self.fish_webb_factor*self.u_thrust**2))
            for i in range(len(self.plottimes)): # because list comprehensions aren't supported in numba
                self.plotspeeds[i] = u_t(self.u_i, u_f, self.plottimes[i], tau)
                self.plotpositions[i] = s_t(self.u_i, u_f, self.plottimes[i], tau) / self.length
            for i in range(0,self.plottimes.size-2):
                if self.plottimes[i+1]-self.plottimes[i] < 1e-20: # would use 0, but some numerical errors create slightly nonzero values
                    self.plotaccelerations[i] = self.plotaccelerations[i-1] # maintain previous acceleration level when going between steps with 0 time
                else:
                    self.plotaccelerations[i] = (self.plotspeeds[i+1]-self.plotspeeds[i])/(self.plottimes[i+1]-self.plottimes[i])
            self.plotaccelerations[0] = 0.0 # filler, because the first element is removed by the plotting function later