import numpy as np
from numba import float64, uint64, boolean, jit
from numba.experimental import jitclass
from .constants import *

@jit(float64(float64, float64, float64), nopython=True)
def swimming_activity_cost(mass, speed, u_crit):
    # Provides cost associated with swimming activity ONLY (in J/s), excluding standard metabolic rate (SMR), for a fish 
    # of the given mass (g) swimming at the given speed (cm/s). Used in this file and in segment.py.
    # 
    # It is based on Trudel & Welch's (2005) best-fit model for Brett and Glass's (1973) sockeye salmon data. This model
    # is additive on SMR, rather than multiplicative, which provides a closer fit to Brett's data and more realistic 
    # qualitative relationships between cost, speed, and temperature (especially Brett 1964 Fig 16). Unlike models
    # that make swimming costs multiplicative on SMR (while usually still not making the multiplier temperature-dependent),
    # this one takes temperature out of the picture completely with regard to activity cost, meaning the maneuver interpolation 
    # tables don't need to be temperature-specific (so we don't need 25X as many of them and can be more precise with the ones
    # we have). Effects of temperature on total cost via SMR can be added back in by the end user, using species-specific
    # SMR equations if desired, which would not be possible if SMR were used within maneuver model itself. Thus, only the 
    # additional activity cost portion of this model is sockeye-specific, and sockeye are the best species from which to 
    # borrow these costs because Brett's dataset contains by far the largest ranges of temperature, mass, and swimming speed.
    # The oxycalorific equivalent is the same as the 14.1 recommended in Videler 1993 (which Nick used in the original model) 
    # and slightly higher than 13.556 J/mgO2 recommnded by Elliott and Davidson 1975 for non-ureotelic carnivores like trout
    # It also matches Brett's (1976) recommendation for steadily swimming sockeye salmon of (3.36 cal/mgO2) and 4.184 J/cal
    # Because this function is called so often, its components were commented out and combined into the simplest form below.
    # alpha_0 = 0.00193045 # =exp(-6.25)
    # delta = 0.72
    # twlambda = 1.60
    # oq = 3.36*4.184 # Oxycaloric equivalent in J/mgO2
    # hoursPerSecond = 1/3600
    # activity_portion_of_metabolic_rate = alpha_0 * mass**delta * speed**twlambda # mgO2/hour, directly from Trudel & Welch 2005
    # return hoursPerSecond * oq * activity_portion_of_metabolic_rate
    #
    # A customization of this follows the logic (but not the exact form) of Rand & Hinch 1998 to penalize anaerobic swimming
    # above and beyond what would happen from extrapolating the aerobic equation described above. The fish uses purely aerobic
    # respiration at speeds below 0.8*U_crit, but any energy spent above the cost of swimming at 0.8*U_crit is increased by
    # 15 % to account for reduced energetic efficiency of anaerobic respiration.
    u_thresh = 0.8 * u_crit
    if speed <= u_thresh:
        return 0.0000075385359467 * mass**0.72 * speed**1.60
    else:
        aerobic_cost_portion = 0.0000075385359467 * mass**0.72 * u_thresh**1.60
        aerobic_estimate_at_speed = 0.0000075385359467 * mass**0.72 * speed**1.60
        difference = aerobic_estimate_at_speed - aerobic_cost_portion
        anaerobic_multiplier = 1.15
        return aerobic_cost_portion + anaerobic_multiplier * difference

maneuvering_fish_spec = [
    ('fork_length', float64),
    ('total_length', float64),
    ('mean_water_velocity', float64),
    ('rho', float64),
    ('waterlambda', float64),
    ('temperature', float64),
    ('webb_factor', float64),
    ('min_thrust', float64),
    ('max_thrust', float64),
    ('min_turn_radius',float64),
    ('NREI', float64),
    ('base_mass',float64), # actual mass for purposes of physiology
    ('mass', float64),     # apparent mass for purposes of physics
    ('volume', float64),
    ('area', float64),
    ('nu', float64),
    ('swimming_speed',float64),
    ('u_crit', float64),
    ('SMR',float64),
    ('SMR_J_per_s',float64),
    ('focal_swimming_cost_of_locomotion', float64),
    ('focal_swimming_cost_including_SMR', float64),
    ('coasting_cost_including_SMR', float64),
    ('use_total_cost', boolean),
    ('disable_wait_time', boolean)
]

@jitclass(maneuvering_fish_spec)
class ManeuveringFish(object):
    
    def __init__(self, fork_length, mean_water_velocity, base_mass, temperature, SMR, max_thrust, NREI, use_total_cost, disable_wait_time): # can't use **kwargs in a jitclass
        # Note: Pass in mass and SMR of 0 to use the respective defaults for each
        self.fork_length = fork_length                     # fork length in cm
        self.mean_water_velocity = mean_water_velocity     # focal point water velocity in cm/s -- used to compute main maneuver speed and costs at focal point
        self.temperature = temperature                     # temperature in degrees C, only used for optional post-processing analysis (dependent on SMR)
        self.rho = 1.0                                     # density of water = 1 g/cm^3
        self.waterlambda = 0.2                             # factor by which effective mass increases due to entrained water
        self.webb_factor = 2.83                            # 'Webb Factor' used to modify thrust for unsteady swimming, taken from Webb 1991 Table 4, not the rounded suggestion of 3 on p. 589
        if SENSITIVITY_WEBB_FACTOR_MULTIPLIER != 1.0:
            self.webb_factor = 1 + SENSITIVITY_WEBB_FACTOR_MULTIPLIER * (self.webb_factor - 1)
        self.NREI = NREI # expected long-term NREI (Joules/second), for optional optimization incorporating energetic opportunity cost of time spent maneuvering
        # For maximum thrust, specify 'None' to use the default of 250 cm/s, which was roughly estimated as the size-independent physiological maximum for rainbow trout from
        # 9.6 to 38.7 cm based on Table 4 in Webb 1976. This is a good estimate at the real maximum. But optimum thrusts in practice are always far smaller, so much of the algorithm
        # search space is wasted by considering such high thrusts. For species-independent analyses of model performance (like plotting costs vs fish mass), using the default via 'None'
        # is appropriate, but for our tests on individual species we are using species-specific limits slightly larger than the maximum optimal thrust in any maneuver.
        self.max_thrust = 250.0 if max_thrust is None else max_thrust
        self.min_thrust = 0.1 # cm/s (there is no real minimum obviously, just need something to avoid zero)
        self.total_length = -0.027 + 1.072*self.fork_length # Total length in cm from Simpkins & Hubert 1996 for rainbow trout, derived from Carlander 1969; this relationship had an R^2 of 0.999
        # Minimum turning radius, which I looked up from Webb 1976 (calibrated for rainbow trout from 9.6 to 38.7 cm)
        self.min_turn_radius = 0.17*self.total_length # cm; comes out to about 4.3 for default parameters
        if base_mass == 0: # Provide a mass of 0 to use length-mass regression from Simpkins & Hubert 1996 for rainbow trout >= 120 mm: log_10 mass (grams) = -5.023 + 3.024 * log_10 total_length (mm)
            self.base_mass = 10**(-5.023 + 3.024 * np.log10(10*self.total_length)) # multiply by 10 to convert cm to the mm input of the original formula
        else:
            self.base_mass = base_mass
        self.volume = self.base_mass # volume in cubic centimeters, because a neutrally buoyant fish has same density as water (1 g/cm^3) -- set this based on true base mass before modifying for tiny fish
        self.mass = self.base_mass * (1 + self.waterlambda) # mass in grams accounting account for entrained water mass
        self.area = 1.78 * self.fork_length**1.88 # From O'Shea et al 2006, formula for rainbow trout

        oq = 14.05834 # oxycaloric equivalent in units (J/mgO2), swimming_activity_cost() for details
        self.SMR = self.sockeye_SMR() if SMR == 0 else SMR # Takes SMR input in mgO2/kg/hour, or else calculated using sockeye salmon as the default
        self.SMR_J_per_s = self.SMR * (1/3600.0) * (self.base_mass/1000.0) * oq
        self.focal_swimming_cost_of_locomotion = swimming_activity_cost(self.base_mass, mean_water_velocity, self.u_crit)
        self.focal_swimming_cost_including_SMR = self.SMR_J_per_s + self.focal_swimming_cost_of_locomotion
        self.coasting_cost_including_SMR = self.SMR_J_per_s
        self.u_crit = 36.23 * self.fork_length**0.19         # Grayling relationship from Jones et al 1974, via Hughes & Dill 1990
        # Numerical algorithm settings and variables
        self.use_total_cost = use_total_cost                 # Fitness goal -- True to use swimming + opportunity cost, False to just use swimming cost
        self.disable_wait_time = disable_wait_time           # Prevent fish from ever waiting before beginning maneuver, useful for testing against observed maneuvers defined by start of motion

    def sockeye_SMR(self): 
        # Returns standard metabolic in mgO2/kg/hr using Trudel & Welch's (2005) fit to Brett & Glass's (1973) sockeye data.
        # The original equation gives mgO2/hr, and it is converted here to use the more standard mass-specific number so that
        # custom-provided SMR values can also be given in the more common mgO2/kg/hr units.
        alpha_1 = np.exp(-2.94)
        beta = 0.87
        phi = 0.064
        return alpha_1 * self.base_mass**beta * np.exp(phi * self.temperature) / (self.base_mass / 1000.0)
        
    def maneuver_energy_cost_including_SMR(self, maneuver): # full energy spent by the fish during the maneuver (including SMR) but not opportunity cost
        return maneuver.dynamics.moving_duration * self.SMR_J_per_s + maneuver.dynamics.activity_cost
        
    def maneuver_total_cost_including_SMR(self, maneuver): # full cost of the maneuver including opportunity cost
        return maneuver.dynamics.moving_duration * self.SMR_J_per_s + maneuver.dynamics.total_cost