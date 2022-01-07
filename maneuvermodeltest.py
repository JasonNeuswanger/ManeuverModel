# -*- coding: utf-8 -*-

import numpy as np
from maneuvermodel import optimize, visualize, maneuveringfish
from maneuvermodel.constants import DEFAULT_OPT_N, DEFAULT_OPT_ITERATIONS

# import cProfile
# import pstats
# import timeit, time
# from importlib import reload

# ---------------------------------------------------------------------------------------------------#
#                                                                                                   #
#                                        TESTING CODE                                               #
#                                                                                                   #
# ---------------------------------------------------------------------------------------------------#

typical = {'Chinook Salmon': {'fork_length': 4.6, 'mean_water_velocity':10, 'mass': 0.85, 'temperature': 9, 'max_thrust': 62, 'NREI': 0.017, 'detection_distance': 8, 'SMR': 226},
           'Dolly Varden': {'fork_length': 18, 'mean_water_velocity': 28, 'mass': 51, 'temperature': 10, 'max_thrust': 94, 'NREI': 0.017, 'detection_distance': 17, 'SMR': 52},
           'Arctic Grayling': {'fork_length': 43, 'mean_water_velocity': 45, 'mass': 920, 'temperature': 6, 'max_thrust': 159, 'NREI': 0.017, 'detection_distance': 35, 'SMR': 40}
           }

def create_typical_fish(species, **kwargs):  # typical fish
    return maneuveringfish.ManeuveringFish(fork_length=typical[species]['fork_length'],
                                           mean_water_velocity=kwargs.get('mean_water_velocity', typical[species]['mean_water_velocity']),
                                           base_mass=typical[species]['mass'],
                                           temperature=typical[species]['temperature'],
                                           SMR=typical[species]['SMR'],
                                           max_thrust=typical[species]['max_thrust'],
                                           NREI=typical[species]['NREI'],
                                           use_total_cost=kwargs.get('use_total_cost', False),
                                           disable_wait_time=False)

typical_fish = {}
typical_fish['Chinook Salmon'] = create_typical_fish('Chinook Salmon')
typical_fish['Dolly Varden'] = create_typical_fish('Dolly Varden')
typical_fish['Arctic Grayling'] = create_typical_fish('Arctic Grayling')

def typical_maneuver(species, **kwargs):
    fish = kwargs.get('modified_fish', typical_fish[species])
    default_detection_point_3D = (-typical[species]['detection_distance'] / 1.414, typical[species]['detection_distance'] / 1.414, 0.0)
    detection_point_3D = kwargs.get('modified_detection_point_3D', default_detection_point_3D)
    return optimize.optimal_maneuver(fish, detection_point_3D=detection_point_3D, **kwargs)

# species = 'Dolly Varden'
# test_fish = create_typical_fish('Dolly Varden', use_total_cost=False)
# detection_point_3D = (-typical[species]['detection_distance'] / 1.414, typical[species]['detection_distance'] / 1.414, 0.0)
# # opt = optimize.run_convergence_test(typical_fish[species], detection_point_3D=detection_point_3D)
# opt, opt_model = optimize.optimal_maneuver(test_fish, n=DEFAULT_OPT_N, max_iterations=DEFAULT_OPT_ITERATIONS, detection_point_3D=detection_point_3D, tracked=True, return_optimization_model=True)

# todo make spreadsheet creation code export reproduction info for nans

# best solution is 0.159198

# visualize.summarize_solution(opt, display = True, title = 'Typical Dolly', export_path = None, detailed=True, add_text_panel=True)


test_x=-135.3565092
test_y=1.2
fork_length=3.0
velocity=1.0
fish = maneuveringfish.ManeuveringFish(fork_length=fork_length,
                                           mean_water_velocity=velocity,
                                           base_mass=0,  # defaults to regression from length
                                           temperature=10,  # irrelevant in this case
                                           SMR=0,  # irrelevant in this case
                                           max_thrust=250,
                                           NREI=0,  # irrelevant in this case
                                           use_total_cost=False,
                                           disable_wait_time=False)
opt, opt_model = optimize.optimal_maneuver(fish,
                                detection_point_3D=(test_x, test_y, 0.0),
                                max_iterations=(DEFAULT_OPT_ITERATIONS/2),
                                max_n=(DEFAULT_OPT_N/2),
                                tracked=True,
                                return_optimization_model=True,
                                suppress_output=False)
print("Had {0} nonconvergent evaluations".format(opt_model.nfe_nonconvergent))



# Testing the number of function evaluations requiring slowdowns over time and therefore basically lost in the algorithm
import seaborn as sns
sns.lineplot(x=opt_model.tracked_nfe, y=opt_model.tracked_nfe_final_turn_adjusted)
# previously about one third of the effort was wasted evaluating solutions that slowed down and went haywire
print(opt_model.tracked_best_had_final_turn_adjusted)

from collections import Counter
print("Convergence failure codes and frequencies:", Counter(opt_model.tracked_convergence_failure_codes.split('_')[1:]).most_common())

# ---------------------------------------------------------------------------------------------------#
#                                                                                                   #
#                                        TESTING VS A PAPER                                         #
#                                                                                                   #
# ---------------------------------------------------------------------------------------------------#

# This comparison is too crude (uncertainty in the applicable numbers from the paper to compare model predictions)
# to include in our paper in any way, but it was interesting to see what came out from the comparison.

def check_johansen_et_al(display=False, suppress_output=False):
    # Checks maneuver cost predictions against empirical measurements from Johansen et al
    # It looks like they had costs per maneuver of 1.28 mg O2 / kg for freestream swimming and 0.78 for refuge swimming,
    # both at a temperature of 15 C and velocity of 68 cm/s, with rainbow trout of size 33 cm and 423 g. With the width x
    # depth of the tube being 25 x 26 cm, we might assume the average detected prey's lateral distance was around 15 cm or
    # so, and I'll assume prey were detected fairly early (say 30 cm upstream) on average.
    fork_length = 33
    mean_water_velocity = 68
    xd = -30
    yd = 15
    max_thrust = 2.4 * fork_length + 40
    fish_mass = 423  # this input defaults the model to a rainbow trout length-mass regression
    fish_SMR = 0.0  # default to sockeye SMR for given temperature
    fish_NREI = 0.0  # unused in this case
    temperature = 15  # only for SMR
    use_total_cost = False
    disable_wait_time = False
    fish = maneuveringfish.ManeuveringFish(fork_length, mean_water_velocity, fish_mass, temperature, fish_SMR, max_thrust, fish_NREI, use_total_cost, disable_wait_time)
    detection_point_3D = (xd, yd, 0.0)
    maneuver = optimize.optimal_maneuver(fish, detection_point_3D=detection_point_3D)
    visualize.summarize_solution(maneuver, display=display, title='Cost Table Check', export_path=None, detailed=True)
    joulesPerMgO2 = 3.36 * 4.184
    costMgO2 = maneuver.activity_cost / joulesPerMgO2
    costMgO2PerKG = costMgO2 / (fish_mass / 1000)
    print("Maneuver cost of locomotion alone is ", costMgO2PerKG, "mg O2 / kg")
    print("Focal swimming cost including SMR in is ", 3600 * (fish.focal_swimming_cost_including_SMR / joulesPerMgO2), " mg O2 / kg / hr")  # convert from focal_swimming_cost_including_SMR in J/s
    SMR_per_hour = 3600 * (fish.SMR_J_per_s / joulesPerMgO2) / (fish_mass / 1000)
    print("SMR per hour is ", SMR_per_hour, " mg O2 / kg / hr")  # convert from focal_swimming_cost_including_SMR in J/s
    print("Cost of SMR during maneuver is ", SMR_per_hour * maneuver.duration / 3600)
    print("Locomotion cost for focal swimming for duration of maneuver is ", (maneuver.duration * fish.focal_swimming_cost_of_locomotion / joulesPerMgO2) / (fish_mass / 1000), " mg O2 / kg")
    total_cost_during_maneuver = costMgO2PerKG + SMR_per_hour * maneuver.duration / 3600
    print(f"MAIN COMPARISON: estimated cost (locomotion + SMR) of maneuver is {total_cost_during_maneuver} mg O2 / kg, compared to Johansen's 1.28 mg O2 / kg" )
    return maneuver, fish

test_maneuver, test_fish = check_johansen_et_al(display=True)
# RESULT: estimated cost (locomotion + SMR) of maneuver is 0.31746643784432216 mg O2 / kg, compared to Johansen's 1.28 mg O2 / kg


def check_cost_table_value(fork_length, mean_water_velocity, xd, yd, display=False, suppress_output=False):
    max_thrust = 2.4 * fork_length + 40
    fish_mass = 0.0  # this input defaults the model to a rainbow trout length-mass regression
    fish_SMR = 0.0  # unused in this case
    fish_NREI = 0.0  # also unused in this case
    temperature = 10  # also unused in this case
    use_total_cost = False
    disable_wait_time = False
    fish = maneuveringfish.ManeuveringFish(fork_length, mean_water_velocity, fish_mass, temperature, fish_SMR, max_thrust, fish_NREI, use_total_cost, disable_wait_time)
    detection_point_3D = (xd, yd, 0.0)
    maneuver = optimize.optimal_maneuver(fish, detection_point_3D=detection_point_3D)
    # visualize.summarize_solution(maneuver, display=display, title='Cost Table Check', export_path=None, detailed=True)
    return maneuver

# This one converges to a best solution despite an acceleration penalty
# I earlier noted, when debugging a crash here: "thrusts keep increasing but fish can't maneuver fast enough"
man = check_cost_table_value(3, 7, 40, -0.1, 0.7, display=True, suppress_output=False)


# Compare vs Puckett & Dill 1984
# They also estimated a cost-per-maneuver of 5.7 mgO2/kg for maneuvers covering a distance averaging 18.8 cm in 0.48 s at an average speed of 43.5 cm/s (in water coordinates) or 39 cm/s (in ground coordinates).
# This comes out to 5.7 mgO2/kg = 0.00684 mgO2 for a 1.2 g fish = 0.0927 J per maneuver.
# It seems there's no way to get the fish to swim anywhere near that fast to catch the prey at their treatment velocity of 4.5 cm/s: the fish just wasn't maneuvering optimally, but
# burning energy in a suboptimal burst for kicks. Trying a higher water velocity to better approximate their swimming speed and corresponding cost.
pdFish = maneuveringfish.ManeuveringFish(fork_length = 4.9, mean_water_velocity= 30, base_mass = 1.2, temperature = 15.0, SMR = 0, max_thrust = 250, NREI = 2.0, use_total_cost = False, disable_wait_time = True)
fittest_solution = optimize.optimal_maneuver(pdFish, detection_point_3D=(-1, 3, 0))
print(""""\n Total path length: {0}\n
          Duration: {1}\n
  Pursuit duration: {2}\n
        Mean speed: {3}\n
   Locomotion cost: {4}\n
 Energy cost w/SMR: {5}\n""".format(fittest_solution.path.total_length, fittest_solution.duration, fittest_solution.pursuit_duration, fittest_solution.dynamics.mean_speed, fittest_solution.activity_cost, pdFish.maneuver_energy_cost_including_SMR(fittest_solution)))
visualize.summarize_solution(fittest_solution, display = True, title = "Attempt to replicate Puckett & Dill", export_path = None, detailed=True)
#  Total path length: 17.950554721707903
#           Duration: 0.47928813234599077
#   Pursuit duration: 0.12136950814424816
#         Mean speed: 37.45253326812956
#    Locomotion cost: 0.00435545547755718
#  Energy cost w/SMR: 0.004658295191422853
# This result gets kind of close to swimming the same speed for the same amount of time, and the energy cost is about 20X below Puckett & Dill's estimate.
#
# Look at Rand & Hinch 1998 for the method of partitioning aerobic and anaerobic pathways and rationale for taxing anaerobic pathways at 15 % extra
#
