# -*- coding: utf-8 -*-

import numpy as np
from maneuvermodel import maneuver, segment, maneuveringfish, optimize, optimize_cuckoo, visualize, dynamics

# import cProfile
# import pstats
# import timeit, time
# from importlib import reload

# ---------------------------------------------------------------------------------------------------#
#                                                                                                   #
#                                        TESTING CODE                                               #
#                                                                                                   #
# ---------------------------------------------------------------------------------------------------#

typical = {'Chinook Salmon': {'fork_length': 4.6, 'focal_velocity': 10, 'prey_velocity': 11, 'mass': 0.85, 'temperature': 9, 'max_thrust': 62, 'NREI': 0.017, 'detection_distance': 8, 'SMR': 226},
           'Dolly Varden': {'fork_length': 18, 'focal_velocity': 28, 'prey_velocity': 29, 'mass': 51, 'temperature': 10, 'max_thrust': 94, 'NREI': 0.017, 'detection_distance': 17, 'SMR': 52},
           'Arctic Grayling': {'fork_length': 43, 'focal_velocity': 42, 'prey_velocity': 48, 'mass': 920, 'temperature': 6, 'max_thrust': 159, 'NREI': 0.017, 'detection_distance': 35, 'SMR': 40}
           }


def create_typical_fish(species, **kwargs):  # typical fish
    return maneuveringfish.ManeuveringFish(fork_length=typical[species]['fork_length'],
                                           focal_velocity=typical[species]['focal_velocity'],
                                           mass=typical[species]['mass'],
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
    detection_point_3D = kwargs.get('detection_point_3D', default_detection_point_3D)
    prey_velocity = typical[species]['prey_velocity']
    return optimize.optimal_maneuver(fish, detection_point_3D=detection_point_3D, prey_velocity=prey_velocity,**kwargs)

test_fish = create_typical_fish('Dolly Varden', use_total_cost=True)
test = typical_maneuver('Dolly Varden', modified_fish=test_fish)


# fittest = typical_maneuver('Dolly Varden')
# visualize.summarize_solution(fittest, display = True, title = 'Typical Dolly', export_path = None, detailed=True)

# Testing out the optimize convergence testing code
species = 'Dolly Varden'
fish = typical_fish[species]
detection_point_3D = (-typical[species]['detection_distance'] / 1.414, typical[species]['detection_distance'] / 1.414, 0.0)
prey_velocity = typical[species]['prey_velocity']
#optimize.run_convergence_test(fish, detection_point_3D, iterations=100, n=20, global_iterations=500, global_n=50, export_path="/Users/Jason/Desktop/test.pdf", display=False)
#optimize.run_convergence_test(fish, detection_point_3D, iterations=600, n=70, global_iterations=2500, global_n=250, n_tests=5)

for i in range(10):
    opt_CS = optimize_cuckoo.optimal_maneuver_CS(fish, detection_point_3D)
for i in range(10):
    opt = optimize.optimal_maneuver(fish, detection_point_3D)

import cProfile, pstats
cProfile.run('optimize.optimal_maneuver(fish, detection_point_3D)','runstats')
#cProfile.run('optimize_cuckoo.optimal_maneuver_CS(fish, detection_point_3D)','runstats')
p = pstats.Stats('runstats')
p.strip_dirs().sort_stats('tottime').print_stats(25)
p.print_callers(25)

# Lowest energy cost after 1000 CS iterations (   50026 evaluations) was   0.162862 joules. Mean speed 33.7 cm/s,  1.75 bodylengths/s. Metabolic rate   453.3 mg O2/kg/hr ( 9.7X SMR).
# Lowest energy cost after 1000 CS iterations (   50026 evaluations) was   0.154223 joules. Mean speed 34.3 cm/s,  1.78 bodylengths/s. Metabolic rate   486.1 mg O2/kg/hr (10.3X SMR).
# Lowest energy cost after 1000 CS iterations (   50026 evaluations) was   0.165572 joules. Mean speed 33.6 cm/s,  1.74 bodylengths/s. Metabolic rate   461.1 mg O2/kg/hr ( 9.9X SMR).
# Lowest energy cost after 1000 CS iterations (   50026 evaluations) was   0.164980 joules. Mean speed 33.9 cm/s,  1.76 bodylengths/s. Metabolic rate   473.4 mg O2/kg/hr (10.1X SMR).
# Lowest energy cost after 1000 CS iterations (   50026 evaluations) was   0.172153 joules. Mean speed 33.2 cm/s,  1.72 bodylengths/s. Metabolic rate   449.1 mg O2/kg/hr ( 9.6X SMR).
# Lowest energy cost after 10000 CS iterations (  500026 evaluations) was   0.151216 joules. Mean speed 34.5 cm/s,  1.79 bodylengths/s. Metabolic rate   489.1 mg O2/kg/hr (10.4X SMR).
# Lowest energy cost after 300000 CS iterations (15000026 evaluations) was   0.146549 joules. Mean speed 35.3 cm/s,  1.83 bodylengths/s. Metabolic rate   517.5 mg O2/kg/hr (11.0X SMR).
# ----------------------------------------------------------------------
# Dynamics of segment pursuit turn      of length   3.88: thrust=33.749, duration=0.132, final_speed= 30.005,      cost:   0.0135053.
# Dynamics of segment pursuit straight  of length  13.53: thrust=35.077, duration=0.393, final_speed= 35.076,      cost:   0.0353966.
# Dynamics of segment return turn       of length   5.27: thrust=39.635, duration=0.154, final_speed= 33.749,      cost:   0.0228593.
# Dynamics of segment return straight   of length  21.25: thrust=38.432, duration=0.559, final_speed= 38.432,      cost:   0.0591407.
# Dynamics of segment final turn        of length   3.60: thrust=36.013, duration=0.101, final_speed= 34.273,      cost:   0.0107322.
# Dynamics of segment final straight    of length   2.62: duration_a=0.041, duration_b=0.043, thrusts=( 29.07,  25.69), cost:   0.0049147.
# ----------------------------------------------------------------------
# Energy cost =  0.1613 J ( 0.1465 J without SMR,  0.1702 J w/opportunity cost of  0.0089 J).
# Total duration = 1.421941 s (wait time 0.000 s, pursuit 0.524 s). Traveled distance   50.1 cm at mean speed 35.3 cm/s
# Fittest.proportions(): simplifies to array([0.70, 1.0, 0.40, 1.0, 0.0, 0.13, 0.002, 0.018, 0.0, 0.45, 0.71, 0.0])
# array([7.01904006e-01, 9.99999997e-01, 4.03324215e-01, 9.99999846e-01,
#        1.93926614e-10, 1.30305404e-01, 2.02804963e-03, 1.79763709e-02,
#        8.93744561e-08, 4.51044756e-01, 7.05350112e-01, 0.00000000e+00])


def typical_maneuver_SAMA(species, **kwargs):
    fish = typical_fish[species]
    default_detection_point_3D = (-typical[species]['detection_distance'] / 1.414, typical[species]['detection_distance'] / 1.414, 0.0)
    detection_point_3D = kwargs.get('detection_point_3D', default_detection_point_3D)
    prey_velocity = typical[species]['prey_velocity']
    # return optimize_sama.optimal_maneuver_SAMA(fish, detection_point_3D=detection_point_3D, prey_velocity=prey_velocity, N=kwargs.get('N', 30), iterations=kwargs.get('iterations', 1000), suppress_output=kwargs.get('suppress_output', False))
    return optimize_sama.optimal_maneuver_SAMA(fish, detection_point_3D=detection_point_3D, prey_velocity=prey_velocity, **kwargs)


fittest_SAMA = typical_maneuver_SAMA('Dolly Varden',
                                     iterations=2000,
                                     N=30,
                                     N_ordinary_min=5,
                                     N_ordinary_max=25,
                                     alpha=0.95,
                                     beta=1.05,
                                     gamma=0.03
                                     )
# visualize.summarize_solution(fittest_SAMA, display = False, title = 'Typical Dolly', export_path = None, detailed=True)
# difficult optimizing may come from rarely considering actual max and min

# Scenarios that did not work right, before the most recent fixes, but are working well now
# stretch = typical_maneuver('Dolly Varden', iterations=1000, detection_point_3D=(238.2,600,0), suppress_output=True) # did not converge
# stretch = typical_maneuver('Dolly Varden', iterations=1000, detection_point_3D=(238.1,600,0), suppress_output=True) # did converge
# stretch = typical_maneuver('Dolly Varden', iterations=1000, detection_point_3D=(10,-0.001,0), suppress_output=True) # gave ZeroDivisionError
# visualize.summarize_solution(stretch, display = True, title = 'Typical Dolly', export_path = None, detailed=True)


# ---------------------------------------------------------------------------------------------------#
#                                                                                                   #
#                                       TESTING NAN CIRCUMSTANCES                                   #
#                                                                                                   #
# ---------------------------------------------------------------------------------------------------#

from maneuvermodel import maneuver, segment, maneuveringfish, optimize_cuckoo, visualize, dynamics
import numpy as np

# fvs = np.linspace(1, 50, 100)
fvs = [20]

adj_thrust_bs = []
ecs = []
cost_bs = []
final_duration_a_s = []

for fv in fvs:
    nan_fork_length = 5.5  # 3.4
    nan_focal_velocity = fv  # 34
    nan_prey_velocity = 20  # 34

    nan_fish = maneuveringfish.ManeuveringFish(fork_length=nan_fork_length,
                                               focal_velocity=nan_focal_velocity,
                                               mass=0.0,
                                               temperature=10,
                                               SMR=0.0,
                                               max_thrust=2.4 * nan_fork_length + 40,
                                               NREI=0.0,
                                               use_total_cost=False,
                                               disable_wait_time=False)

    # nan_detection_point_3D = (160, 160, 0.0) # hardest one for this example
    nan_detection_point_3D = (6, 16, 0.0)  # test something more realistic

    sol = optimize_cuckoo.optimal_maneuver_CS(nan_fish, detection_point_3D=nan_detection_point_3D, prey_velocity=nan_prey_velocity, n=25, iterations=3000, p_a=0.25, suppress_output=False)
    # sol.dynamics.straight_3.print_segment_b_inputs()
    # sol.print_inputs()
    # sol.dynamics.straight_3.verify_convergence(True)
    adj_thrust_bs.append(sol.dynamics.straight_3.adj_thrust_b)
    ecs.append(sol.energy_cost)
    final_duration_a_s.append(sol.dynamics.straight_3.duration_a)
    cost_bs.append(sol.dynamics.straight_3.cost_b)

import matplotlib.pyplot as plt

# plt.plot(fvs, adj_thrust_bs)
# plt.plot(fvs, final_duration_a_s)


visualize.summarize_solution(sol, display=True, title='Cost Table Check', export_path=None, detailed=True)
print("thrust_a", sol.dynamics.straight_3.thrust_a)
print("adj_thrust_a", sol.dynamics.straight_3.adj_thrust_a)
print("cost per unit time straight 3a :", sol.dynamics.straight_3.cost_a / sol.dynamics.straight_3.duration_a)
print("cost per unit time straight 2  :", sol.dynamics.straight_2.cost / sol.dynamics.straight_2.duration)

# ---------------------------------------------------------------------------------------------------#
#                                                                                                   #
#                TESTING ACTUAL SIZE OF EFFECT OF MEAN VS SEPARATE VELOCITIES                       #
#                                                                                                   #
# Rather than testing on all my measured maneuvers (cumbersome), I arrange a separate test with
# randomly selected values from realistic ranges.
# ---------------------------------------------------------------------------------------------------#

from maneuvermodel import maneuver, segment, maneuveringfish, optimize_cuckoo, visualize, dynamics
import numpy as np
import matplotlib.pyplot as plt

chinook_vals = {'species': 'Chinook', 'length_range': (3.2, 6.4), 'velocity_range': (4, 22), 'reaction_distance_mean': 8.2}
dolly_vals = {'species': 'Dollies', 'length_range': (11.2, 23.1), 'velocity_range': (14, 39), 'reaction_distance_mean': 17.6}
grayling_vals = {'species': 'Grayling', 'length_range': (32.2, 51.4), 'velocity_range': (21, 76), 'reaction_distance_mean': 36.7}

NUM_RANDOM_TESTS = 50

fish_vals = chinook_vals
cuckoo_n = 25
cuckoo_iter = 1000


def test_velocity_handling_method(fish_vals):
    print("Testing velocity handling method for ", fish_vals['species'])
    energy_differences = []
    pursuit_differences = []
    for i in range(NUM_RANDOM_TESTS):
        print("Running test ", i, " of ", NUM_RANDOM_TESTS)
        detection_point_3D = (np.sign(np.random.random() - 0.25) * np.random.exponential(fish_vals['reaction_distance_mean']), np.random.exponential(fish_vals['reaction_distance_mean']), 0)
        fork_length = np.random.uniform(fish_vals['length_range'][0], fish_vals['length_range'][1])
        velocity_1 = np.random.uniform(fish_vals['velocity_range'][0], fish_vals['velocity_range'][1])
        velocity_2 = np.random.uniform(fish_vals['velocity_range'][0], fish_vals['velocity_range'][1])
        focal_velocity = min(velocity_1, velocity_2)
        prey_velocity = max(velocity_1, velocity_2)
        mean_velocity = (focal_velocity + prey_velocity) / 2
        fish_normal = maneuveringfish.ManeuveringFish(fork_length=fork_length,
                                                      focal_velocity=focal_velocity,
                                                      mass=0.0,
                                                      temperature=10,
                                                      SMR=0.0,
                                                      max_thrust=2.4 * fork_length + 40,
                                                      NREI=0.0,
                                                      use_total_cost=False,
                                                      disable_wait_time=True)
        sol_normal = optimize_cuckoo.optimal_maneuver_CS(fish_normal, detection_point_3D=detection_point_3D, prey_velocity=prey_velocity, n=cuckoo_n, iterations=cuckoo_iter, p_a=0.25, suppress_output=False)
        fish_onevelocity = maneuveringfish.ManeuveringFish(fork_length=fork_length,
                                                           focal_velocity=mean_velocity,
                                                           mass=0.0,
                                                           temperature=10,
                                                           SMR=0.0,
                                                           max_thrust=2.4 * fork_length + 40,
                                                           NREI=0.0,
                                                           use_total_cost=False,
                                                           disable_wait_time=True)
        sol_onevelocity = optimize_cuckoo.optimal_maneuver_CS(fish_onevelocity, detection_point_3D=detection_point_3D, prey_velocity=mean_velocity, n=cuckoo_n, iterations=cuckoo_iter, p_a=0.25, suppress_output=False)
        energy_differences.append((sol_normal.energy_cost - sol_onevelocity.energy_cost) / sol_normal.energy_cost)
        pursuit_differences.append((sol_normal.pursuit_duration - sol_onevelocity.pursuit_duration) / sol_normal.pursuit_duration)
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(energy_differences)
    ax1.set_xlabel("Proportional energy differences")
    ax2.hist(pursuit_differences)
    ax2.set_xlabel("Proportional pursuit differences")
    plt.suptitle(fish_vals['species'] + "\nvalues >0 mean normal model costs more")
    plt.tight_layout()
    plt.savefig("/Users/Jason/Desktop/velocity_handling_method_differences_{0}.png".format(fish_vals['species']))
    plt.close()


plt.ioff()
for fish_vals in (chinook_vals, dolly_vals, grayling_vals):
    test_velocity_handling_method(fish_vals)
plt.ion()


# Results from this analysis
# Theres often a 5-20 % reduction in energy cost (more for bigger fish) when using the model that allows lower costs in
# the return straight. This pretty well justifies using both velocities despite the complexity it entails.

# ---------------------------------------------------------------------------------------------------#
#                                                                                                   #
#                                        TESTING VS A PAPER                                         #
#                                                                                                   #
# ---------------------------------------------------------------------------------------------------#


def check_johansen_et_al(display=False, suppress_output=False):
    # Checks maneuver cost predictions against empirical measurements from Johansen et al
    # It looks like they had costs per maneuver of 1.28 mg O2 / kg for freestream swimming and 0.78 for refuge swimming,
    # both at a temperature of 15 C and velocity of 68 cm/s, with rainbow trout of size 33 cm and 423 g. With the width x
    # depth of the tube being 25 x 26 cm, we might assume the average detected prey's lateral distance was around 15 cm or
    # so, and I'll assume prey were detected fairly early (say 30 cm upstream) on average.
    fork_length = 33
    focal_velocity = 68
    prey_velocity = 68
    xd = -30
    yd = 15
    max_thrust = 2.4 * fork_length + 40
    fish_mass = 423  # this input defaults the model to a rainbow trout length-mass regression
    fish_SMR = 0.0  # default to sockeye SMR for given temperature
    fish_NREI = 0.0  # unused in this case
    temperature = 15  # only for SMR
    use_total_cost = False
    disable_wait_time = False
    fish = maneuveringfish.ManeuveringFish(fork_length, focal_velocity, fish_mass, temperature, fish_SMR, max_thrust, fish_NREI, use_total_cost, disable_wait_time)
    detection_point_3D = (xd, yd, 0.0)
    maneuver = optimize_cuckoo.optimal_maneuver_CS(fish, detection_point_3D=detection_point_3D, prey_velocity=prey_velocity, n=25, iterations=1000, suppress_output=suppress_output)
    visualize.summarize_solution(maneuver, display=display, title='Cost Table Check', export_path=None, detailed=True)
    joulesPerMgO2 = 3.36 * 4.184
    costMgO2 = maneuver.energy_cost / joulesPerMgO2
    costMgO2PerKG = costMgO2 / (fish_mass / 1000)
    print("Maneuver cost of locomotion alone is ", costMgO2PerKG, "mg O2 / kg")
    print("Focal swimming cost including SMR in is ", 3600 * (fish.focal_swimming_cost_including_SMR / joulesPerMgO2), " mg O2 / kg / hr")  # convert from focal_swimming_cost_including_SMR in J/s
    print("SMR alone is ", 3600 * (fish.SMR_J_per_s / joulesPerMgO2) / (fish_mass / 1000), " mg O2 / kg / hr")  # convert from focal_swimming_cost_including_SMR in J/s
    print("Locomotion cost for focal swimming for duration of maneuver is ", (maneuver.duration * fish.focal_swimming_cost_of_locomotion / joulesPerMgO2) / (fish_mass / 1000), " mg O2 / kg")
    return maneuver, fish


test_maneuver, test_fish = check_johansen_et_al(display=True)


def check_cost_table_value(fork_length, focal_velocity, prey_velocity, xd, yd, display=False, suppress_output=False):
    max_thrust = 2.4 * fork_length + 40
    fish_mass = 0.0  # this input defaults the model to a rainbow trout length-mass regression
    fish_SMR = 0.0  # unused in this case
    fish_NREI = 0.0  # also unused in this case
    temperature = 10  # also unused in this case
    use_total_cost = False
    disable_wait_time = False
    fish = maneuveringfish.ManeuveringFish(fork_length, focal_velocity, fish_mass, temperature, fish_SMR, max_thrust, fish_NREI, use_total_cost, disable_wait_time)
    detection_point_3D = (xd, yd, 0.0)
    maneuver = optimize_cuckoo.optimal_maneuver_CS(fish, detection_point_3D=detection_point_3D, prey_velocity=prey_velocity, n=25, iterations=1000, suppress_output=suppress_output)
    # visualize.summarize_solution(maneuver, display=display, title='Cost Table Check', export_path=None, detailed=True)
    return maneuver


# This one converges to a best solution despite an acceleration penalty
# I earlier noted, when debugging a crash here: "thrusts keep increasing but fish can't maneuver fast enough"
man = check_cost_table_value(3, 7, 40, -0.1, 0.7, display=True, suppress_output=False)
# This one formerly crashed the remote machine, but now just gives a small acceleration penalty
man = check_cost_table_value(3, 28, 28, -14.082914807729107, 0.2, display=False, suppress_output=False)  # crashed the remote machine

# This is a pretty substantial difference in energy cost based on the arbitrary acceleration threshold.

# With threshold = 100 cm/s^2
# Lowest energy cost after 25000 CS iterations ( 1250026 evaluations) was   0.140523 joules. Mean speed 32.7 cm/s,  1.69 bodylengths/s. Metabolic rate   294.2 mg O2/kg/hr ( 6.7X SMR).
# With threshold = 200 cm/s^2
# Lowest energy cost after 25000 CS iterations ( 1250026 evaluations) was   0.134424 joules. Mean speed 35.2 cm/s,  1.83 bodylengths/s. Metabolic rate   444.3 mg O2/kg/hr ( 9.5X SMR).
# With threshold = 980 cm/s^2 (1G)
# Lowest energy cost after 25000 CS iterations ( 1250026 evaluations) was   0.124169 joules. Mean speed 38.1 cm/s,  1.98 bodylengths/s. Metabolic rate   568.7 mg O2/kg/hr (11.9X SMR).


# best with 1000 iterations was 0.001474
# best with 5000 iterations was 0.001449
fittest = typical_maneuver('Chinook Salmon', iterations=1000)
visualize.summarize_solution(fittest, display=True, title='Typical Chinook', export_path=None, detailed=True)

# for x in np.arange(-5,10,2.5):
#    xorderstr = str(int(1000*x+10000000))
#    fittest_solution = optimize_cuckoo.optimal_maneuver_CS(fish, detection_point_3D=(-25, 25, 0), n = 25, iterations = 500, suppress_output = False, label="")
#    #fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D=(-25, 25, 0), iterations = 3500, use_starting_iterations = True, num_starting_populations = 10, num_starting_iterations = 500)
#    visualize.summarize_solution(fittest_solution, display = False, title = "Test Solution", export_path = "/Users/Jason/Desktop/TestManeuvers/TestDir2Solution{0} (x={1:.3f}).pdf".format(xorderstr, x))

# print(fittest_solution.fitness)
# fittest_solution.dir2 = 1
# fittest_solution.calculate_fitness()
# print(fittest_solution.fitness)
# visualize.summarize_solution(fittest_solution, display = True, title = "Test Solution at x={0}".format(xstr), export_path = None)
#
# fittest_solution = optimize_cuckoo.optimal_maneuver_CS(typical_fish['Dolly Varden'], detection_point_3D = (10, 0, 100), prey_velocity = 100.0, iterations=500)
# #
# #visualize.summarize_solution(fittest_solution, display = False, title = "Test Solution at x={0}".format(xstr), export_path = "/Users/Jason/Desktop/TestManeuvers/TestDir2Solution{0}.pdf".format(xstr))
#
# print("Fittest solution's jerk penalty was {0}".format(fittest_solution.dynamics.jerk_penalty)) # I WANT THIS TO ALWAYS BE ZERO
#
# dv = fittest_solution.mean_water_velocity - fish.focal_velocity
#
# visualize.summarize_solution(fittest_solution, display = True, title = "Test Solution DV={0}".format(dv), export_path = "/Users/Jason/Desktop/TestManeuvers/GraylingSurfaceTestDV_{0}.pdf".format(dv), detailed=True)
#
# # With the current example I've got a DV of 30.... upping that even more.

# Grayling surface maneuver at DV = 45
# Lowest energy cost after 5000 CS iterations ( 1000102 evaluations) was  13.459550 joules. Mean speed 77.1 cm/s,  1.60 bodylengths/s. Metabolic rate   198.6 mg O2/kg/hr ( 6.7X SMR).
# ----------------------------------------------------------------------
# Dynamics of segment pursuit turn      of length  12.71: thrust=0.138, duration=0.270, final_speed= 35.197,      cost:   0.00007.
# Dynamics of segment pursuit straight  of length  91.98: thrust=71.760, duration=1.367, final_speed= 71.757,      cost:   4.08477.
# Dynamics of segment return turn       of length  22.56: thrust=0.100, duration=0.569, final_speed= 24.165,      cost:   0.00009.
# Dynamics of segment return straight   of length 103.19: thrust=55.343, duration=1.988, final_speed= 55.342,      cost:   3.83252.
# Dynamics of segment final turn        of length  12.39: thrust=0.107, duration=0.306, final_speed= 30.442,      cost:   0.00005.
# Dynamics of segment final straight    of length 842.34: duration_a  =6.295, duration_b  =3.283, thrusts=(101.62,  64.98), cost:   5.54205.
# ----------------------------------------------------------------------
# Energy cost = 15.830 J (13.460 J without SMR, 19.103 J w/opportunity cost of  3.273 J).
# Total duration = 14.078265 s (wait time 0.000 s, pursuit 1.637 s). Traveled distance 1085.2 cm at mean speed 77.1 cm/s

# Grayling surface maneuver at DV = 22.5
# Lowest energy cost after 5000 CS iterations ( 1000102 evaluations) was  16.769673 joules. Mean speed 85.5 cm/s,  1.77 bodylengths/s. Metabolic rate   458.6 mg O2/kg/hr (14.1X SMR).
# ----------------------------------------------------------------------
# Dynamics of segment pursuit turn      of length  12.71: thrust=0.122, duration=0.270, final_speed= 35.197,      cost:   0.00006.
# Dynamics of segment pursuit straight  of length  91.98: thrust=71.241, duration=1.376, final_speed= 71.237,      cost:   4.06284.
# Dynamics of segment return turn       of length  20.93: thrust=0.100, duration=0.508, final_speed= 25.955,      cost:   0.00008.
# Dynamics of segment return straight   of length 111.49: thrust=85.971, duration=1.400, final_speed= 85.970,      cost:   5.65023.
# Dynamics of segment final turn        of length  22.41: thrust=64.035, duration=0.329, final_speed= 59.261,      cost:   1.05342.
# Dynamics of segment final straight    of length 389.72: duration_a  =2.585, duration_b  =1.127, thrusts=(122.42,  60.94), cost:   6.00304.
# ----------------------------------------------------------------------
# Energy cost = 18.048 J (16.770 J without SMR, 21.340 J w/opportunity cost of  3.292 J).
# Total duration = 7.595561 s (wait time 0.000 s, pursuit 1.646 s). Traveled distance  649.2 cm at mean speed 85.5 cm/s

# Grayling surface maneuver at DV = 0
# Lowest energy cost after 5000 CS iterations ( 1000102 evaluations) was  19.270672 joules. Mean speed 88.1 cm/s,  1.83 bodylengths/s. Metabolic rate   630.8 mg O2/kg/hr (19.0X SMR).
# ----------------------------------------------------------------------
# Dynamics of segment pursuit turn      of length  12.71: thrust=0.163, duration=0.270, final_speed= 35.197,      cost:   0.00010.
# Dynamics of segment pursuit straight  of length  91.98: thrust=69.287, duration=1.412, final_speed= 69.284,      cost:   3.97966.
# Dynamics of segment return turn       of length  19.84: thrust=0.250, duration=0.480, final_speed= 26.597,      cost:   0.00034.
# Dynamics of segment return straight   of length 118.91: thrust=108.783, duration=1.183, final_speed=108.783,      cost:   7.03012.
# Dynamics of segment final turn        of length  34.90: thrust=120.896, duration=0.310, final_speed=113.780,      cost:   2.41142.
# Dynamics of segment final straight    of length 280.69: duration_a  =1.523, duration_b  =1.168, thrusts=(124.21,  61.26), cost:   5.84904.
# ----------------------------------------------------------------------
# Energy cost = 20.339 J (19.271 J without SMR, 23.703 J w/opportunity cost of  3.364 J).
# Total duration = 6.346053 s (wait time 0.000 s, pursuit 1.682 s). Traveled distance  559.0 cm at mean speed 88.1 cm/s


# Compare vs Puckett & Dill 1984
# They also estimated a cost-per-maneuver of 5.7 mgO2/kg for maneuvers covering a distance averaging 18.8 cm in 0.48 s at an average speed of 43.5 cm/s (in water coordinates) or 39 cm/s (in ground coordinates).
# This comes out to 5.7 mgO2/kg = 0.00684 mgO2 for a 1.2 g fish = 0.0927 J per maneuver.
# It seems there's no way to get the fish to swim anywhere near that fast to catch the prey at their treatment velocity of 4.5 cm/s: the fish just wasn't maneuvering optimally, but
# burning energy in a suboptimal burst for kicks. Trying a higher water velocity to better approximate their swimming speed and corresponding cost.
# pdFish = maneuveringfish.ManeuveringFish(fork_length = 4.9, water_velocity = 25, mass = 1.2, temperature = 15.0, SMR = 0, max_thrust = 250, NREI = 2.0, use_total_cost = False, disable_wait_time = True, treat_final_straight_as_steady_swimming = False)
# fittest_solution = optimize_cuckoo.optimal_maneuver_CS(pdFish, detection_point_3D=(-1, 7, 0), n = 25, iterations = 500, suppress_output = False, label="")
# print(""""\n Total path length: {0}\n
#           Duration: {1}\n
#   Pursuit duration: {2}\n
#         Mean speed: {3}\n
#    Locomotion cost: {4}\n
#  Energy cost w/SMR: {5}\n""".format(fittest_solution.path.total_length,fittest_solution.duration,fittest_solution.pursuit_duration,fittest_solution.dynamics.mean_speed,fittest_solution.energy_cost,pdFish.maneuver_energy_cost_including_SMR(fittest_solution)))
# visualize.summarize_solution(fittest_solution, display = True, title = "Attempt to replicate Puckett & Dill", export_path = None, detailed=True)
# Lowest energy cost after 500 CS iterations (   50052 evaluations) was   0.004516 joules. Mean speed 45.5 cm/s,  8.70 bodylengths/s. Metabolic rate  1986.5 mg O2/kg/hr (15.7X SMR).
#  Total path length: 22.06541118942081
#           Duration: 0.48509312777545743
#   Pursuit duration: 0.16094115146766969
#         Mean speed: 45.48695894870432
#    Locomotion cost: 0.004515630482284048
#  Energy cost w/SMR: 0.00482213810076953
# This result gets kind of close to swimming the same speed for the same amount of time, and the energy cost is about 20X below Puckett & Dill's estimate.


# MORNING

# Make the 'detailed' plots work
# -- axis scaling on bottom plots

# Catch Jacobian determinants of 0 more gracefully
# Figure out actual max thrusts for my species

# Start running the new maneuver for a real fish, preferably multiple times per maneuver, and see how much results vary per maneuver with a reasonable computation time
# Also see how often the fish are predicted to use unrealistically brief accelerations outside the range used by real fish, and consider killing those by using only one thrust per segment
# Look into Cuckoo algorithm variants and whether any of them can be implemented reasonably quickly and/or current one can have params tweaked for faster convergence
# Look at implementing an extra cost penalty for burst swimming like that one paper I saw suggested, if fish still burst too much
# Look at taking out coasting because the real fish don't really do it, but model fish do it all the time
#
# Start finding SMR equations for my species
# In the paper, look at calculating what activity multipliers for WI bioenergetics (ACT) would be for my field fish excluding aggression etc
#
# Look at Rand & Hinch 1998 for the method of partitioning aerobic and anaerobic pathways and rationale for taxing anaerobic pathways at 15 % extra
#
# Add water velocity printout to detailed maneuvers

# Look into whether I could figure out an empirical correction factor to drag in the final straight to make effort comparable to foraging across a velocity gradient
# and returning at the (lower) focal velocity, and use that instead of treating it as steady swimming... if it's a single variable that plain adds/subtracts the same
# amount to the average than maybe it could be the third dimension of the interpolation tables instead of temperature

# What if I limit how quickly thrusts can change by saying any given thrust must be within
# say 25 % of the previous one or something, to assume that's more gradual?
# Then assign values from proportions on that basis. Keep the unrealistic short-term thrust changes
# to a minimum.


# Plot ideas: 
# - Build an option into the individual maneuver plotting code to put the data for real maneuvers (both capture position and capture time)

# Now let's do comparisons for 5x as many fEvals to see what converges the best when given lots of resources
# most_fittest_solution = None
# for i in range(10):
#    fittest_solution = optimize.optimal_maneuver(fish, iterations = 50000, use_starting_iterations = True, num_starting_populations = 100, num_starting_iterations = 6000)
#    if most_fittest_solution is None or fittest_solution.fitness > most_fittest_solution.fitness:
#        most_fittest_solution = fittest_solution
##    fittest_solution = optimize_greywolf.optimal_maneuver_GWO(fish, n_wolves = 10, max_iterations = 50000, suppress_output = False, label="")
##    #fittest_solution = optimize_mothflame.optimal_maneuver_MFO(fish, n_moths = 50, max_iterations = 2000, suppress_output = False, label="")
##    #fittest_solution = optimize_whale.optimal_maneuver_WOA(fish, n_whales = 50, max_iterations = 10000, suppress_output = False, label="")
##    #fittest_solution = optimize_particleswarm.optimal_maneuver_PSO(fish, n_particles = 200, max_iterations = 500, suppress_output = False, label="")
#    fittest_solution = optimize_cuckoo.optimal_maneuver_CS(fish, n = 200, max_iterations = 5000, suppress_output = False, label="")
#    if fittest_solution.fitness > most_fittest_solution.fitness:
#        most_fittest_solution = fittest_solution


# Really relaly high number of iterations now for the best 2 algos
# Lowest energy cost after 100x6000+50000 uGA iterations ( 3819150 evaluations) was   0.153059 joules (origin: mutation ). Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    70.7 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 5000 CS iterations ( 4000402 evaluations) was   0.153218 joules. Mean speed 29.4 cm/s,  1.10 bodylengths/s. Metabolic rate    74.7 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 100x6000+50000 uGA iterations ( 3819150 evaluations) was   0.153490 joules (origin: mutation ). Mean speed 29.2 cm/s,  1.09 bodylengths/s. Metabolic rate    72.7 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 5000 CS iterations ( 4000402 evaluations) was   0.153195 joules. Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    71.3 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 100x6000+50000 uGA iterations ( 3819150 evaluations) was   0.153678 joules (origin: mutation ). Mean speed 28.8 cm/s,  1.07 bodylengths/s. Metabolic rate    68.2 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 5000 CS iterations ( 4000402 evaluations) was   0.153911 joules. Mean speed 29.6 cm/s,  1.10 bodylengths/s. Metabolic rate    76.7 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 100x6000+50000 uGA iterations ( 3819150 evaluations) was   0.154309 joules (origin: crossover). Mean speed 29.2 cm/s,  1.09 bodylengths/s. Metabolic rate    73.1 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 5000 CS iterations ( 4000402 evaluations) was   0.154122 joules. Mean speed 28.7 cm/s,  1.07 bodylengths/s. Metabolic rate    67.6 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 100x6000+50000 uGA iterations ( 3819150 evaluations) was   0.153557 joules (origin: mutation ). Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    70.2 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 5000 CS iterations ( 4000402 evaluations) was   0.153021 joules. Mean speed 29.2 cm/s,  1.09 bodylengths/s. Metabolic rate    72.5 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 100x6000+50000 uGA iterations ( 3819150 evaluations) was   0.153142 joules (origin: mutation ). Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    69.3 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 5000 CS iterations ( 4000402 evaluations) was   0.153359 joules. Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    69.1 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 100x6000+50000 uGA iterations ( 3819150 evaluations) was   0.154261 joules (origin: crossover). Mean speed 28.4 cm/s,  1.06 bodylengths/s. Metabolic rate    64.6 mg O2/kg/hr ( 2.1X SMR).
# Lowest energy cost after 5000 CS iterations ( 4000402 evaluations) was   0.153705 joules. Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    72.1 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 100x6000+50000 uGA iterations ( 3819150 evaluations) was   0.153207 joules (origin: crossover). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    71.4 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 5000 CS iterations ( 4000402 evaluations) was   0.153590 joules. Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    71.9 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 100x6000+50000 uGA iterations ( 3819150 evaluations) was   0.153401 joules (origin: mutation ). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    71.6 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 5000 CS iterations ( 4000402 evaluations) was   0.152986 joules. Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    69.3 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 100x6000+50000 uGA iterations ( 3819150 evaluations) was   0.153263 joules (origin: crossover). Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    69.0 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 5000 CS iterations ( 4000402 evaluations) was   0.153088 joules. Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    69.1 mg O2/kg/hr ( 2.2X SMR).

# And the uGA cranked up like crazy
# Lowest energy cost after 35x3000+50000 uGA iterations (  910800 evaluations) was   0.153796 joules (origin: crossover). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    71.6 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 35x3000+50000 uGA iterations (  910800 evaluations) was   0.154791 joules (origin: mutation ). Mean speed 28.8 cm/s,  1.07 bodylengths/s. Metabolic rate    68.5 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 35x3000+50000 uGA iterations (  910800 evaluations) was   0.153668 joules (origin: mutation ). Mean speed 28.8 cm/s,  1.08 bodylengths/s. Metabolic rate    69.1 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 35x3000+50000 uGA iterations (  910800 evaluations) was   0.154161 joules (origin: mutation ). Mean speed 29.7 cm/s,  1.11 bodylengths/s. Metabolic rate    78.0 mg O2/kg/hr ( 2.4X SMR).
# Lowest energy cost after 35x3000+50000 uGA iterations (  910800 evaluations) was   0.155813 joules (origin: mutation ). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    72.6 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 50x3500+50000 uGA iterations ( 1322150 evaluations) was   0.154502 joules (origin: mutation ). Mean speed 29.5 cm/s,  1.10 bodylengths/s. Metabolic rate    76.4 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 50x3500+50000 uGA iterations ( 1322150 evaluations) was   0.155007 joules (origin: mutation ). Mean speed 28.8 cm/s,  1.08 bodylengths/s. Metabolic rate    68.9 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 50x3500+50000 uGA iterations ( 1322150 evaluations) was   0.153035 joules (origin: mutation ). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    71.4 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 50x3500+50000 uGA iterations ( 1322150 evaluations) was   0.154022 joules (origin: mutation ). Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    70.6 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 50x3500+50000 uGA iterations ( 1322150 evaluations) was   0.154236 joules (origin: mutation ). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    72.0 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 50x3500+50000 uGA iterations ( 1322150 evaluations) was   0.161230 joules (origin: mutation ). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    75.8 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 50x3500+50000 uGA iterations ( 1322150 evaluations) was   0.153064 joules (origin: crossover). Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    70.3 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 50x3500+50000 uGA iterations ( 1322150 evaluations) was   0.154483 joules (origin: mutation ). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    72.1 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 50x3500+50000 uGA iterations ( 1322150 evaluations) was   0.153362 joules (origin: mutation ). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    71.9 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 50x3500+50000 uGA iterations ( 1322150 evaluations) was   0.153935 joules (origin: mutation ). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    71.8 mg O2/kg/hr ( 2.3X SMR).

# With 3000 iterations and 100 cuckoos (slightly more iterations though)
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.153563 joules. Mean speed 29.3 cm/s,  1.09 bodylengths/s. Metabolic rate    73.6 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.160842 joules. Mean speed 29.0 cm/s,  1.09 bodylengths/s. Metabolic rate    74.7 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.153960 joules. Mean speed 28.6 cm/s,  1.07 bodylengths/s. Metabolic rate    66.6 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.152847 joules. Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    71.2 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.160870 joules. Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    74.5 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.157701 joules. Mean speed 29.5 cm/s,  1.10 bodylengths/s. Metabolic rate    77.9 mg O2/kg/hr ( 2.4X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.156739 joules. Mean speed 29.3 cm/s,  1.09 bodylengths/s. Metabolic rate    74.9 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.157594 joules. Mean speed 29.6 cm/s,  1.10 bodylengths/s. Metabolic rate    77.9 mg O2/kg/hr ( 2.4X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.154126 joules. Mean speed 29.3 cm/s,  1.09 bodylengths/s. Metabolic rate    73.6 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.157262 joules. Mean speed 28.2 cm/s,  1.05 bodylengths/s. Metabolic rate    63.6 mg O2/kg/hr ( 2.1X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.159536 joules. Mean speed 29.4 cm/s,  1.10 bodylengths/s. Metabolic rate    77.3 mg O2/kg/hr ( 2.4X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.153905 joules. Mean speed 28.8 cm/s,  1.08 bodylengths/s. Metabolic rate    69.3 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.154192 joules. Mean speed 29.2 cm/s,  1.09 bodylengths/s. Metabolic rate    73.1 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.158547 joules. Mean speed 28.1 cm/s,  1.05 bodylengths/s. Metabolic rate    62.7 mg O2/kg/hr ( 2.1X SMR).
# Lowest energy cost after 3000 CS iterations ( 1200202 evaluations) was   0.157241 joules. Mean speed 28.0 cm/s,  1.05 bodylengths/s. Metabolic rate    60.9 mg O2/kg/hr ( 2.1X SMR).

# With 50 whales
# Lowest energy cost after 10000 WOA iterations ( 1000002 evaluations) was   0.186541 joules. Mean speed 29.3 cm/s,  1.10 bodylengths/s. Metabolic rate    89.8 mg O2/kg/hr ( 2.6X SMR).
# Lowest energy cost after 10000 WOA iterations ( 1000002 evaluations) was   0.162208 joules. Mean speed 28.8 cm/s,  1.08 bodylengths/s. Metabolic rate    73.0 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 10000 WOA iterations ( 1000002 evaluations) was   0.163989 joules. Mean speed 29.4 cm/s,  1.10 bodylengths/s. Metabolic rate    79.9 mg O2/kg/hr ( 2.4X SMR).
# Lowest energy cost after 10000 WOA iterations ( 1000002 evaluations) was   0.235465 joules. Mean speed 26.3 cm/s,  0.98 bodylengths/s. Metabolic rate    50.5 mg O2/kg/hr ( 1.9X SMR).
# Lowest energy cost after 10000 WOA iterations ( 1000002 evaluations) was   0.198635 joules. Mean speed 26.9 cm/s,  1.00 bodylengths/s. Metabolic rate    54.1 mg O2/kg/hr ( 1.9X SMR).

# With 100,000 iterations of 5 wolves
# Lowest energy cost after 100000 GWO iterations ( 1000002 evaluations) was   0.161198 joules. Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    73.0 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 100000 GWO iterations ( 1000002 evaluations) was   0.182621 joules. Mean speed 26.9 cm/s,  1.00 bodylengths/s. Metabolic rate    48.5 mg O2/kg/hr ( 1.9X SMR).
# Lowest energy cost after 100000 GWO iterations ( 1000002 evaluations) was   0.182501 joules. Mean speed 26.8 cm/s,  1.00 bodylengths/s. Metabolic rate    47.6 mg O2/kg/hr ( 1.8X SMR).
# Lowest energy cost after 100000 GWO iterations ( 1000002 evaluations) was   0.182566 joules. Mean speed 26.7 cm/s,  1.00 bodylengths/s. Metabolic rate    45.4 mg O2/kg/hr ( 1.8X SMR).
# Lowest energy cost after 100000 GWO iterations ( 1000002 evaluations) was   0.182447 joules. Mean speed 26.8 cm/s,  1.00 bodylengths/s. Metabolic rate    47.9 mg O2/kg/hr ( 1.8X SMR).

# With 50,000 iterations of 10 wolves
# Lowest energy cost after 50000 GWO iterations ( 1000002 evaluations) was   0.161801 joules. Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    75.5 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 50000 GWO iterations ( 1000002 evaluations) was   0.161854 joules. Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    75.1 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 50000 GWO iterations ( 1000002 evaluations) was   0.170635 joules. Mean speed 26.9 cm/s,  1.00 bodylengths/s. Metabolic rate    46.3 mg O2/kg/hr ( 1.8X SMR).
# Lowest energy cost after 50000 GWO iterations ( 1000002 evaluations) was   0.170715 joules. Mean speed 26.7 cm/s,  1.00 bodylengths/s. Metabolic rate    42.8 mg O2/kg/hr ( 1.8X SMR).
# Lowest energy cost after 50000 GWO iterations ( 1000002 evaluations) was   0.161077 joules. Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    74.9 mg O2/kg/hr ( 2.3X SMR).


# for i in range(5):
#    fittest_solution = optimize.optimal_maneuver(fish, iterations = 11529, use_starting_iterations = True, num_starting_populations = 15, num_starting_iterations = 1500)
# fittest_solution = optimize_greywolf.optimal_maneuver_GWO(fish, n_wolves = 5, max_iterations = 20000, suppress_output = False, label="")
# fittest_solution = optimize_mothflame.optimal_maneuver_MFO(fish, n_moths = 50, max_iterations = 2000, suppress_output = False, label="")
# fittest_solution = optimize_whale.optimal_maneuver_WOA(fish, n_whales = 10, max_iterations = 10000, suppress_output = False, label="")
# fittest_solution = optimize_cuckoo.optimal_maneuver_CS(fish, n = 50, max_iterations = 1000, suppress_output = False, label="")
# fittest_solution = optimize_bat.optimal_maneuver_BAT(fish, n = 50, max_iterations = 2000, suppress_output = False, label="")
# fittest_solution = optimize_salp.optimal_maneuver_SSA(fish, n = 50, max_iterations = 2000, suppress_output = False, label="")
# fittest_solution = optimize_particleswarm.optimal_maneuver_PSO(fish, n_particles = 200, max_iterations = 500, suppress_output = False, label="")

# Now after reducing thrusts to 2 per stage...
# Cuckoo version
# Lowest energy cost after 1000 CS iterations (  200102 evaluations) was   0.161614 joules. Mean speed 29.3 cm/s,  1.10 bodylengths/s. Metabolic rate    78.0 mg O2/kg/hr ( 2.4X SMR).
# Lowest energy cost after 1000 CS iterations (  200102 evaluations) was   0.161204 joules. Mean speed 29.0 cm/s,  1.09 bodylengths/s. Metabolic rate    74.8 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 1000 CS iterations (  200102 evaluations) was   0.162049 joules. Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    75.9 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 1000 CS iterations (  200102 evaluations) was   0.161370 joules. Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    73.4 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 1000 CS iterations (  200102 evaluations) was   0.161062 joules. Mean speed 29.3 cm/s,  1.09 bodylengths/s. Metabolic rate    77.4 mg O2/kg/hr ( 2.4X SMR).

# uGA version
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.163797 joules (origin: crossover). Mean speed 28.6 cm/s,  1.07 bodylengths/s. Metabolic rate    70.2 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.162402 joules (origin: mutation ). Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    73.8 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.162341 joules (origin: mutation ). Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    75.0 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.163816 joules (origin: mutation ). Mean speed 28.3 cm/s,  1.06 bodylengths/s. Metabolic rate    66.4 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.161468 joules (origin: crossover). Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    74.6 mg O2/kg/hr ( 2.3X SMR).


# visualize.summarize_solution(fittest_solution, display = True, title = "Test Solution", export_path = None)

# Not half bad performance from Cuckoo Search! Best yet other than the GA...
# Lowest energy cost after 1000 CS iterations (  200102 evaluations) was   0.158608 joules. Mean speed 28.5 cm/s,  1.06 bodylengths/s. Metabolic rate    67.3 mg O2/kg/hr ( 2.2X SMR).
# Lowest energy cost after 1000 CS iterations (  200102 evaluations) was   0.164320 joules. Mean speed 28.1 cm/s,  1.05 bodylengths/s. Metabolic rate    64.7 mg O2/kg/hr ( 2.1X SMR).
# Lowest energy cost after 1000 CS iterations (  200102 evaluations) was   0.160832 joules. Mean speed 29.2 cm/s,  1.09 bodylengths/s. Metabolic rate    76.6 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 1000 CS iterations (  200102 evaluations) was   0.161201 joules. Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    74.3 mg O2/kg/hr ( 2.3X SMR).
# Lowest energy cost after 1000 CS iterations (  200102 evaluations) was   0.161421 joules. Mean speed 28.8 cm/s,  1.08 bodylengths/s. Metabolic rate    72.3 mg O2/kg/hr ( 2.3X SMR).

# most_fittest_solution = None
# for i in range(300):
#    print("Computing solution {0}".format(i))
#    fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (-15, 5, 7), iterations = 10000, use_starting_iterations = True, num_starting_populations = 10, num_starting_iterations = 1000)
#    visualize.summarize_solution(fittest_solution, display = False, title = "Test Solution", export_path = "/Users/Jason/Desktop/TestManeuvers/FitSolutions{0}_{1}.pdf".format(i,fittest_solution.dynamics.energy_cost))
#    if most_fittest_solution is None or fittest_solution.fitness > most_fittest_solution.fitness:
#        most_fittest_solution = fittest_solution
#
# visualize.summarize_solution(most_fittest_solution, display = False, title = "Test Solution", export_path = "/Users/Jason/Desktop/TestManeuvers/_MostFitSolution.pdf".format(i))

# sol = solution.test_solution(fish)
# for i in range(40):
#    test_solution = solution.convergent_random_solution(fish, -10.0, 10.0)
#    path.print_path(test_solution.path)
#    visualize.summarize_solution(test_solution, display = False, title = "Test Solution", export_path = "/Users/Jason/Desktop/TestManeuvers/TestManeuver{0}.pdf".format(i))


# def time_function(function, nruns = 5, *args, **kwargs):
#    results = np.zeros(nruns)
#    times = np.zeros(nruns)
#    durations = np.zeros(nruns)
#    for i in range(nruns):
#        start = time.clock()
#        result = function(*args, **kwargs)
#        end = time.clock()
#        results[i] = result.fitness
#        durations[i] = result.duration
#        times[i] = end - start
#        popsize = kwargs.get('n', kwargs.get('popsize', 0))
#        fname = function.__name__
#    return (nruns,kwargs['iterations'],popsize,fname,results,times,durations)
# same_nruns = 30
# dummy_solution = optimize.optimal_maneuver(fish, iterations = 10, suppress_output = True) # run once quick just to precompile before timing
# #(nruns,iterations,popsize,fname,fitnesses,times,durations) = time_function(optimize.optimal_maneuver, fish = fish, nruns = same_nruns, iterations=6000, popsize=4,  num_starting_populations = 10, num_starting_iterations = 600)
# (nruns,iterations,popsize,fname,fitnesses,times,durations) = time_function(optimize_cuckoo.optimal_maneuver_CS, fish = fish, nruns = same_nruns, n = 20, iterations=440)
# print("Out of {0} {1}-iteration, n={11} tests ({12}) completed in {2:.5f} s/run, fitnesses were {3:.5f} +/- {4:.5f} (range {5:.5f} to {6:.5f}) in durations {7:.5f} +/- {8:.5f} sd (range {9:.5f} to {10:.5f})".format(
#     nruns, iterations, times.mean(), fitnesses.mean(), fitnesses.std(), fitnesses.min(), fitnesses.max(),durations.mean(), durations.std(), durations.min(), durations.max(), popsize, fname))

# Quick test with uGA -- uses 17630 fEvals each
# Out of 1000x 3000-iteration tests completed in 0.49110 s/run, fitnesses were -0.17584 +/- 0.01406 (range -0.29512 to -0.16106) in durations 3.42225 +/- 0.89778 sd (range 2.13685 to 5.56422)
# Now let's try Cuckoo -- uses 17612 fEvals eachs
# Out of 1000x 293-iteration tests completed in 0.86321 s/run, fitnesses were -0.18727 +/- 0.01668 (range -0.29319 to -0.16094) in durations 3.61079 +/- 0.97643 sd (range 1.87346 to 6.60077)
# For a given number of fEvals, Cuckoo search generally performed worse than uGA and took longer... although it hasn't been tuned to this problem at all.

# Out of 30 3000-iteration, n=4 tests (optimal_maneuver) completed in 0.66470 s/run, fitnesses were -0.62058 +/- 0.00672 (range -0.64273 to -0.61256) in durations 0.85485 +/- 0.04670 sd (range 0.78892 to 0.96389)
# Out of 30 293-iteration, n=15 tests (optimal_maneuver_CS) completed in 0.90668 s/run, fitnesses were -0.61948 +/- 0.00730 (range -0.64517 to -0.61182) in durations 0.87778 +/- 0.10940 sd (range 0.77548 to 1.23971)
# Similar convergence per iteration but faster iterations with uGA than Cuckoo when doing really quick maneuvers that shouldn't necessarily converge well.

# Out of 30 6000-iteration, n=4 tests (optimal_maneuver) completed in 2.09538 s/run, fitnesses were -0.61816 +/- 0.00199 (range -0.62250 to -0.61451) in durations 0.84275 +/- 0.02846 sd (range 0.78876 to 0.91063)
# Out of 30 440-iteration, n=20 tests (optimal_maneuver_CS) completed in 1.37649 s/run, fitnesses were -0.61454 +/- 0.00300 (range -0.62234 to -0.61120) in durations 0.83466 +/- 0.02663 sd (range 0.76504 to 0.90574)
# Cuckoo seems to better leverage a similar # of fEvals here for beter convergence....


# test_solution = solution.convergent_random_solution(fish, -10.0, 10.0)
##visualize.summarize_solution(test_solution, display = True)
# print(test_solution.proportions())


# for i in range(1000000):
#    print("SOLUTION NUMBER {0}".format(i))
#    test_solution = solution.random_solution(fish, -10.0, 10.0)
#    if not np.isfinite(test_solution.dynamics.duration):
#        break
# print("Finished after {0} iterations.".format(i+1))

# It seems all the ua convergence failures are with ut3 < v < ua
# Until just now, a failure with ut3=44 and ua=27

# Every one of my "divide by zero" errors comes when min_t_a > max_t_a... and typically after a series of slowdowns with low durations to begin with and therefore small increments.
# Perhaps I am not slowing things quite enough?
# Every time this problem rears its head, we have ut3 < v < ua. Not sure if it's caused by ut3 < v, or v < ua, or both of these conditions.

# for i in range(10000):
#    solution.mutate_solution(test_solution, 0.1)

# visualize.summarize_solution(test_solution, display = True, title = "Test Solution", export_path = "/Users/Jason/Desktop/TestManeuver.pdf")
# test_solution.dynamics.final_thrust # THIS IS COMING OUT WAY TOO HIGH BUT LETS FIND OTHER PROBLEM FIRST
# test_solution.dynamics.energy_cost


# x_max = 45.0
# inc = 2.5
# y_max = 30.0
#
# x_range = np.linspace(-x_max,x_max,2*x_max/inc)
# y_range = np.linspace(-y_max,y_max,2*y_max/inc)
# durations = np.zeros((x_range.size, y_range.size))
# costs = np.zeros((x_range.size, y_range.size))
# mid_y = y_range.size / 2
#
# from scipy import interpolate
#
# for i in range(x_range.size):
#    for j in range(mid_y):
#        x = x_range[i]
#        y = y_range[j] if y_range[j] != 0.0 else 0.000001 # avoid singularities for now
#        sol = optimize.optimal_maneuver(fish, detection_point_3D = (x, y, 0.0), iterations = 2000) 
#        durations[i][j] = sol.duration
#        costs[i][j] = -sol.fitness # energy cost
#        durations[i][(y_range.size-1) - j] = sol.duration # These two rows work to reflect the result across the y-axis
#        costs[i][(y_range.size-1)  - j] = -sol.fitness    # This process works at least when y_range is symmetrical and of even size without 0 in the middle
#
# print durations
# print costs
#
# interp_durations = interpolate.RegularGridInterpolator((x_range,y_range),durations)
# interp_costs = interpolate.RegularGridInterpolator((x_range,y_range),costs)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, facecolor='w',figsize=(16,6))
# ax1.set_aspect('equal')
# ax2.set_aspect('equal')
# plot_x = np.linspace(x_max, -x_max, 100)
# plot_y = np.linspace(y_max, -y_max, 100)
# plot_durations = [[interp_durations((x,y)) for x in plot_x] for y in plot_y]
# plot_costs = [[interp_costs((x,y)) for x in plot_x] for y in plot_y]
#
# ax1.contourf(plot_x, plot_y, plot_durations, cmap='viridis')
# c1 = ax1.contour(plot_x, plot_y, plot_durations, cmap='viridis')
# plt.clabel(c1, inline=1, fontsize=10, colors='k')
# ax1.set_xlabel('detection x (cm)', fontsize=14)
# ax1.set_ylabel('detection y (cm)', fontsize=14)
# ax1.set_title('Duration (s)')
#
# ax2.contourf(plot_x, plot_y, plot_costs, cmap='viridis')
# c2 = ax2.contour(plot_x, plot_y, plot_costs, cmap='viridis')
# plt.clabel(c2, inline=1, fontsize=10, colors='k')
# ax2.set_xlabel('detection x (cm)', fontsize=14)
# ax2.set_ylabel('detection y (cm)', fontsize=14)
# ax2.set_title('Swimming cost (J)')
#
# plt.subplots_adjust(top=0.85) # make room for the overall title
# tempstr = (u"%.1f \xb0C" % fish.temperature)
# plt.suptitle('Model-predicted maneuver costs for a %.1f-cm FL fish in %.1f cm/s current at %s' % (fish.fork_length, fish.focal_current_speed, tempstr), fontsize=16)
#
# plt.tight_layout()
# plt.show()


# Examples for Gary
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (-35.0, 1.0, 1.0), iterations = 5000) # Detected far almost straight in front; demonstrates coasting + waiting
####fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (-35.0, 25.0, 0.0), iterations = 5000) # Detected far diagonally in front 
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (0.0, 25.0, 0.0), iterations = 5000) # Detected far laterally
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (35.0, 25.0, 0.0), iterations = 5000) # Detected far diagonally downstream
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (-10.0, 1.0, 1.0), iterations = 5000) # Detected near almost straight in front
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (-10.0, 10.0, 0.0), iterations = 5000) # Detected near diagonally in front
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (0.0, 10.0, 0.0), iterations = 5000) # Detected near laterally
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (10.0, 10.0, 0.0), iterations = 5000) # Detected near diagonally downstream

# Nice solution to illustrate both wait time and two-stage acceleration
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (-50.0, 15.0, 20.0), iterations = 3000)
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (-30.0, 25.0, 0.0), iterations = 3000)
# fittest_solution = optimize.optimal_maneuver(fish, iterations = 2000)
# visualize.summarize_solution(fittest_solution, display = True, title = "Testing a title", export_path = "/Users/Jason/Desktop/TestManeuver.pdf")


# testing repeatability
# for i in range(50):
#    fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = (-35.0, 25.0, 0.0), iterations = 1000) # Detected far diagonally in front 
#    visualize.summarize_solution(fittest_solution, display = False, title = "Test iteration {0}, cost {1:0.4f} J.".format(i,fittest_solution.dynamics.energy_cost), export_path = "/Users/Jason/Desktop/TestRepeatManeuver/TestIteration{0:2.0f}.pdf".format(i))

# mass = 3000
# length = 0.1 * (mass / 4.8e-06)**(1/3.1)
# detection_point = (-12.0, 12.0, 0.0)
# fish = maneuveringfish.ManeuveringFish(length, 29, 10, mass, None, 0.08, False, 0.1, True)
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = detection_point, iterations = 5000)
# visualize.summarize_solution(fittest_solution, display = True, title = "The problem one")

# mass = 46.2
# length = 17.9
# focal_velocity = 36
# temperature = 8.3
# max_thrust = 94
# fish = maneuveringfish.ManeuveringFish(length, focal_velocity, temperature, mass, max_thrust, 0.0, False, 0.1, True)
# detection_point = (-9.05716327, -4.33530083,  8.38973115)
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = detection_point, iterations = 3000)
# visualize.summarize_solution(fittest_solution, display = True, title = "The problem one", should_print_dynamics = False)
##


# GLITCHY CHINOOK
# mass 0.33 g = 0.08 J cost
# mass 0.32 g = 0.28 J
# mass 0.31 g = 3.08 J
# mass 0.305 g = 26+ J -- all over the map 

# mass = 0.5 # 0.3
# length = 3.2
# focal_velocity = 5
# temperature = 6.3
# max_thrust = 42
# fish = maneuveringfish.ManeuveringFish(length, focal_velocity, temperature, mass, max_thrust, 0.0, False, 0.1, True)
##detection_point = (-3.08423913109, 1.89482285411, 3.45125938101)
##fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = detection_point, iterations = 1000)
##visualize.summarize_solution(fittest_solution, display = True, title = "The problem one", should_print_dynamics = True)
# print("Fish of mass {0:.3f} g has u_ms = {1:.5f}, SMR = {2:.5f}, AMR = {3:.5f}, and focal cost of {4:.5f}.".format(mass, fish.u_ms, fish.SMR, fish.AMR, fish.focal_swimming_cost))

# focal_current_speed = 29
# mass = 45 # default 45
# fork_length = 0.1 * (mass / 4.8e-06)**(1/3.1) # reverse of the dolly varden length-mass regression used in Fish.py
# NREI = 0.08
# focal_return_tolerance = 0.1 # 1 mm
# detection_vector = (-12.0, 12.0, 0.0)
# disable_wait_time = True # want to be false for environmental relationships, true for spatial relationships
# fix_u_ms_at_opt_temp = False
# temperature = 10
# max_thrust = 97
#
# use_starting_iterations = False
# num_starting_populations = 15
# num_starting_iterations = 1000
# num_iterations = 4500
# num_tests = 20
#
# popsize = 4 # initially 6
# variant_scale = 1.5 # initially 10
# mixing_ratio = 3.0 # initially 3.0
#
# fish = maneuveringfish.ManeuveringFish(fork_length, focal_current_speed, temperature, mass, max_thrust, 0.0, False, 0.1, disable_wait_time, fix_u_ms_at_opt_temp)
# fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = detection_vector, popsize = popsize, variant_scale = variant_scale, mixing_ratio = mixing_ratio, iterations = num_iterations, use_starting_iterations = use_starting_iterations, num_starting_populations = num_starting_populations, num_starting_iterations = num_starting_iterations)
# visualize.summarize_solution(fittest_solution, display = True, title = None, should_print_dynamics = True)
#
#
#


# costs = []

# start_time = time.clock()
# for i in range(num_tests):
#    fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = detection_vector, popsize = popsize, variant_scale = variant_scale, mixing_ratio = mixing_ratio, iterations = num_iterations, use_starting_iterations = use_starting_iterations, num_starting_populations = num_starting_populations, num_starting_iterations = num_starting_iterations)
#    costs.append(fittest_solution.dynamics.energy_cost)
# end_time = time.clock()
# mean_time = (end_time - start_time) / num_tests
#
# if use_starting_iterations:
#    print("With {3:6d} iterations and {4:3d} initial solutions refined by {5:6d} iterations, ps = {7}, vs = {8}, mr = {9}, the mean cost from {6:3d} tests averaging {1:5.3f} s was {0:.8f} with CV = {2:.8f}.".format(np.mean(costs),mean_time,np.std(costs)/np.mean(costs),num_iterations,num_starting_populations,num_starting_iterations,num_tests,popsize,variant_scale,mixing_ratio))
# else:
#    print("With {3:6d} iterations and no initial batch, ps = {5}, vs = {6}, mr = {7}, the mean cost from {4:3d} tests averaging {1:5.3f} s was {0:.8f} with CV = {2:.8f}.".format(np.mean(costs),mean_time,np.std(costs)/np.mean(costs),num_iterations,num_tests,popsize,variant_scale,mixing_ratio))
#
# visualize.summarize_solution(fittest_solution, display = True, title = "Wheeee", should_print_dynamics = False)

# With   1000 final iterations and no initial batch, popsize = 6, variant_scale = 0.3, mixing_ratio = 3.0, the mean cost from  20 tests was 0.15549521 with SD 0.00366755 and CV = 0.02358624.
# And only 2/20 found the .144 stratgy
# With   3000 final iterations and no initial batch, popsize = 6, variant_scale = 0.3, mixing_ratio = 3.0, the mean cost from  20 tests was 0.15245495 with SD 0.00578518 and CV = 0.03794682.
# And 7/20 found th .144 strategy

# With   1000 final iterations and no initial batch, popsize = 12, variant_scale = 0.3, mixing_ratio = 3.0, the mean cost from  20 tests was 0.14946362 with SD 0.00589152 and CV = 0.03941778
# This shows that the CV can be VERY misleading, becaus near-unanimous convergnce on a poor solution scores better than a poor/good mix. Mean cost is the number to watch.

# TARGET FITNESS: 0.144382 joules

# Brute forcing the right answer: 
# With  10000 final iterations and   6 initial solutions refined by   3000 iterations each, popsize = 6, variant_scale = 0.3, mixing_ratio = 3.0, the mean cost from  20 tests was 0.14503089 with SD 0.00266680 and CV = 0.01838783.
# Best solution was:
# Lowest energy cost after 6x3000+10000 iterations was   0.144382 joules. Mean speed 35.9 cm/s,  1.89 bodylengths/s. Metabolic rate   602.0 mg O2/kg/hr (10.1X SMR). 
# Most were very similar, but got one bad one:
# Lowest energy cost after 6x3000+10000 iterations was   0.156654 joules. Mean speed 36.6 cm/s,  1.93 bodylengths/s. Metabolic rate   718.8 mg O2/kg/hr (12.0X SMR). 
# So 19/20 found the .144 strategy


# temperatures = np.arange(2,20.0,0.1)
# stats = [maneuveringfish.ManeuveringFish(fork_length, focal_current_speed, temperature, mass, None, NREI, False, focal_return_tolerance, disable_wait_time).focal_swimming_cost for temperature in temperatures]
# import matplotlib.pyplot as plt
# fig = plt.figure(facecolor='w')
# ax = fig.add_subplot(111)
# ax.plot(temperatures,stats)
# ax.set_xlabel("Temperature (C)")
# ax.set_ylabel("Focal swimming cost (J/s)")
# plt.show()

# import matplotlib.pyplot as plt
# test_solution = solution.Solution(fish, 9.555545396139958, 9.554226011252831, 1, np.array([40.29449682,3.44345861,38.37612873,14.76867844,38.46936086,\
#                13.8324925,11.15203851,37.5022379]), 0.0, -60.0, 45.0)
# test_path = path.ManeuverPath(test_solution,3.0)
# fig = plt.figure(facecolor='w')
# ax = fig.add_subplot(111, aspect='equal')
# visualize.plot_water_coords_path(ax, test_path, padding=5)
# plt.show()

# def time_function(function, nruns = 5, *args, **kwargs):
#    results = np.zeros(nruns)
#    times = np.zeros(nruns)
#    durations = np.zeros(nruns)
#    for i in range(nruns):
#        start = time.clock()
#        result = function(*args, **kwargs)
#        end = time.clock()
#        results[i] = result.fitness
#        durations[i] = result.duration
#        times[i] = end - start
#    return (nruns,kwargs['iterations'],results,times,durations)
#    
# dummy_solution = optimize.optimal_maneuver(fish, iterations = 10, suppress_output = True) # run once quick just to precompile before timing
# (nruns,iterations,fitnesses,times,durations) = time_function(optimize.optimal_maneuver, fish = fish, nruns = 100, iterations=3000, use_starting_iterations = False)
# print "Out of %dx %d-iteration tests completed in %.5f s/run, fitnesses were %.5f +/- %.5f (range %.5f to %.5f) in durations %.5f +/- %.5f sd (range %.5f to %.5f)" \
#        % (nruns, iterations, times.mean(), fitnesses.mean(), fitnesses.std(), fitnesses.min(), fitnesses.max(),durations.mean(), durations.std(), durations.min(), durations.max())


# Using timeit to get around this recent numba/cprofile interaction bug: https://github.com/numba/numba/issues/1786

# dummy = optimize.optimal_maneuver(fish, suppress_output = True, iterations = 1000)
# cProfile.run('optimize.optimal_maneuver(fish, iterations = 10000)','runstats')
# p = pstats.Stats('runstats')
# p.strip_dirs().sort_stats('tottime').print_stats(25)
# p.print_callers(25)

# man = fish.optimal_maneuver(fitness_goal="energy_cost",iterations=2000)

# only use timeit for really small self-contained things requiring many thousands or millions of test runs
# ntests = 1000
# def test():
#    x=2
# print timeit.timeit("test()", setup="from __main__ import test", number = ntests)/ntests

# For example, testing the speed of three equivalent formulations of the maneuver cost function varying only in speed. This confirms that 
# the formula we're using, rather than the form given by Hughes & Kelly with logs, etc, faster.
# ntests = 1000
# def test1():
#    return (1/3600.0) * (fish.base_mass/1000.0) * 3.24 * 4.184 * np.exp(np.log(fish.SMR) + 50*((np.log(fish.AMR)-np.log(fish.SMR))/fish.u_ms))
# def test2():
#    return (1/3600.0) * (fish.base_mass/1000.0) * 3.24 * 4.184 * fish.SMR * (fish.AMR / fish.SMR) ** (50 / fish.u_ms)
# def test3():
#    return 0.0000037656 * fish.base_mass *  fish.SMR * (fish.AMR / fish.SMR) ** (50 / fish.u_ms)
#    
# print timeit.timeit("test1()", setup="from __main__ import test1, test2, test3, fish", number = ntests)/ntests
# print timeit.timeit("test2()", setup="from __main__ import test1, test2, test3, fish", number = ntests)/ntests
# print timeit.timeit("test3()", setup="from __main__ import test1, test2, test3, fish", number = ntests)/ntests
