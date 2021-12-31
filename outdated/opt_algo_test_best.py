# Test different optimization algorithms

import numpy as np
import os

from maneuvermodel import maneuveringfish, maneuver
from mealpy.human_based import SARO # Search and Rescue Optimization (Original)
from maneuvermodel import saro_compiled

FIGURE_OUTPUT_PATH = "/Users/Jason/Dropbox/drift model project/Papers/Capture Maneuver Model/Figures/Ancillary Figures 2021/OptAlgoTests/"

typical = {'Chinook Salmon': {'fork_length': 4.6, 'focal_velocity': 10, 'prey_velocity': 11, 'mass': 0.85, 'temperature': 9, 'max_thrust': 62, 'NREI': 0.017, 'detection_distance': 8, 'SMR': 226},
           'Dolly Varden': {'fork_length': 18, 'focal_velocity': 28, 'prey_velocity': 29, 'mass': 51, 'temperature': 10, 'max_thrust': 94, 'NREI': 0.017, 'detection_distance': 17, 'SMR': 52},
           'Arctic Grayling': {'fork_length': 43, 'focal_velocity': 42, 'prey_velocity': 48, 'mass': 920, 'temperature': 6, 'max_thrust': 159, 'NREI': 0.017, 'detection_distance': 35, 'SMR': 40}}
species = 'Dolly Varden'
fish = maneuveringfish.ManeuveringFish(fork_length = typical[species]['fork_length'],
                                       mean_water_velocity= typical[species]['focal_velocity'],
                                       mass = typical[species]['mass'],
                                       temperature = typical[species]['temperature'],
                                       SMR = typical[species]['SMR'],
                                       max_thrust = typical[species]['max_thrust'],
                                       NREI = typical[species]['NREI'],
                                       use_total_cost = False,
                                       disable_wait_time = False)
xd, yd = (-typical[species]['detection_distance'] / 1.414, typical[species]['detection_distance'] / 1.414)
prey_velocity = typical[species]['prey_velocity']

def objective_function(p):
    return maneuver.maneuver_from_proportions(fish, prey_velocity, xd, yd, p).fitness

def problem_description(verbose=True):
    return {
        "obj_func": objective_function,
        "lb": [0, ] * 12,
        "ub": [1, ] * 12,
        "minmax": "max",
        "verbose": verbose,
    }

def process_completion(model, plot=False):
    lowest_energy_cost = -model.solution[1][0]
    model_name = 'Unnamed' if not hasattr(model, 'name') else model.name
    n_function_evals = model.nfe_per_epoch * model.max_iterations
    run_name = "{0} epoch={1} pop_size={2} n_evals={3}".format(model_name, model.max_iterations, model.pop_size, n_function_evals)
    print("{4} : Lowest energy cost was {0} J after {1} fEvals ({2}x{3}).".format(lowest_energy_cost, n_function_evals, model.max_iterations, model.pop_size, model_name))
    if plot: # note I manually edited the code from the package that output figures in both png and pdf and changed to pdf-only
        if not os.path.exists(os.path.join(FIGURE_OUTPUT_PATH, run_name)): os.mkdir(os.path.join(FIGURE_OUTPUT_PATH, run_name))
        model.history.save_global_objectives_chart(filename=os.path.join(FIGURE_OUTPUT_PATH, run_name, "GlobalObjectives {0}".format(run_name)))
        model.history.save_local_objectives_chart(filename=os.path.join(FIGURE_OUTPUT_PATH, run_name, "LocalObjectives {0}".format(run_name)))
        model.history.save_global_best_fitness_chart(filename=os.path.join(FIGURE_OUTPUT_PATH, run_name, "GlobalBestFitness {0}".format(run_name)))
        model.history.save_local_best_fitness_chart(filename=os.path.join(FIGURE_OUTPUT_PATH, run_name, "GlobalBestFitness {0}".format(run_name)))
        model.history.save_runtime_chart(filename=os.path.join(FIGURE_OUTPUT_PATH, run_name, "RunTime {0}".format(run_name)))
        model.history.save_exploration_exploitation_chart(filename=os.path.join(FIGURE_OUTPUT_PATH, run_name, "Exploitation {0}".format(run_name)))
        model.history.save_diversity_chart(filename=os.path.join(FIGURE_OUTPUT_PATH, run_name, "Diversity {0}".format(run_name)))
        model.history.save_trajectory_chart(list_agent_idx=[3, 5], list_dimensions=[3], filename=os.path.join(FIGURE_OUTPUT_PATH, run_name, "Trajectory {0}".format(run_name)))

# BENCHMARK: Previous best with 15 million function evaluations was 0.146549 joules
# Best current value possible is 0.14628342129912647 with 4.5 million evaluations from Self Adaptive Differential Evolution
# SADE at one point found -0.146466 with only 100k evals, beating my old Cuckoo algo's 15 million
# SADE found 0.14628342129912647 J after 4500000 evals.
# SARO found 0.1462834105496333 J after 1500000 evals

from mealpy.bio_based import SMA # Slime Mold Algorithm
from mealpy.evolutionary_based import DE # Self-Adaptive Differential Evolution
from mealpy.human_based import QSA # Queuing Search Algorithm (Improved)
from mealpy.human_based import TLO # Teaching-Learning-based Optimization (original)
from mealpy.physics_based import EO # Equilibrium Optimization
from mealpy.system_based import GCO # Germinal Center Optimization
from mealpy.swarm_based import MFO  # Moth Flame Optimizer
true_best_value = None

import matplotlib.pyplot as plt
def compare_multiple_models(comparison_label="Comparison", models=()):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for model in models:
        global_best = -np.array(model.history.list_global_best_fit)
        function_evals = np.arange(0, model.nfe_per_epoch * (model.max_iterations + 1), model.nfe_per_epoch)
        plt.plot(function_evals, global_best, axes=ax, label=model.name)  # Plot some data on the (implicit) axes.
        ax.set_yscale('log')
        plt.legend()
    ax.set_xlabel("Objective function evaluations")
    ax.set_ylabel("Maneuver cost (J)")
    ax.set_title(comparison_label)
    if true_best_value is not None:
        ax.axhline(y=true_best_value, ls='dotted', color='0.7', label='True Best')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin, 1.5*ymin] if true_best_value is None else [0.99*true_best_value, 1.5*ymin])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, comparison_label + ".pdf"))
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------
# TESTING ORIGINAL VS COMPILED SARO
# Before compiling, I've already saved about 1.2 seconds just reducing various overhead.
#-------------------------------------------------------------------------------------------------------------------------


import time
from importlib import reload

reload(saro_compiled)

t2 = time.perf_counter()
model_c = saro_compiled.CompiledSARO(fish, prey_velocity, xd, yd, max_iterations=700, pop_size=250, se=0.5, mu=50,
                                     dims=12, tracked=True)
solution = model_c.solve(True)
print("Solution took time", time.perf_counter() - t2) # value should be -0.146, or local min at -0.153



t1 = time.perf_counter()
model = SARO.OriginalSARO(problem_description(False), epoch=400, pop_size=60, se=0.5, mu=50)
model.solve()
print("Solution 1", model.solution[1][0], "in time", time.perf_counter() - t1) # value should be -0.146, or local min at -0.153



#-------------------------------------------------------------------------------------------------------------------------
# CALCULATE TRUE BEST FITNESS FOR GROUNDTRUTH
#-------------------------------------------------------------------------------------------------------------------------

print("Calculating high-iteration SARO for groundtruth value...")
model_true = SARO.OriginalSARO(problem_description(True), epoch=1500, pop_size=300, se=0.5, mu=50)
model_true.name = "SARO 1000x300 Groundtruth"
model_true.solve()
process_completion(model_true, plot=False)
true_best_value = -model_true.history.list_global_best_fit[-1]

#-------------------------------------------------------------------------------------------------------------------------
# COMPARE TOP EIGHT MODELS WITH ORIGINAL PARAMETERS TO 100K ITERATIONS
#-------------------------------------------------------------------------------------------------------------------------

for run_index in range(20):

    model_sma = SMA.BaseSMA(problem_description(False), epoch=500, pop_size=200, pr=0.03)
    model_sma.name = "Slime Mold Default"
    model_sma.solve()
    process_completion(model_sma, plot=False) # 0.149, 0.153, 0.152, 0.22 @ 100k (1000x100), 0.150, 0.150 @ 100k (2000x50), 0.150, 0.152 @ 100k (500x200)

    model_sade = DE.SADE(problem_description(False), epoch=1000, pop_size=100)
    model_sade.name = "SADE"
    model_sade.solve(mode='sequential') # sequential is the default, switching mode to 'thread' doesn't make things faster
    process_completion(model_sade, plot=False) # 0.148, 0.148, 0.147, 0.147, 0.147, 0.147, 0.148, 0.147 @ 100k, 0.14628342129912647 @ 4.5 mil (15000x300)

    model_qsa = QSA.ImprovedQSA(problem_description(False), epoch=250, pop_size=100)
    model_qsa.name = "Improved QSA"
    model_qsa.solve()
    process_completion(model_qsa, plot=False) # 0.149, 0.147, 0.149, 0.149, 0.148, 0.151, 0.147 @ 100k (250x100), 0.173, 0.160, 0.158 @ 100k (1000x25), 0.1465 @ 400k

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=500, pop_size=100, se=0.5, mu=50)
    model_saro.name = "SARO"
    model_saro.solve()
    process_completion(model_saro, plot=False) # 0.1462874499455973, 0.1464892747988185, 0.14629692591064095 @ 100k (500x100), 0.14628342250169188 @ 200k
    # Lowest energy cost was 0.14628341054963373 J after 3000000 fEvals.
    # Lowest energy cost was 0.1462834105496333 J after 1500000 fEvals.

    model_tlo = TLO.OriginalTLO(problem_description(False), epoch=500, pop_size=100)
    model_tlo.name = "TLO"
    model_tlo.solve()
    process_completion(model_tlo, plot=False) # 0.1469, 0.1561, 0.1466, 0.1467, 0.1472 @ 100k (500x100), 0.14677 @ 200k

    model_eo = EO.BaseEO(problem_description(False), epoch=1000, pop_size=100)
    model_eo.name = "EO"
    model_eo.solve()
    process_completion(model_eo, plot=False) # 0.151, 0.151, 0.153, 0.152, 0.151, 0.152 @ 100k (1000x100), 0.150, 0.150 @ 100k (500x200), 0.151, @ 100k (2000 x 50)

    model_gco = GCO.BaseGCO(problem_description(False), epoch=1000, pop_size=100, cr=0.7, wf=1.25)
    model_gco.name = "GCO"
    model_gco.solve()
    process_completion(model_gco, plot=False) # 0.149, 0.151, 0.150, 0.152, 0.151, 0.149, 0.152  @ 100k

    model_mfo = MFO.BaseMFO(problem_description(False), epoch=1000, pop_size=100)
    model_mfo.name = "MFO"
    model_mfo.solve()
    process_completion(model_mfo, plot=False) # 0.148, 0.150, 0.149, 0.148, 0.148, 0.150  @ 100k

    compare_multiple_models(f"Typical Chinook, Default Settings, Run {run_index}", [model_sma, model_sade, model_qsa, model_saro, model_tlo, model_eo, model_gco, model_mfo])


#-------------------------------------------------------------------------------------------------------------------------
# COMPARE TOP THREE MODELS USING DIFFERENT PARAMETER SETTINGS TO REACH 100K FUNCTION EVALS
# It's pretty clear from the tests with all 3 species that SARO is the leader with TLO and SADE close behind, and
# those three generally performing better than the others. So time to look at them more closely.
#-------------------------------------------------------------------------------------------------------------------------

for run_index in range(20):

    model_saro_1 = SARO.OriginalSARO(problem_description(False), epoch=500, pop_size=100, se=0.5, mu=50)
    model_saro_1.name = "SARO_500_100"
    model_saro_1.solve()
    process_completion(model_saro_1, plot=False)

    model_saro_2 = SARO.OriginalSARO(problem_description(False), epoch=1000, pop_size=50, se=0.5, mu=50)
    model_saro_2.name = "SARO_1000_50"
    model_saro_2.solve()
    process_completion(model_saro_2, plot=False)

    model_saro_3 = SARO.OriginalSARO(problem_description(False), epoch=250, pop_size=200, se=0.5, mu=50)
    model_saro_3.name = "SARO_250_200"
    model_saro_3.solve()
    process_completion(model_saro_3, plot=False)

    model_tlo_1 = TLO.OriginalTLO(problem_description(False), epoch=500, pop_size=100)
    model_tlo_1.name = "TLO_500_100"
    model_tlo_1.solve()
    process_completion(model_tlo_1, plot=False)

    model_tlo_2 = TLO.OriginalTLO(problem_description(False), epoch=1000, pop_size=50)
    model_tlo_2.name = "TLO_1000_50"
    model_tlo_2.solve()
    process_completion(model_tlo_2, plot=False)

    model_tlo_3 = TLO.OriginalTLO(problem_description(False), epoch=250, pop_size=200)
    model_tlo_3.name = "TLO_250_200"
    model_tlo_3.solve()
    process_completion(model_tlo_3, plot=False)

    model_sade_1 = DE.SADE(problem_description(False), epoch=1000, pop_size=100)
    model_sade_1.name = "SADE_1000_100"
    model_sade_1.solve()
    process_completion(model_sade_1, plot=False)

    model_sade_2 = DE.SADE(problem_description(False), epoch=2000, pop_size=50)
    model_sade_2.name = "SADE_2000_50"
    model_sade_2.solve()
    process_completion(model_sade_2, plot=False)

    model_sade_3 = DE.SADE(problem_description(False), epoch=500, pop_size=200)
    model_sade_3.name = "SADE_500_200"
    model_sade_3.solve()
    process_completion(model_sade_3, plot=False)

    compare_multiple_models(f"Dollies Popsize Strategy, Run {run_index}", [model_saro_1, model_saro_2, model_saro_3, model_tlo_1, model_tlo_2, model_tlo_3, model_sade_1, model_sade_2, model_sade_3])

#-------------------------------------------------------------------------------------------------------------------------
# COMPARE TOP THREE MODELS FOR DOLLY VARDEN USING WONKY MANEUVERS
#-------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
def plot_averaged_results(comparison_label="Comparison"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for model_name, result in combined_run_results.items():
        global_best = np.mean(np.array(combined_run_results[model_name]), axis=0)
        function_evals = combined_function_evals[model_name]
        plt.plot(function_evals, global_best, axes=ax, label=model_name)  # Plot some data on the (implicit) axes.
        ax.set_yscale('log')
        plt.legend()
    ax.set_xlabel("Objective function evaluations")
    ax.set_ylabel("Average maneuver cost (J)")
    ax.set_title(comparison_label)
    if true_best_value is not None:
        ax.axhline(y=true_best_value, ls='dotted', color='0.7', label='True Best')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin, 1.5*ymin] if true_best_value is None else [0.99*true_best_value, 1.5*ymin])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, comparison_label + ".pdf"))
    plt.show()

# xd, yd = 30, 250 # way far out and behind the fish
# xd, yd = 15, 1 # almost straight behind the fish -- recalculate the groundtruth value after doing this one!

combined_run_results = {}
combined_function_evals = {}

def add_to_combined_run_results(model):
    if model.name not in combined_run_results.keys():
        combined_run_results[model.name] = []
        combined_function_evals[model.name] = np.arange(0, model.nfe_per_epoch * (model.max_iterations + 1), model.nfe_per_epoch)
    combined_run_results[model.name].append(-np.array(model.history.list_global_best_fit))

n_runs = 50
for run_index in range(n_runs):
    print("-------------------- Run", run_index,"of",n_runs,"--------------------")

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=1000, pop_size=50, se=0.2, mu=50)
    model_saro.name = "SARO_1000x50_0.2_50"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=500, pop_size=100, se=0.5, mu=50)
    model_saro.name = "SARO_500_100_0.5_50"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=500, pop_size=100, se=0.2, mu=50)
    model_saro.name = "SARO_500_100_0.2_50"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=250, pop_size=200, se=0.5, mu=50)
    model_saro.name = "SARO_250_200_0.5_50"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=250, pop_size=200, se=0.2, mu=50)
    model_saro.name = "SARO_250_200_0.2_50"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)


current_num_runs = len(list(combined_run_results.values())[0])
plot_averaged_results(f"Dollies SARO Param Test 2 x,y={xd:.1f},{yd:.1f} 100k Evals {current_num_runs} Runs Averaged")








# compare_multiple_models(f"Dollies Top 4 Test - Run {run_index}", [model_saro_1, model_saro_2, model_tlo_1, model_sade_1])

# The models with popsize=100 take consistently more evaluations to reach the solution than the other ones.
# The models with popsize=25 are very efficient but may be more liable to get trapped in local minima.
# The models with popsize=13 are clearly more susceptible to local minima, no doubt about it.
# There's a pretty clear tradeoff that low popsize = faster convergence (per obj func eval) but greater likelihood of stalling in a local minimum.
# Right now it looks like popsize=50 might be the winner of this tradeoff for SARO and 25 for TLO and SADE, but I should revisit it after tweaking parameters.













xd, yd = 15, 1 # almost straight behind the fish -- recalculate the groundtruth value after doing this one!

print("Calculating high-iteration SARO for groundtruth value...")
model_true = SARO.OriginalSARO(problem_description(True), epoch=1000, pop_size=300, se=0.5, mu=50)
model_true.name = "SARO 1000x300 Groundtruth"
model_true.solve()
process_completion(model_true, plot=False)
true_best_value = -model_true.history.list_global_best_fit[-1]

combined_run_results = {}
combined_function_evals = {}


n_runs = 100
for run_index in range(n_runs):
    print("-------------------- Run", run_index,"of",n_runs,"--------------------")

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=1429, pop_size=35, se=0.5, mu=50)
    model_saro.name = "SARO_1429x35"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=1250, pop_size=40, se=0.5, mu=50)
    model_saro.name = "SARO_1250x40"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=1111, pop_size=45, se=0.5, mu=50)
    model_saro.name = "SARO_1111x45"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=1000, pop_size=50, se=0.5, mu=50)
    model_saro.name = "SARO_1000x50"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=909, pop_size=55, se=0.5, mu=50)
    model_saro.name = "SARO_909x55"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=833, pop_size=60, se=0.5, mu=50)
    model_saro.name = "SARO_833x60"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=769, pop_size=65, se=0.5, mu=50)
    model_saro.name = "SARO_769x65"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=714, pop_size=70, se=0.5, mu=50)
    model_saro.name = "SARO_714x70"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=167, pop_size=300, se=0.5, mu=50)
    model_saro.name = "SARO_167x300"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

current_num_runs = len(list(combined_run_results.values())[0])
plot_averaged_results(f"Dollies SARO Popsize Test x,y={xd},{yd} 100k Evals {current_num_runs} Runs Averaged")






xd, yd = 30, 250 # way far out and behind the fish

print("Calculating high-iteration SARO for groundtruth value...")
model_true = SARO.OriginalSARO(problem_description(True), epoch=1000, pop_size=300, se=0.5, mu=50)
model_true.name = "SARO 1000x300 Groundtruth"
model_true.solve()
process_completion(model_true, plot=False)
true_best_value = -model_true.history.list_global_best_fit[-1]

combined_run_results = {}
combined_function_evals = {}


n_runs = 100
for run_index in range(n_runs):
    print("-------------------- Run", run_index,"of",n_runs,"--------------------")

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=1429, pop_size=35, se=0.5, mu=50)
    model_saro.name = "SARO_1429x35"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=1250, pop_size=40, se=0.5, mu=50)
    model_saro.name = "SARO_1250x40"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=1111, pop_size=45, se=0.5, mu=50)
    model_saro.name = "SARO_1111x45"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=1000, pop_size=50, se=0.5, mu=50)
    model_saro.name = "SARO_1000x50"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=909, pop_size=55, se=0.5, mu=50)
    model_saro.name = "SARO_909x55"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=833, pop_size=60, se=0.5, mu=50)
    model_saro.name = "SARO_833x60"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=769, pop_size=65, se=0.5, mu=50)
    model_saro.name = "SARO_769x65"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=714, pop_size=70, se=0.5, mu=50)
    model_saro.name = "SARO_714x70"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

    model_saro = SARO.OriginalSARO(problem_description(False), epoch=167, pop_size=300, se=0.5, mu=50)
    model_saro.name = "SARO_167x300"
    model_saro.solve()
    process_completion(model_saro, plot=False)
    add_to_combined_run_results(model_saro)

current_num_runs = len(list(combined_run_results.values())[0])
plot_averaged_results(f"Dollies SARO Popsize Test x,y={xd},{yd} 100k Evals {current_num_runs} Runs Averaged")


# When running with 200k fEvals...
# SARO_5716_35 : Lowest energy cost was 0.15289295060088245 J after 200060 fEvals (2858x35).
# SARO_2500_40 : Lowest energy cost was 0.15289350920240802 J after 200000 fEvals (2500x40).
# SARO_2222_45 : Lowest energy cost was 0.1462834109472221 J after 199980 fEvals (2222x45).
# SARO_2000_50 : Lowest energy cost was 0.1462834105933778 J after 200000 fEvals (2000x50).
# SARO_1818_55 : Lowest energy cost was 0.14628341057259944 J after 199980 fEvals (1818x55).
# SARO_1666_60 : Lowest energy cost was 0.15289363278265258 J after 199920 fEvals (1666x60).
# SARO_1538_65 : Lowest energy cost was 0.14628341081487023 J after 199940 fEvals (1538x65).
# SARO_1428_70 : Lowest energy cost was 0.14628341065161643 J after 199920 fEvals (1428x70).
# There is celarly a local minimum at 0.15289 and global at 0.146283
# At least with the 'typical' maneuver test, averages will come down to which one