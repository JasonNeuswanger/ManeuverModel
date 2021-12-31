import numpy as np
import os
import pickle
from maneuvermodel import maneuveringfish, maneuver
from maneuvermodel import saro_compiled
import matplotlib.pyplot as plt

MANEUVER_INPUT_CACHE_PATH = '/Users/Jason/Dropbox/Drift Model Project/Data/Cached Maneuver Input Data/'
FIGURE_OUTPUT_PATH = "/Users/Jason/Dropbox/drift model project/Papers/Capture Maneuver Model/Figures/Ancillary Figures 2021/OptAlgoTests/"

def load_real_maneuver(input_cache_fish_label, maneuver_index, use_total_cost=False, disable_wait_time=True):
    cache = pickle.load(open(os.path.join(MANEUVER_INPUT_CACHE_PATH, input_cache_fish_label + ".pickle"), "rb"))
    fish = maneuveringfish.ManeuveringFish(fork_length = cache['fish']['fork_length_cm'],
                                           mean_water_velocity= cache['fish']['mean_focal_current_speed_cm_per_s'],
                                           base_mass = cache['fish']['mass_g'],
                                           temperature = cache['fish']['temperature'],
                                           SMR = cache['fish']['SMR'],
                                           max_thrust = cache['fish']['max_thrust'],
                                           NREI = cache['fish']['NREI'],
                                           use_total_cost = use_total_cost,
                                           disable_wait_time = disable_wait_time)
    maneuver = cache['maneuvers'][maneuver_index]
    prey_velocity = maneuver['prey_speed_cm_per_s']
    fish.focal_velocity = maneuver['focal_current_speed_cm_per_s'] # todo just make this a maneuver attribute to avoid confusion
    xd, yd = maneuver['detection_point_2D_cm']
    return {'fish': fish, 'fv': fish.focal_velocity, 'pv': prey_velocity, 'xd': xd, 'yd': yd}

def plot_averaged_results(true_best_value, comparison_label="Comparison"):
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

tm = load_real_maneuver("2015-06-10-1 Chena - Chinook Salmon (id #1)", 25) # 25 was good, 2 ok, 6, 31, 73, 83, 87
combined_run_results = {}
combined_function_evals = {}

def add_to_combined_run_results(model):
    if model.label not in combined_run_results.keys():
        combined_run_results[model.label] = []
        combined_function_evals[model.label] = model.tracked_nfe
    combined_run_results[model.label].append(-model.tracked_fitness)


long_run_model = saro_compiled.CompiledSARO(tm['fish'], tm['pv'], tm['xd'], tm['yd'], max_iterations=5000, pop_size=500,
                                            se=0.5, mu=50, dims=12, tracked=True)  # 2500/300
long_run_model.solve(True)
true_best_fitness = long_run_model.solution.fitness

n_runs = 25
for run_index in range(n_runs):
    print("-------------------- Run", run_index,"of",n_runs,"--------------------")

    model = saro_compiled.CompiledSARO(tm['fish'], tm['pv'], tm['xd'], tm['yd'], max_iterations=round(150000 / 325),
                                       pop_size=325, se=0.6, mu=round(150000 / 325), dims=12, tracked=True)
    model.label = "SARO_{0}x{1}_{2:.2f}_{3}".format(model.max_iterations, model.pop_size, model.se, model.mu)
    solution = model.solve(True)
    add_to_combined_run_results(model)

    model = saro_compiled.CompiledSARO(tm['fish'], tm['pv'], tm['xd'], tm['yd'], max_iterations=round(150000 / 325),
                                       pop_size=325, se=0.1, mu=50, dims=12, tracked=True)
    model.label = "SARO_{0}x{1}_{2:.2f}_{3}".format(model.max_iterations, model.pop_size, model.se, model.mu)
    solution = model.solve(True)
    add_to_combined_run_results(model)

    model = saro_compiled.CompiledSARO(tm['fish'], tm['pv'], tm['xd'], tm['yd'], max_iterations=round(150000 / 325),
                                       pop_size=325, se=5, mu=50, dims=12, tracked=True)
    model.label = "SARO_{0}x{1}_{2:.2f}_{3}".format(model.max_iterations, model.pop_size, model.se, model.mu)
    solution = model.solve(True)
    add_to_combined_run_results(model)

    model = saro_compiled.CompiledSARO(tm['fish'], tm['pv'], tm['xd'], tm['yd'], max_iterations=round(150000 / 100),
                                       pop_size=100, se=0.6, mu=round(150000 / 100), dims=12, tracked=True)
    model.label = "SARO_{0}x{1}_{2:.2f}_{3}".format(model.max_iterations, model.pop_size, model.se, model.mu)
    solution = model.solve(True)
    add_to_combined_run_results(model)




# Try the best from this test: se=0.8, mu=300, popsize=350
# Try it modified with the best individual values from this test: se=0.7, mu=833, popsize=350
# Test effect of dropping SE to 0.6 and 0.5
# Test the best dropping mu to 50

# In 50 runs, the best is 350 / 0.60 / 714, worst is 0.70 / same, others near identical in between... need to try a fresh set of runs.
# In 50 separate runs, results were the same as in the 50 above, so not spurious.

# Now the best seems to be 0.60 / 325 / max mu

current_num_runs = len(list(combined_run_results.values())[0])
plot_averaged_results(-true_best_fitness, f"Testing Multi Params C on Maneuver 25 x,y={tm['xd']:.1f},{tm['yd']:.1f} 100k Evals {current_num_runs} Runs Averaged")


# Maneuver 25
# 500x300 0.6 300 worked best...
# 500x300 0.6 75  worked similarly, but converged more slowly and was a bit jagged even with 200 runs, seemed to be trending downward more
# 150 worked worse than 75 and 300, kind of odd
# SE = 1.0 was flat out bad regardless of mu
# SE = 0.8 was mostly worse than 0.6 on average, but not by a lot

# Maneuver 2
# 500x300 0.6 75 was slowest to converge - 3rd best
# 500x300 0.8 75 did not perfectly converge - 4th best
# 500x300 0.6 300 converged a little slower but got three - 2nd best
# 500x300 0.8 300 converged all the way fastest - best

# Maneuver 6
# 0.8 converges faster than 0.6, none converge perfectly, all reach same average eventually

# Maneuver 31
# All four converge by about 150k iteraitons
# Both 0.8 converges a bit faster than either 0.6
# 0.8/300 converges a bit faster than 75

# Maneuver 73
# All four come close at 300k, very similarly
# Fastest were 0.8-75 and 0.8-300, though 0.8-300 was very similar to 0.6-300 in the middle of the pack

# Maneuver 83
# 0.8 and 300 was best
# Other 0.8 was next best
# All others grouped closely

# Maneuer 87
# Both 0.8 are almost identical and better than both 0.6

# Big overnight test at 500k iters
# Lowest energy cost was 0.00025527937717960555 J after 500400 evals  ( 833 x 300 ; se =  0.7 ; mu =  300 ).
# Lowest energy cost was 0.00025527936354050317 J after 500400 evals  ( 833 x 300 ; se =  0.75 ; mu =  300 ).
# Lowest energy cost was 0.0002552793635205535 J after 500400 evals  ( 833 x 300 ; se =  0.8 ; mu =  300 ).
# Lowest energy cost was 0.00025527937429121214 J after 500400 evals  ( 833 x 300 ; se =  0.85 ; mu =  300 ).
# Lowest energy cost was 0.0002552794377417951 J after 500400 evals  ( 833 x 300 ; se =  0.8 ; mu =  833 ).
# Lowest energy cost was 0.00025527936243501567 J after 500500 evals  ( 1000 x 250 ; se =  0.8 ; mu =  300 ).
# Lowest energy cost was 0.00025527937400053396 J after 500500 evals  ( 714 x 350 ; se =  0.8 ; mu =  300 ).
# Lowest energy cost was 0.000255279431078958 J after 500800 evals  ( 625 x 400 ; se =  0.8 ; mu =  300 ).
# Lowest energy cost was 0.0002552797525673629 J after 501300 evals  ( 556 x 450 ; se =  0.8 ; mu =  300 ).
# Lowest energy cost was 0.0002552804930453066 J after 501000 evals  ( 500 x 500 ; se =  0.8 ; mu =  300 ).
#
# Conclusions from the plot
# At popsize=300, dropping SE from 0.8 made things better, raising made things worse. 0.7 was best, and lowest in this test.
# With se=0.8, mu = 833 (equal to epoch) was better than mu = 300
# For popsize, 350-400 were best, being nearly identical, with 350 converging a little faster and 400 just a sliver better at very high iters.
# Popsizes 200-250 were the worst, and 450-500 were worse than 350-400.

# New test:
# Try the best from this test: se=0.8, mu=300, popsize=350
# Try it modified with the best individual values from this test: se=0.7, mu=833, popsize=350
# Test effect of dropping SE to 0.6 and 0.5


# Good solution is 0.000255279
# 0.00026....
n_runs = 10
for run_index in range(n_runs):
    print("-------------------- Run", run_index,"of",n_runs,"--------------------")

    model = saro_compiled.CompiledSARO(tm['fish'], tm['pv'], tm['xd'], tm['yd'], max_iterations=round(16000 / 100),
                                       pop_size=100, se=0.5, mu=50, dims=12, tracked=True)
    model.label = "SARO_{0}x{1}_{2:.2f}_{3}".format(model.max_iterations, model.pop_size, model.se, model.mu)
    solution = model.solve(True)


