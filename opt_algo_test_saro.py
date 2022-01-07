import numpy as np
import os
import matplotlib.pyplot as plt

from maneuvermodel import maneuveringfish, maneuver, optimize, save_and_load
from maneuvermodel.constants import SLOW_OPT_N, SLOW_OPT_ITERATIONS

SAVED_TEST_INPUT_PATH = "/Users/Jason/Dropbox/drift model project/Papers/Capture Maneuver Model/Saved Test Maneuver Inputs/"
FIGURE_OUTPUT_PATH = "/Users/Jason/Dropbox/drift model project/Papers/Capture Maneuver Model/Figures/Ancillary Figures 2021/OptAlgoTests/"

def plot_averaged_results(true_best_value, comparison_label="Comparison"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    if true_best_value is not None:
        ax.axhline(y=true_best_value, ls='dotted', color='0.7', label='True Best')
    for model_label, result in combined_run_results.items():
        average_best = np.mean(np.array(combined_run_results[model_label]), axis=0)
        function_evals = np.mean(np.array(combined_function_evals[model_label]), axis=0)
        plt.plot(function_evals, average_best, axes=ax, label=model_label, linewidth=0.25)  # Plot some data on the (implicit) axes.
        ax.set_yscale('log')
        plt.legend()
    ax.set_xlabel("Objective function evaluations")
    ax.set_ylabel("Average maneuver cost (J)")
    ax.set_title(comparison_label)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin, 1.5*ymin] if true_best_value is None else [0.99*true_best_value, 1.5*true_best_value])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, comparison_label + ".pdf"))
    plt.show()

# saved_test_maneuver_name = 'Typical Dolly Maneuver'
# saved_test_maneuver_name = 'Typical Chinook Maneuver'
# saved_test_maneuver_name = 'Typical Grayling Maneuver'

saved_test_maneuver_names = [
"2015-08-06-1 Chena - Chinook Salmon (id #3) Attempt 98",
"2015-07-28-1 Clearwater - Arctic Grayling (id #1) Attempt 172",
"2015-07-31-1 Clearwater - Arctic Grayling (id #2) Attempt 45",
"2016-08-02-1 Clearwater - Arctic Grayling (id #1) Attempt 179",
"2016-06-16-1 Panguingue - Dolly Varden (id #1) Attempt 62",
"2015-07-11-2 Chena - Chinook Salmon (id #4) Attempt 51",
"2016-06-02-2 Chena - Chinook Salmon (id #5) Attempt 168",
"2015-09-04-1 Clearwater - Arctic Grayling (id #1) Attempt 37",
"2015-07-11-1 Chena - Chinook Salmon (id #4) Attempt 67",
"2015-08-04-1 Chena - Chinook Salmon (id #3) Attempt 135" # gets trapped sometimes in a local min with 3-4x cost, doing a circular loop rather than direct maneuver
]

test_name = "CoarserTestOfN"

for saved_test_maneuver_name in saved_test_maneuver_names:
    print("\nProcessing attempt", saved_test_maneuver_name,"\n")
    saved_test_maneuver_path = os.path.join(SAVED_TEST_INPUT_PATH, saved_test_maneuver_name + ".json")

    combined_run_results = {}
    combined_function_evals = {}

    def add_to_combined_run_results(model):
        if model.label not in combined_run_results.keys():
            combined_run_results[model.label] = []
            combined_function_evals[model.label] = [] # np.arange(0, model.nfe_per_epoch * (model.epoch + 1), model.nfe_per_epoch)
        combined_run_results[model.label].append(model.tracked_activity_cost)
        combined_function_evals[model.label].append(model.tracked_nfe)

    opt_true, model_true = save_and_load.optimal_maneuver_from_saved_setup(saved_test_maneuver_path, n=SLOW_OPT_N, max_iterations=SLOW_OPT_ITERATIONS)
    n_runs = 250
    for run_index in range(n_runs):
        print("-------------------- Run", run_index+1,"of",n_runs,"--------------------")

        opt, model = save_and_load.optimal_maneuver_from_saved_setup(saved_test_maneuver_path, n=250, max_iterations=400, tracked=True, return_optimization_model=True)
        model.label = f"SARO_{model.max_iterations}x{model.pop_size}"
        add_to_combined_run_results(model)

        opt, model = save_and_load.optimal_maneuver_from_saved_setup(saved_test_maneuver_path, n=200, max_iterations=500, tracked=True, return_optimization_model=True)
        model.label = f"SARO_{model.max_iterations}x{model.pop_size}"
        add_to_combined_run_results(model)

        opt, model = save_and_load.optimal_maneuver_from_saved_setup(saved_test_maneuver_path, n=150, max_iterations=667, tracked=True, return_optimization_model=True)
        model.label = f"SARO_{model.max_iterations}x{model.pop_size}"
        add_to_combined_run_results(model)

        opt, model = save_and_load.optimal_maneuver_from_saved_setup(saved_test_maneuver_path, n=100, max_iterations=1000, tracked=True, return_optimization_model=True)
        model.label = f"SARO_{model.max_iterations}x{model.pop_size}"
        add_to_combined_run_results(model)


    plot_averaged_results(opt_true.activity_cost, comparison_label="{0} - {1} runs={2}".format(test_name, saved_test_maneuver_name, len(list(combined_run_results.values())[0])))
