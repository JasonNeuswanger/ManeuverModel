import time
import numpy as np
import matplotlib.pyplot as plt
from .maneuver import maneuver_from_proportions
from .saro_compiled import CompiledSARO
from .constants import CONVERGENCE_FAILURE_COST
from .visualize import param_labels

def run_convergence_test(fish, detection_point_3D, prey_velocity=None, label="Unnamed", export_path=None, display=True, iterations=500, n=100, global_iterations=1000, global_n=325, n_tests=10, se=0.5, mu=100):
    """ This is a wrapper for optimal_maneuver which runs it multiple times, once slowly with over-the-top resources
        to hopefully determine the global optimum for reference, and then n_tests times with more common run settings
        to see how well the algorithm converges under those conditions."""
    prey_velocity_passed = fish.focal_velocity if prey_velocity is None else prey_velocity
    print("Calculating global optimum...")
    global_opt, global_opt_model = optimal_maneuver(fish, detection_point_3D, prey_velocity=prey_velocity_passed, iterations=global_iterations, n=global_n, tracked=True, return_optimization_model=True)
    print("Calculating replicates...")
    plt.ioff()
    fig, ((ax, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    # ----------------------------------------- Plot histories histories for each fast solution ---------------------------------#
    ax.axhline(y=global_opt.energy_cost, ls='dotted', color='0.7', label='Global Optimum')
    ax3.axhline(y=global_opt.pursuit_duration, ls='dotted', color='0.7', label='Global Optimum')
    ax4.axhline(y=global_opt.capture_x, ls='dotted', color='0.7', label='Global Optimum')
    for _ in range(n_tests):
        opt, opt_model = optimal_maneuver(fish, detection_point_3D, prey_velocity=prey_velocity_passed, iterations=iterations, n=n, se=se, mu=mu, tracked=True, return_optimization_model=True)
        ax.plot(opt_model.tracked_nfe, opt_model.tracked_energy_cost, label="{0:7.6f} x Glob Opt".format(opt.energy_cost / global_opt.energy_cost))
        ax3.plot(opt_model.tracked_nfe, opt_model.tracked_pursuit_duration, label="{0:7.6f}".format(opt.pursuit_duration))
        ax4.plot(opt_model.tracked_nfe, opt_model.tracked_capture_x, label="{0:7.6f}".format(opt.capture_x))
        ax.set_yscale('log')
    ax.set_ylabel("Maneuver activity cost (J)")
    ax3.set_ylabel("Pursuit duration (s)")
    ax4.set_ylabel("Capture x (cm)")
    ax.set_ylim([0.99 * global_opt.energy_cost, 1.5 * global_opt.energy_cost])
    # -------------------------- Plot parameter values over the evolution of the global optimum solution -------------------------#
    param_values = global_opt_model.tracked_position.transpose()
    for i, pv in enumerate(param_values): ax2.plot(global_opt_model.tracked_nfe, pv, label=param_labels[i])
    ax2.set_ylabel("Proportional parameters")
    for axis in (ax, ax2, ax3, ax4):
        axis.set_xlabel("Objective function evaluations")
        axis.legend()
    if label != "Unnamed": fig.suptitle(label)
    fig.tight_layout()
    if export_path is not None:
        fig.savefig(export_path)
    if display:
        fig.show()
    else:
        plt.close(fig)
    return global_opt

def optimal_maneuver(fish, detection_point_3D, **kwargs):
    start_time = time.perf_counter()
    #-------------------------------------------------------------------------------------------------------------------
    # Convert the maneuver from 3D to 2D and save the information to convert back again
    #-------------------------------------------------------------------------------------------------------------------
    y3D, z3D = detection_point_3D[1:3]
    R = np.sqrt(y3D**2 + z3D**2)
    matrix_2Dfrom3D = np.array([[1,0,0],[0,y3D/R,z3D/R],[0,-z3D/R,y3D/R]]) # matrix to rotate the 3-D detection point about the x-axis into the x-y plane
    matrix_3Dfrom2D = matrix_2Dfrom3D.T                                    # because the inverse of this matrix is also its transpose
    (xd, yd) = matrix_2Dfrom3D.dot(np.array(detection_point_3D))[0:2]      # 2-D detection point to use for the model, not yet sign-adjusted
    ysign = np.sign(yd)
    yd *= ysign # Because of symmetry, we do maneuver calculations in the positive-y side of the x-y plane, saving the sign to convert back at the end
    #-------------------------------------------------------------------------------------------------------------------
    # Find the optimal maneuver using Search and Rescue Optimization
    #-------------------------------------------------------------------------------------------------------------------
    prey_velocity = kwargs.get('prey_velocity', fish.focal_velocity)  # Prey velocity defaults to fish focal velocity if prey velocity not specified
    dims = 12 if not (fish.disable_wait_time or xd > 0) else 11  # Don't bother optimizing wait time if it's disabled or item was detected downstream
    optimization_model = CompiledSARO(fish,
                                      prey_velocity,
                                      xd,
                                      yd,
                                      epoch=kwargs.get('iterations', 500),
                                      pop_size=kwargs.get('n', 325),
                                      se=kwargs.get('se', 0.6),
                                      mu=kwargs.get('mu', 500),
                                      dims=dims,
                                      tracked=kwargs.get('tracked', False))
    solution = optimization_model.solve()
    fittest_maneuver = maneuver_from_proportions(fish, prey_velocity, xd, yd, solution.position)
    fittest_maneuver.objective_function_evaluations = optimization_model.nfe

    fittest_maneuver.matrix_3Dfrom2D = np.ascontiguousarray(matrix_3Dfrom2D) # Set attributes to allow the fittest solution to convert; the contiguous array typing prevents a silly warning about Numba execution speed in np.dot in maneuver.to_3D
    fittest_maneuver.ysign = ysign                     # results back into 3-D
    end_time = time.perf_counter()
    time_cost_s = end_time - start_time
    #-------------------------------------------------------------------------------------------------------------------
    # Calculate summary metrics and print/export any output
    #-------------------------------------------------------------------------------------------------------------------
    fittest_maneuver.calculate_summary_metrics()  # calculate final summary quantities like average metabolic rate that are only needed for the optimal solution, not to evaluate fitness while finding it
    label = kwargs.get('label', "")
    if not kwargs.get('suppress_output', False):
        if fittest_maneuver.energy_cost != CONVERGENCE_FAILURE_COST:
            print("Lowest energy cost after {0} iterations ({7:8d} evaluations, {8:5.1f} s) was {1:10.6f} joules. Mean speed {2:4.1f} cm/s, {3:5.2f} bodylengths/s. Metabolic rate {4:7.1f} mg O2/kg/hr ({5:4.1f}X SMR). {6}".format(optimization_model.epoch, fittest_maneuver.energy_cost, fittest_maneuver.mean_swimming_speed, fittest_maneuver.mean_swimming_speed_bodylengths, fittest_maneuver.mean_metabolic_rate, fittest_maneuver.mean_metabolic_rate_SMRs,label, fittest_maneuver.objective_function_evaluations, time_cost_s))
            # if fittest_maneuver.dynamics.bad_thrust_b_penalty > 0:
            #     print("The best maneuver included a penalty for a bad thrust in stage b of the final straight, penalty factor {0:.3f}.".format(fittest_maneuver.dynamics.bad_thrust_b_penalty))
            # if fittest_maneuver.dynamics.violates_acceleration_limit_penalty > 0:
            #     print("The best maneuver included a penalty for violating the acceleration limit, penalty factor {0:.3f}.".format(fittest_maneuver.dynamics.violates_acceleration_limit_penalty))
        else:
            print("Maneuver failed to converge in all possible paths/dynamics considered. Did not find an optimal maneuver.")
            if hasattr(fittest_maneuver, 'convergence_failure_code'):
                print("Convergence failure code in maneuver.py was {0}.".format(fittest_maneuver.convergence_failure_code))
                if fittest_maneuver.convergence_failure_code == 4:  # failure in final straight
                    print("Convergence failure code in finalstraight.py was {0}.".format(fittest_maneuver.dynamics.straight_3.convergence_failure_code))
    if kwargs.get('return_optimization_model', False):
        return fittest_maneuver, optimization_model
    else:
        return fittest_maneuver
