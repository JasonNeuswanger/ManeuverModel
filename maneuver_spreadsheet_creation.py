import numpy as np
import os
import sys
from platform import uname
import pymysql
import traceback
from maneuvermodel import maneuveringfish, optimize_cuckoo
from maneuvermodel.dynamics import CONVERGENCE_FAILURE_COST

# This file is kept fairly slim.
# See maneuver_spreadsheet_support.py for extensive comments justifying this code and setting up the server structures to host/track results.
#
# Code to call the script from the command line: "python maneuver_spreadsheet_creation.py"
#
# The script will compare a list of final intended spreadsheets against a list of currently completed or in-progress spreadsheets and figure out the 
# next one to work on automatically. When completed, it will create the folder for the results (if it doesn't already exist) and place the spreadsheet there.
# When using this file to create custom tables, don't import the whole thing. Instead do from maneuver_spreadsheet_creation import calculate_cost_tables, etc.
#
# Alternatively, just import calculate_cost_tables into another Python script. This will return the energy costs, pursuit durations, x values of the grid, 
# and y values of the grid.

IS_MAC = (os.uname()[0] == 'Darwin')

def calculate_cost_tables(fork_length, focal_velocity, prey_velocity, taskid, dbcursor):
    max_thrust = 2.4 * fork_length + 40
    fish_mass = 0.0 # this input defaults the model to a rainbow trout length-mass regression
    fish_SMR = 0.0  # unused in this case
    fish_NREI = 0.0 # also unused in this case
    temperature = 10 # also unused in this case
    use_total_cost = False
    disable_wait_time = False
    fish = maneuveringfish.ManeuveringFish(fork_length, focal_velocity, fish_mass, temperature, fish_SMR, max_thrust, fish_NREI, use_total_cost, disable_wait_time)
    prey_length_mm_for_max_distance = 20  # mm, set the max distance considered to the max at which 20-mm prey could be resolved
    max_visible_distance = 12 * prey_length_mm_for_max_distance * (1. - np.exp(-0.2 * fork_length)) # max visible distance of prey in cm
    # the minimum lateral (y) distance we'll consider for maneuver is 0.2 cm, basically just a head-snap maneuver for the smallest drift foragers
    # We also insert a couple of manual values in there to get consistent coverage at short distances before the scaled values kick in for all fish.
    (xmin, xmax, ymin, ymax) = (-max_visible_distance, max_visible_distance, 0.2, max_visible_distance)
    sp = 4 # sp = spacing power used to emphasize points closer to the fish while still covering much more distant points adequately
    def scale(x):
        return (x)**sp
    def scale_inv(x):
        return abs(abs(x)**(1.0/sp))
    xs = np.concatenate([-abs(np.flip(scale(np.linspace(scale_inv(1), scale_inv(-xmin), 20))[1:], axis=0)), [-0.1], scale(np.linspace(scale_inv(1), scale_inv(xmax), 10)[1:])])
    ys = np.concatenate([[ymin, 0.7, 1.2], scale(np.linspace(scale_inv(1), scale_inv(xmax), 20)[2:])])
    # This grid has 609 values per sheet, similar to the previous Amazon grid.
    # for x in xs: print("x = {0:.5f}".format(x)) # print statements that can be used to check grid spacing
    # for y in ys: print("y = {0:.5f}".format(y))
    ec = np.zeros(shape=(len(xs),len(ys)))
    pd = np.zeros(shape=(len(xs),len(ys)))
    count = 1
    final_count = float(len(xs) * len(ys))
    for i in range(len(xs)):
        for j in range(len(ys)):
            print("Calculating optimal maneuver for detection point ", xs[i], ", ", ys[j], ", 0")
            sol = optimize_cuckoo.optimal_maneuver_CS(fish, detection_point_3D = (xs[i], ys[j], 0.0), prey_velocity=prey_velocity, n=50, iterations=3000, p_a=0.25, suppress_output=True)
            if sol.energy_cost != CONVERGENCE_FAILURE_COST:
                ec[i,j] = sol.energy_cost
                pd[i,j] = sol.pursuit_duration
            else:
                ec[i,j] = np.nan
                pd[i,j] = np.nan
            if IS_MAC:
                print("Solution {0} of {1}: For fl={2:.1f} cm, fv={3:.1f} cm/s, pv={4:.1f} C, at x={7:.2f} and y={8:.2f}, energy cost is {5:.5f} J and pursuit duration is {6:.3f} s.".format(count, len(xs)*len(ys), fork_length, focal_velocity, prey_velocity, sol.energy_cost, sol.pursuit_duration, xs[i], ys[j]))
            if count % 11 == 0:
                dbcursor.execute("UPDATE maneuver_model_tasks SET progress={0} WHERE taskid={1}".format(count/final_count, taskid))
            count += 1 
    # Now, run quality control check on the ec and pd values, redoing calculation if they're too far off from their neighbors
    for table in (ec, pd):
        imax = table.shape[0]
        jmax = table.shape[1]
        for i in range(imax):
            for j in range(jmax):
                i_min = 0 if i == 0 else i-1
                i_max = imax+1 if i == imax else i+2
                j_min = 0 if j == 0 else j-1
                j_max = jmax+1 if j == jmax else j+2
                neighbors = table[i_min:i_max, j_min:j_max]
                notnan_neighbors = neighbors[~np.isnan(neighbors)]
                if len(notnan_neighbors) >= 3: # only do the neighbor-based QC check if there are enough "neighbors" (2 + current number) to check against the median
                    neighbor_median = np.median(notnan_neighbors) # median of 4 (corner), 6 (edge), or 9-value (center) block around present cell
                    ratio_to_median = table[i,j] / neighbor_median
                    worst_allowable_ratio = 3.0 # assume optimal solution wasn't found if solution differs from neighbors by factor of 3
                    if ratio_to_median < 1/worst_allowable_ratio or ratio_to_median > worst_allowable_ratio:
                        for retry in range(5): # If we didn't get reasonable values the first time, try again up to 5 times with more rigorous but time-consuming algorithm parameters
                            dbcursor.execute("UPDATE maneuver_model_tasks SET retries=retries+1 WHERE taskid={0}".format(taskid))
                            sol = optimize_cuckoo.optimal_maneuver_CS(fish, detection_point_3D=(xs[i], ys[j], 0.0), n=100, iterations=3000+retry*2000, p_a=0.25, suppress_output=True)
                            if sol.energy_cost < ec[i,j]:
                                ec[i,j] = sol.energy_cost
                                pd[i,j] = sol.pursuit_duration
                                ratio_to_median = table[i,j] / neighbor_median
                                if 1/worst_allowable_ratio <= ratio_to_median <= worst_allowable_ratio:
                                    break
                        if ratio_to_median < 1/worst_allowable_ratio or ratio_to_median > worst_allowable_ratio:
                            print("Retries to match neighbors failed for x={0}, y={1} with fl={2}, fv={3}, pv={4}. ratio_to_median={5}".format(xs[i], ys[j], fork_length, focal_velocity, prey_velocity, ratio_to_median))
                            dbcursor.execute("UPDATE maneuver_model_tasks SET has_failed_retries=1 WHERE taskid={0}".format(taskid))
    # Now add on the mirror image of the first 4 columns to each extrapolation, with negative y values, to facilitate smooth interpolation near y=0
    ys = np.concatenate([np.flip(-ys[:4], axis=0), ys])
    ec = np.concatenate([np.flip(ec[:,:4], axis=1), ec], axis=1)
    pd = np.concatenate([np.flip(pd[:,:4], axis=1), pd], axis=1)
    return ec, pd, xs, ys

def save_cost_tables(ec, pd, xs, ys, fl, fv, pv):
    if IS_MAC:
        base_folder = os.path.join(os.path.sep, 'Users', 'Jason', 'Desktop', 'Maneuver Sheet Test')
    else:
        base_folder = os.path.join(os.path.sep, 'home', 'alaskajn', 'new_maneuver_tables')
    folder = os.path.join(base_folder, "fl_{0}".format(fl), "fv_{0}".format(fv), "pv_{0}".format(pv))
    if not os.path.exists(folder):
        os.makedirs(folder)    
    def save_cost_table(costs, xs, ys, filename):
        costs_labeled = np.insert(np.insert(costs, 0, xs, axis=1), 0, np.insert(ys, 0, np.nan), axis=0) # add y labels along horizontal axis, x along vertical axis
        np.savetxt(os.path.join(folder, filename), costs_labeled, fmt='%.7f,')
    save_cost_table(ec, xs, ys, 'energy_cost.csv')
    save_cost_table(pd, xs, ys, 'pursuit_duration.csv')
        
try:
    db = pymysql.connect(host="troutnut.com", port=3306, user="jasonn5_calibtra", passwd="aVoUgLJyKo926", db="jasonn5_calibration_tracking", autocommit=True)
    cursor = db.cursor()
    select_query = "SELECT taskid, fork_length, focal_velocity, prey_velocity FROM maneuver_model_tasks WHERE start_time IS NULL"
    cursor.execute(select_query)
    task_data = cursor.fetchone()
    while task_data is not None:
        taskid, fork_length, focal_velocity, prey_velocity = task_data
        instance_id = 'my_laptop' if IS_MAC else uname()[1]
        print("Beginning task {0}.".format(taskid))
        cursor.execute("UPDATE maneuver_model_tasks SET start_time=NOW(), machine='{1}', progress=0.0 WHERE taskid={0}".format(taskid, instance_id))
        ec, pd, xs, ys = calculate_cost_tables(fork_length, focal_velocity, prey_velocity, taskid, cursor)
        save_cost_tables(ec, pd, xs, ys, fork_length, focal_velocity, prey_velocity)
        cursor.execute("UPDATE maneuver_model_tasks SET completion_time=NOW(), progress=NULL WHERE taskid={0}".format(taskid))
        cursor.execute(select_query)
        task_data = cursor.fetchone() # keep fetching new tasks until interrupted
except Exception as e:
    traceback.print_exc()
finally:
    db.close()



