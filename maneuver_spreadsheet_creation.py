import numpy as np
import os
import time
from platform import uname
import pymysql
import pickle
from maneuvermodel import optimize, maneuveringfish
from maneuvermodel.constants import DEFAULT_OPT_N, DEFAULT_OPT_ITERATIONS, CONVERGENCE_FAILURE_COST

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

FAST_TEST = False # Set to true for debugging, to create a whole result set in a couple minutes with less grid resolution and convergence

IS_MAC = (os.uname()[0] == 'Darwin')

db_credentials = pickle.load(open("calibration_db_credentials.pickle", "rb"))

def db_connect():
    return pymysql.connect(host="troutnut.com", port=3306, user=db_credentials["user"],
                           passwd=db_credentials["passwd"], db=db_credentials["db"], autocommit=True)
db = db_connect() # Maintain a global database connection that may be rebuilt with this function when the connection is lost

def db_execute(query):
    global db
    completed = False
    while not completed:
        try:
            cursor = db.cursor()
            cursor.execute(query)
        except pymysql.err.OperationalError:
            print("Database not connected. Waiting 5 minutes and retrying.")
            time.sleep(5*60)
            db = db_connect()
            continue
        else:
            completed = True
            return cursor

def calculate_cost_tables(fork_length, velocity, taskid):
    fish = maneuveringfish.ManeuveringFish(fork_length=fork_length,
                                           mean_water_velocity=velocity,
                                           base_mass=0,  # defaults to regression from length
                                           temperature=10,  # irrelevant in this case
                                           SMR=0,  # irrelevant in this case
                                           max_thrust=250,
                                           NREI=0,  # irrelevant in this case
                                           use_total_cost=False,
                                           disable_wait_time=False)
    prey_length_mm_for_max_distance = 25  # mm, set the max distance considered to the max at which 25-mm prey could be resolved (asymptotically approaches 3 m for large fish)
    max_visible_distance = 12 * prey_length_mm_for_max_distance * (1. - np.exp(-0.2 * fork_length)) # max visible distance of prey in cm
    # the minimum lateral (y) distance we'll consider for maneuver is 0.2 cm, basically just a head-snap maneuver for the smallest drift foragers
    # We also insert a couple of manual values in there to get consistent coverage at short distances before the scaled values kick in for all fish.
    (xmin, xmax, ymin, ymax) = (-max_visible_distance, max_visible_distance, 0.2, max_visible_distance)
    sp = 4 # sp = spacing power used to emphasize points closer to the fish while still covering much more distant points adequately
    def scale(x):
        return (x)**sp
    def scale_inv(x):
        return abs(abs(x)**(1.0/sp))
    xs = np.concatenate([-abs(np.flip(scale(np.linspace(scale_inv(1), scale_inv(-xmin), 24))[1:], axis=0)), [-0.1], scale(np.linspace(scale_inv(1), scale_inv(xmax), 14)[1:])])
    ys = np.concatenate([[ymin, 0.7, 1.2, 1.8], scale(np.linspace(scale_inv(1), scale_inv(xmax), 25)[2:])])
    if FAST_TEST:
        xs = xs[::5]
        ys = ys[::5]
    # This grid has 999 values per sheet.
    # for x in xs: print("x = {0:.5f}".format(x)) # print statements that can be used to check grid spacing
    # for y in ys: print("y = {0:.5f}".format(y))
    ac = np.zeros(shape=(len(xs),len(ys)))
    pd = np.zeros(shape=(len(xs),len(ys)))
    rd = np.zeros(shape=(len(xs),len(ys)))
    count = 1
    final_count = float(len(xs) * len(ys))
    for i in range(len(xs)):
        for j in range(len(ys)):
            print("Calculating optimal maneuver for detection point ", xs[i], ", ", ys[j], ", 0")
            sol = optimize.optimal_maneuver(fish,
                                            detection_point_3D=(xs[i], ys[j], 0.0),
                                            max_iterations=(100 if FAST_TEST else DEFAULT_OPT_ITERATIONS),
                                            max_n=(30 if FAST_TEST else DEFAULT_OPT_N),
                                            suppress_output=True)
            if sol.activity_cost != CONVERGENCE_FAILURE_COST:
                ac[i,j] = sol.activity_cost
                pd[i,j] = sol.pursuit_duration
                rd[i,j] = sol.return_duration
            else:
                ac[i,j] = np.nan
                pd[i,j] = np.nan
                rd[i, j] = np.nan
            if IS_MAC:
                print(f"Solution {count} of {final_count:.0f}: For fl={fork_length:.1f} cm, v={velocity:.1f} cm/s, at x={xs[i]:.2f} and y={ys[j]:.2f}, energy cost is {sol.activity_cost:.5f} J and pursuit duration is {sol.pursuit_duration:.3f} and return duration is {sol.return_duration:.3f} s.")
            if count % 5 == 0 and not IS_MAC:
                db_execute("UPDATE maneuver_model_tasks SET progress={0} WHERE taskid={1}".format(count/final_count, taskid))
            count += 1 
    # Now, run quality control check on the ec and pd values, redoing calculation if they're too far off from their neighbors or came out np.nan the first time
    for table in [ac]: # base the check only on ac
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
                if len(notnan_neighbors) >= 3 or np.isnan(table[i,j]): # only do the neighbor-based QC check if there are enough "neighbors" (2 + current number) to check against the median
                    neighbor_median = np.median(notnan_neighbors) # median of 4 (corner), 6 (edge), or 9-value (center) block around present cell
                    ratio_to_median = table[i,j] / neighbor_median
                    worst_allowable_ratio = 3.0 # assume optimal solution wasn't found if solution differs from neighbors by factor of 3
                    if ratio_to_median < 1/worst_allowable_ratio or ratio_to_median > worst_allowable_ratio or np.isnan(table[i,j]):
                        for retry in (2,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4): # If we didn't get reasonable values the first time, try again with more rigorous but time-consuming algorithm parameters
                            if not IS_MAC: db_execute("UPDATE maneuver_model_tasks SET retries=retries+1 WHERE taskid={0}".format(taskid))
                            sol = optimize.optimal_maneuver(fish,
                                                            detection_point_3D=(xs[i], ys[j], 0.0),
                                                            max_iterations=(retry*DEFAULT_OPT_ITERATIONS),
                                                            max_n=(retry*DEFAULT_OPT_N),
                                                            suppress_output=True)
                            if sol.activity_cost < ac[i, j]:
                                ac[i,j] = sol.activity_cost
                                pd[i,j] = sol.pursuit_duration
                                rd[i,j] = sol.return_duration
                                ratio_to_median = table[i,j] / neighbor_median
                                if 1/worst_allowable_ratio <= ratio_to_median <= worst_allowable_ratio:
                                    break
                        if np.isnan(table[i,j]):
                            print(f"Retries still produced NaN activity cost for x={xs[i]}, y={ys[j]} with fl={fork_length}, velocity={velocity}.")
                        if ratio_to_median < 1/worst_allowable_ratio or ratio_to_median > worst_allowable_ratio:
                            print(f"Retries to match neighbors failed for x={xs[i]}, y={ys[j]} with fl={fork_length}, velocity={velocity}, ratio_to_median={ratio_to_median}.")
                            if not IS_MAC: db_execute(f"UPDATE maneuver_model_tasks SET has_failed_retries=1 WHERE taskid={taskid}")
    # Count up final number of NaNs if any, using the activity cost table
    nan_count = np.count_nonzero(np.isnan(ac))
    if nan_count > 0 and not IS_MAC:
        db_execute(f"UPDATE maneuver_model_tasks SET nan_count={nan_count} WHERE taskid={taskid}")
    # Now add on the mirror image of the first 4 columns to each extrapolation, with negative y values, to facilitate smooth interpolation near y=0
    ys = np.concatenate([np.flip(-ys[:4], axis=0), ys])
    ac = np.concatenate([np.flip(ac[:,:4], axis=1), ac], axis=1)
    pd = np.concatenate([np.flip(pd[:,:4], axis=1), pd], axis=1)
    rd = np.concatenate([np.flip(rd[:,:4], axis=1), rd], axis=1)
    return ac, pd, rd, xs, ys

def save_cost_tables(ec, pd, rd, xs, ys, fl, v):
    if IS_MAC:
        base_folder = os.path.join(os.path.sep, 'Users', 'Jason', 'Desktop', 'Maneuver Sheet Test')
    else:
        base_folder = os.path.join(os.path.sep, 'home', 'alaskajn', 'maneuver_model_tables')
    folder = os.path.join(base_folder, "fl_{0}".format(fl), "v_{0}".format(v))
    if not os.path.exists(folder):
        os.makedirs(folder)    
    def save_cost_table(costs, xs, ys, filename):
        costs_labeled = np.insert(np.insert(costs, 0, xs, axis=1), 0, np.insert(ys, 0, np.nan), axis=0) # add y labels along horizontal axis, x along vertical axis
        np.savetxt(os.path.join(folder, filename), costs_labeled, fmt='%.7f,')
    save_cost_table(ec, xs, ys, 'activity_cost.csv')
    save_cost_table(pd, xs, ys, 'pursuit_duration.csv')
    save_cost_table(rd, xs, ys, 'return_duration.csv')

select_query = "SELECT taskid, fork_length, velocity FROM maneuver_model_tasks WHERE start_time IS NULL"
task_data = db_execute(select_query).fetchone()
while task_data is not None:
    taskid, fork_length, velocity = task_data
    instance_id = 'my_laptop' if IS_MAC else uname()[1]
    print("Beginning task {0}.".format(taskid))
    if not IS_MAC: db_execute("UPDATE maneuver_model_tasks SET start_time=NOW(), machine='{1}', progress=0.0 WHERE taskid={0}".format(taskid, instance_id))
    ac, pd, rd, xs, ys = calculate_cost_tables(fork_length, velocity, taskid)
    save_cost_tables(ac, pd, rd, xs, ys, fork_length, velocity)
    if not IS_MAC: db_execute("UPDATE maneuver_model_tasks SET completion_time=NOW(), progress=NULL WHERE taskid={0}".format(taskid))
    task_data = db_execute(select_query).fetchone() if not FAST_TEST else None
db.close()

