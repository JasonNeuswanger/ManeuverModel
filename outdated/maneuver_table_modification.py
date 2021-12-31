import os
import numpy as np
from maneuvermodel import maneuveringfish, optimize_cuckoo
from maneuvermodel.constants import *

BASE_INTERPOLATION_FOLDER = "/Users/Jason/Dropbox/drift model project/Calculations/maneuvermodeldev/manever_model_tables/"

# Function to fix broken files
# This will vary depending on what's being fixed. I'm using it to correct some
# scenarios that previously produced NaN values when we created the main tables. The model has been
# corrected to prevent those NaNs and give valid results in those areas without changing results elsewhere.
# This file could later be used to make other adjustments under specific conditions based on examining
# the files or the values of fl, fv, and pv inside this loop.

def fix_broken_files(fork_length, focal_velocity, prey_velocity, energy_cost_file, pursuit_duration_file):
    """ I initially use the file_contents variables below to read in the values from the input files, but then
        I also use the same arrays to build up the corrected output files. """
    # Load the files and prepare NaN indices
    print("Fixing NaNs for fl={0:.1f}, fv={1:.1f}, pv={2:.1f}.".format(fork_length, focal_velocity, prey_velocity))
    ec_file_contents = np.genfromtxt(energy_cost_file, delimiter=',')[:,:-1] # odd indexing avoids adding an extra NaN from
    pd_file_contents = np.genfromtxt(pursuit_duration_file, delimiter=',')[:,:-1] # the rightmost delimiter in each row
    xs = ec_file_contents[1:, 0]
    ys = ec_file_contents[0, 1:]
    ec_data = ec_file_contents[1:, 1:]
    nan_indices = np.argwhere(np.isnan(ec_data))
    # nan_indices = np.argwhere(ec_data) # if I wanted this function to rebuild the whole table
    # Prepare the fish for calculations
    max_thrust = 2.4 * fork_length + 40
    fish_mass = 0.0  # this input defaults the model to a rainbow trout length-mass regression
    fish_SMR = 0.0  # unused in this case
    fish_NREI = 0.0  # also unused in this case
    temperature = 10  # also unused in this case
    use_total_cost = False
    disable_wait_time = False
    fish = maneuveringfish.ManeuveringFish(fork_length, focal_velocity, fish_mass, temperature, fish_SMR, max_thrust, fish_NREI, use_total_cost, disable_wait_time)
    for x_ind, y_ind in nan_indices:
        # sol = optimize_cuckoo.optimal_maneuver_CS(fish, detection_point_3D=(xs[x_ind], ys[y_ind], 0.0), prey_velocity=prey_velocity, n=25, iterations=1500, p_a=0.25, suppress_output=False) # TEMP
        sol = optimize_cuckoo.optimal_maneuver_CS(fish, detection_point_3D=(xs[x_ind], ys[y_ind], 0.0), prey_velocity=prey_velocity, n=50, iterations=3000, p_a=0.25, suppress_output=False)
        assert(sol.energy_cost != CONVERGENCE_FAILURE_COST)
        ec_file_contents[x_ind+1, y_ind+1] = sol.energy_cost
        pd_file_contents[x_ind+1, y_ind+1] = sol.pursuit_duration
    np.savetxt("/Users/Jason/Desktop/ManeuverTemp/ec_results.csv", ec_file_contents, fmt='%.7f,')
    np.savetxt("/Users/Jason/Desktop/ManeuverTemp/pd_results.csv", pd_file_contents, fmt='%.7f,')

# test_ec_file = "/Users/Jason/Dropbox/drift model project/Calculations/maneuvermodeldev/manever_model_tables/fl_5.5/fv_1.0/pv_45.0/energy_cost.csv"
# test_pd_file = "/Users/Jason/Dropbox/drift model project/Calculations/maneuvermodeldev/manever_model_tables/fl_5.5/fv_1.0/pv_45.0/pursuit_duration.csv"
# fix_broken_files(5.5, 1.0, 45.0, test_ec_file, test_pd_file)

# fix_broken_files(3.4, 34.0, 34.0, test_ec_file, test_pd_file)


test_ec_file = "/Users/Jason/Dropbox/drift model project/Calculations/maneuvermodeldev/manever_model_tables/fl_5.5/fv_15.0/pv_13.0/energy_cost.csv"
test_pd_file = "/Users/Jason/Dropbox/drift model project/Calculations/maneuvermodeldev/manever_model_tables/fl_5.5/fv_15.0/pv_13.0/pursuit_duration.csv"
fix_broken_files(5.5, 15.0, 13.0, test_ec_file, test_pd_file)






test_ec_file = "/Users/Jason/Dropbox/drift model project/Calculations/maneuvermodeldev/manever_model_tables/fl_3.4/fv_34.0/pv_34.0/ec_results.csv"
test_ec_file = "/Users/Jason/Dropbox/drift model project/Calculations/maneuvermodeldev/manever_model_tables/fl_3.4/fv_34.0/pv_34.0/energy_cost.csv"
ec_file_contents = np.genfromtxt(test_ec_file, delimiter=',')[:,:-1]

def convert_indexed_table_to_xyz_values(table2d):
    """ This function takes an array like those read directly from my CSV files, in which the first row and column
        are x and y values and everything else is an NREI, into an array list of (x, y, NREI) values instead,
        suitable for transposing into vectors for interpolating functions etc."""
    return np.array([[(z[0], y, z[i]) for z in table2d[1:]] for i, y in enumerate(table2d[0][1:], 1)]).reshape(-1,3)

def has_crazy_values(energy_file_contents):
    tabledata = convert_indexed_table_to_xyz_values(energy_file_contents)
    tabledata = np.array([row for row in tabledata if not np.isnan(row[2])])
    x, y, d = tabledata.T
    rbfi = Rbf(x, y, d, smooth=3.0)  # radial basis function interpolator instance
    di = rbfi(x, y)  # calculate values based on smoothed
    rel_diffs = abs(di - d) / np.mean(d)
    return np.max(rel_diffs) > 0.5


from scipy.interpolate import Rbf
tabledata = convert_indexed_table_to_xyz_values(ec_file_contents)
tabledata = np.array([row for row in tabledata if not np.isnan(row[2])])
x, y, d = tabledata.T
rbfi = Rbf(x, y, d, smooth=3)  # radial basis function interpolator instance
di = rbfi(x, y)
rel_diffs = abs(di - d) / np.mean(d)
print("max rel_diff is ", np.max(rel_diffs)," and mean rel_diff is ", np.mean(rel_diffs))

# Plot the weird ones
# My first code ran this with a rel_diff threshold of 2.0 in has_crazy_values above and allowed for fv=1 or fv=3
# even for very high pv
# second run used rel_diff of 1.5
# third one will use 0.75
# rebuild code should respect y-mirroring
# third one used rel_diff of 0.75 but restricted fv to >= 0.25 pv
# fourth one used rel_diff of 0.5 but restricted to fv == pv

import matplotlib.pyplot as plt
plt.ioff()
available_fl = sorted(np.array([float(foldername.split("fl_", 1)[1]) for foldername in os.listdir(BASE_INTERPOLATION_FOLDER)]))
for fl in available_fl:
    print("Processing fork length ", fl)
    fl_folder = os.path.join(BASE_INTERPOLATION_FOLDER, 'fl_{0:.1f}'.format(fl))
    available_fv = sorted(np.array([float(foldername.split("fv_", 1)[1]) for foldername in os.listdir(fl_folder)]))
    for fv in available_fv:
        fv_folder = os.path.join(fl_folder, 'fv_{0:.1f}'.format(fv))
        available_pv = sorted(np.array([float(foldername.split("pv_", 1)[1]) for foldername in os.listdir(fv_folder)]))
        for pv in available_pv:
            if fv == pv: # THIS TIME, NARROW IT TO FV = PV
                pv_folder = os.path.join(fv_folder, 'pv_{0:.1f}'.format(pv))
                energy_cost_file = os.path.join(pv_folder, 'energy_cost.csv')
                pursuit_duration_file = os.path.join(pv_folder, 'pursuit_duration.csv')
                file_contents = np.genfromtxt(energy_cost_file, delimiter=',')[:,:-1]
                if has_crazy_values(file_contents):
                    data = file_contents[1:, 1:-1]
                    plt.clf()
                    plt.imshow(data[1:,1:], cmap="viridis")
                    plt.savefig("/Users/Jason/Desktop/maneuver_table_ims/fl_{0:.1f}_fv_{1:.1f}_pv_{2:.1f}_ec.png".format(fl, fv, pv))
plt.clf()
plt.ion()

# try to reproduce a weird one

# got an oddball at 3.0 / 31 / 31 -- the values for x=-108 and x=-57.8 are identical to an earlier value to 7 digits past the decimal for both EC and PD -- clearly a bad copy somehow
# several oddballs in 3.4 / 34 / 34 -- none of them are identical, some only coincidentally found in random other files
# got one in 3.4 / 28 / 45 -- also found only once at x=16.7046207, y=101.9727087, got ec=0.2402896 but expect a value near 0.68







# Browse the folder structure and identify files needing fixing

available_fl = np.array([float(foldername.split("fl_", 1)[1]) for foldername in os.listdir(BASE_INTERPOLATION_FOLDER)])
for fl in available_fl:
    fl_folder = os.path.join(BASE_INTERPOLATION_FOLDER, 'fl_{0:.1f}'.format(fl))
    available_fv = np.array([float(foldername.split("fv_", 1)[1]) for foldername in os.listdir(fl_folder)])
    for fv in available_fv:
        fv_folder = os.path.join(fl_folder, 'fv_{0:.1f}'.format(fv))
        available_pv = np.array([float(foldername.split("pv_", 1)[1]) for foldername in os.listdir(fv_folder)])
        for pv in available_pv:
            pv_folder = os.path.join(fv_folder, 'pv_{0:.1f}'.format(pv))
            energy_cost_file = os.path.join(pv_folder, 'energy_cost.csv')
            pursuit_duration_file = os.path.join(pv_folder, 'pursuit_duration.csv')
            file_contents = np.genfromtxt(energy_cost_file, delimiter=',')[:,:-1]
            data = file_contents[1:, 1:]
            # At this point, any check one wants to do on the file can be performed.
            files_are_broken = np.isnan(np.sum(data)) # in this case, files are "broken" if NaNs are found in the data
            if files_are_broken:
                fix_broken_files(fl, fv, pv, energy_cost_file, pursuit_duration_file)

