import numpy as np
import os
from scipy.interpolate import RectBivariateSpline

# This file shows how to interpolate maneuver costs from a precalculated table using scipy's cubic spline interpolation function for
# rectangular grids. The interpolation tables are only for the positive-y side of the x-y plane; requests for interpolation at other
# 3-D coordinates require rotating the problem into that plane, which is where the model does its internal calculations anyway. The
# interpolate_maneuver_cost() function below demonstrates fairly simple geometry from that rotation.
#
# Note that some of the provided interpolation tables contain NaN (Not a Number) values, representing maneuvers for which the model
# could not find a solution, i.e. asking the fish to do the impossible. These should not come up very often in realistic applications,
# but they could cause problems for the interpolating functions when using those tables. This would require working in a more realistic
# subregion.

# REPLACE THIS WITH THE PATH TO THE INTERPOLATION TABLES ON YOUR SYSTEM
BASE_INTERPOLATION_FOLDER = "/Users/Jason/Dropbox/drift model project/Calculations/maneuvermodeldev/manever_model_tables/"

def maneuver_cost_interpolation(file_to_interpolate):
    file_contents = np.genfromtxt(file_to_interpolate, delimiter=',')
    xs = file_contents[1:,0]      # In sample data, x should go from -167.7 to 167.7, but it differs for other precalculated tables based on fish size
    ys = file_contents[0,1:-1]    # In sample data, y should go from -2.63 to 167.7 (all use will be positive, but reflecting across y=0 prevents edge artifacts near y=0
    data = file_contents[1:,1:-1] # We have to trim off the x/y values and last element of each row (nan) because of the trailing comma when saving
    return RectBivariateSpline(xs, ys, data)

def closest_interpolation_file(fork_length, focal_velocity, prey_velocity):
    """ Searches the BASE_INTERPOLATION_FOLDER containing all the maneuver model tables and finds the one closest to a the specified fork length,
        temperature, and focal current speed. Returns a tuple of interpolating functions for the two responses. If files cannot be found close
        enough to the requested parameters, that means the model predicted the fish could not maneuver under those conditions. In those cases,
        very high energy and time costs are returned so you still get an answer, but those unrealistic scenarios are heavily penalized in any
        optimal foraging or other optimization applications."""
    available_fl = np.array([float(foldername.split("fl_", 1)[1]) for foldername in os.listdir(BASE_INTERPOLATION_FOLDER)])
    nearest_fl = available_fl.flat[np.abs(available_fl - fork_length).argmin()]
    fl_folder = os.path.join(BASE_INTERPOLATION_FOLDER, 'fl_{0:.1f}'.format(nearest_fl))
    available_fv = np.array([float(foldername.split("fv_", 1)[1]) for foldername in os.listdir(fl_folder)])
    nearest_fv = available_fv.flat[np.abs(available_fv - focal_velocity).argmin()]
    fv_folder = os.path.join(fl_folder, 'fv_{0:.1f}'.format(nearest_fv))
    available_pv = np.array([float(foldername.split("pv_", 1)[1]) for foldername in os.listdir(fv_folder)])
    nearest_pv = available_pv.flat[np.abs(available_pv - prey_velocity).argmin()]
    pv_folder = os.path.join(fv_folder, 'pv_{0:.1f}'.format(nearest_pv))
    energy_cost_file = os.path.join(pv_folder, 'energy_cost.csv')
    pursuit_duration_file = os.path.join(pv_folder, 'pursuit_duration.csv')
    files_available = True
    if abs(nearest_fl - fork_length) > 2:
        print("Maneuver interpolation tables unavailable for fork length {0} cm, using {1} instead.".format(fork_length, nearest_fl))
    if abs(nearest_fv - focal_velocity) > 3 or abs(nearest_pv - prey_velocity) > 3:
        print("Maneuver interpolation tables unavailable because a fish of fork length {0} cm cannot effectively maneuver at focal velocity {1:.1f} cm/s "
            "and prey velocity {2:.1f} cm/s.".format(fork_length, focal_velocity, prey_velocity))
        files_available = False
    if not files_available:  # give extremely high costs to impossible maneuvers with no interpolation files available
        energy_cost_interpolation = lambda x, z: np.array([[10000]])
        pursuit_duration_interpolation = lambda x, z: np.array([[10000]])
    else:
        energy_cost_interpolation = maneuver_cost_interpolation(energy_cost_file)
        pursuit_duration_interpolation = maneuver_cost_interpolation(pursuit_duration_file)
    return energy_cost_interpolation, pursuit_duration_interpolation

def interpolate_maneuver_cost(detection_point_3D, interpolation_function):
    """ Readers of our manuscript might note that the conversion from 3-D to 2-D coordinates here is much simpler than the matrix formula given
        in equation 1 of the paper. For the 3-D to 2-D case, that matrix formula simplifies to what we're doing here. The reason the paper
        mentions the matrix is the reverse transformation, from 2-D back into 3-D, requires its inverse. But in this function, that is not needed."""
    x, y, z = detection_point_3D
    R = np.sqrt(y*y + z*z)
    return interpolation_function(x, R)[0,0]

test_point = (-15, 5, 3)
fork_length = 6.2    # cm
focal_velocity = 15.5    # cm/s
prey_velocity = 16.5  # cm/s
# result energy cost should be around .OO49 for these
energy_cost_interpolation_function, pursuit_duration_interpolation_function = closest_interpolation_file(fork_length, focal_velocity, prey_velocity)

test_maneuver_energy_cost = interpolate_maneuver_cost(test_point, energy_cost_interpolation_function)
test_maneuver_pursuit_duration = interpolate_maneuver_cost(test_point, pursuit_duration_interpolation_function)

print("Interpolated maneuver energy cost is {0:.6f} J and pursuit duration is {1:.3f} s.".format(test_maneuver_energy_cost, test_maneuver_pursuit_duration))

# the problem sees to be having NaNs in the array, an even a single NaN borks RectBivariateSpline to produce all-NaN results.
# one solution could be to use a more expensive non-grid-based interpolation for all the ones that are NaN
# another would be to figure out how to make the model not produce NaN, then fill in those gaps
# maybe allow larger ranges of thrust or something? i need to understand how an NaN happens

# make it very clear in this file that it is ONLY activity cost