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
# but they could cause problems for the interpolating functions when using those tables and require selection of a more realistic
# subregion.

# REPLACE THIS WITH THE PATH TO THE INTERPOLATION TABLES ON YOUR SYSTEM
BASE_INTERPOLATION_FOLDER = "/Users/Jason/Dropbox/Drift Model Project/Calculations/driftmodeldev/maneuver-model-tables"

def maneuver_cost_interpolation(file_to_interpolate):
    file_contents = np.genfromtxt(file_to_interpolate, delimiter=',')
    xs = file_contents[1:,0]      # In sample data, x should go from -24 to 15, but it differs for other precalculated tables based on fish size
    ys = file_contents[0,1:-1]    # In sample data, y should go from .001 to 18
    data = file_contents[1:,1:-1] # We have to trim off the x/y values and last element of each row (nan) because of the trailing comma when saving
    return RectBivariateSpline(xs, ys, data)  

def closest_interpolation_file(fork_length, temperature, current_speed):
    """ Searches the BASE_INTERPOLATION_FOLDER containing all the maneuver model tables and finds the one closest to a the specified fork length, 
        temperature, and focal current speed. Returns a tuple of interpolating functions for the two responses. If files cannot be found close 
        enough to the requested parameters, that means the model predicted the fish could not maneuver under those conditions. In those cases, 
        very high energy and time costs are returned so you still get an answer, but those unrealistic scenarios are heavily penalized in any
        optimal foraging or other optimization applications.""" 
    available_fl = np.array([float(foldername.split("fl_", 1)[1]) for foldername in os.listdir(BASE_INTERPOLATION_FOLDER)])
    nearest_fl = available_fl.flat[np.abs(available_fl - fork_length).argmin()]
    fl_folder = os.path.join(BASE_INTERPOLATION_FOLDER, 'fl_{0:.1f}'.format(nearest_fl))
    available_fcs = np.array([float(foldername.split("fcs_", 1)[1]) for foldername in os.listdir(fl_folder)])
    nearest_fcs = available_fcs.flat[np.abs(available_fcs - current_speed).argmin()]
    fcs_folder = os.path.join(fl_folder, 'fcs_{0:.1f}'.format(nearest_fcs))
    available_t = np.array([float(foldername.split("t_", 1)[1]) for foldername in os.listdir(fcs_folder)])
    nearest_t = available_t.flat[np.abs(available_t - temperature).argmin()]
    t_folder = os.path.join(fcs_folder, 't_{0:.1f}'.format(nearest_t))
    energy_cost_file = os.path.join(t_folder, 'energy_cost.csv')
    pursuit_duration_file = os.path.join(t_folder, 'pursuit_duration.csv')
    files_available = True
    if abs(nearest_fl - fork_length) > 2:
        print("Maneuver interpolation tables unavailable for fork length {0} cm, using {1} instead.".format(fork_length, nearest_fl))
    if abs(nearest_fcs - current_speed) > 3 or abs(nearest_t - temperature) > 2:
        print("Maneuver interpolation tables unavailable because a fish of fork length {0} cm, at temperature {1} C "
            "cannot maneuver at current speed {2:.3f} cm/s.".format(fork_length, temperature, current_speed))
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
fork_length = 10    # cm
temperature = 14    # degrees
current_speed = 20  # cm/s
energy_cost_interpolation_function, pursuit_duration_interpolation_function = closest_interpolation_file(fork_length, temperature, current_speed)

test_maneuver_energy_cost = interpolate_maneuver_cost(test_point, energy_cost_interpolation_function)
test_maneuver_pursuit_duration = interpolate_maneuver_cost(test_point, pursuit_duration_interpolation_function)

print("Interpolated maneuver energy cost is {0:.6f} J and pursuit duration is {1:.3f} s.".format(test_maneuver_energy_cost, test_maneuver_pursuit_duration))

