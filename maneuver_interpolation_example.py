import numpy as np
import os
from scipy.interpolate import RectBivariateSpline

# This file shows how to interpolate maneuver costs from a precalculated table using scipy's cubic spline interpolation
# function for rectangular grids. The interpolation tables are only for the positive-y side of the x-y plane; requests
# for interpolation at other 3-D coordinates require rotating the problem into that plane, which is where the model does
# its internal calculations anyway. The interpolate_maneuver_cost() function below demonstrates fairly simple geometry
# from that rotation.

# REPLACE THIS WITH THE PATH TO THE INTERPOLATION TABLES ON YOUR SYSTEM
BASE_INTERPOLATION_FOLDER = "/Users/Jason/Dropbox/drift model project/Calculations/maneuvermodeldev/manever_model_tables/"

def maneuver_cost_interpolation(file_to_interpolate):
    file_contents = np.genfromtxt(file_to_interpolate, delimiter=',')
    x_values = file_contents[1:, 0]  # Load values on the x-axis
    y_values = file_contents[0, 1:-1]  # Load values on the y-axis
    data = file_contents[1:, 1:-1]  # We have to trim off the x/y values and last element of each row (nan) because of the trailing comma when saving
    return {'interpolation': RectBivariateSpline(x_values, y_values, data), 'x_values': x_values, 'y_values': y_values}

def closest_interpolation_file(fork_length, velocity):
    """ Searches the BASE_INTERPOLATION_FOLDER containing all the maneuver model tables and finds the one closest to
        the specified fork length and mean maneuver velocity. Returns a tuple of interpolating functions for the two
        responses. If files cannot be found close enough to the requested parameters, then you're operating in
        territory where real fish wouldn't likely be maneuvering. Arbitrary very high energy and tiem costs are returned
        so you still get an answer, but those unrealistic scenarios are heavily penalized in any applications."""
    available_fl = np.array(
        [float(foldername.split("fl_", 1)[1]) for foldername in os.listdir(BASE_INTERPOLATION_FOLDER)])
    nearest_fl = available_fl.flat[np.abs(available_fl - fork_length).argmin()]
    fl_folder = os.path.join(BASE_INTERPOLATION_FOLDER, 'fl_{0:.1f}'.format(nearest_fl))
    available_fv = np.array([float(foldername.split("fv_", 1)[1]) for foldername in os.listdir(fl_folder)])
    nearest_fv = available_fv.flat[np.abs(available_fv - focal_velocity).argmin()]
    fv_folder = os.path.join(fl_folder, 'fv_{0:.1f}'.format(nearest_fv))
    available_pv = np.array([float(foldername.split("pv_", 1)[1]) for foldername in os.listdir(fv_folder)])
    nearest_pv = available_pv.flat[np.abs(available_pv - prey_velocity).argmin()]
    pv_folder = os.path.join(fv_folder, 'pv_{0:.1f}'.format(nearest_pv))
    activity_cost_file = os.path.join(pv_folder, 'activity_cost.csv')
    pursuit_duration_file = os.path.join(pv_folder, 'pursuit_duration.csv')
    files_available = True
    if abs(nearest_fl - fork_length) > 2:
        print("Maneuver interpolation tables unavailable for fork length {0} cm, using {1} instead.".format(fork_length,
                                                                                                            nearest_fl))
    if abs(nearest_fv - focal_velocity) > 3 or abs(nearest_pv - prey_velocity) > 3:
        print(
            "Maneuver interpolation tables unavailable because a fish of fork length {0} cm cannot effectively maneuver at focal velocity {1:.1f} cm/s "
            "and prey velocity {2:.1f} cm/s.".format(fork_length, focal_velocity, prey_velocity))
        files_available = False
    if not files_available:  # give extremely high costs to impossible maneuvers with no interpolation files available
        activity_cost_interpolation = lambda x, z: np.array([[10000]])
        pursuit_duration_interpolation = lambda x, z: np.array([[10000]])
    else:
        activity_cost_interpolation = maneuver_cost_interpolation(activity_cost_file)
        pursuit_duration_interpolation = maneuver_cost_interpolation(pursuit_duration_file)
    return activity_cost_interpolation, pursuit_duration_interpolation


def interpolate_maneuver_cost(detection_point_3D, interpolation_function):
    """ Readers of our manuscript might note that the conversion from 3-D to 2-D coordinates here is much simpler than the matrix formula given
        in equation 1 of the paper. For the 3-D to 2-D case, that matrix formula simplifies to what we're doing here. The reason the paper
        mentions the matrix is the reverse transformation, from 2-D back into 3-D, requires its inverse. But in this function, that is not needed."""
    x, y, z = detection_point_3D
    R = np.sqrt(y * y + z * z)
    return interpolation_function(x, R)[0, 0]


test_point = (-15, 5, 3)
fork_length = 6.2  # cm
velocity = 15
activity_cost_interpolation_function, pursuit_duration_interpolation_function = closest_interpolation_file(fork_length,
                                                                                                           velocity)

test_maneuver_activity_cost = interpolate_maneuver_cost(test_point, activity_cost_interpolation_function)
test_maneuver_pursuit_duration = interpolate_maneuver_cost(test_point, pursuit_duration_interpolation_function)

print("Interpolated maneuver energy cost is {0:.6f} J and pursuit duration is {1:.3f} s.".format(
    test_maneuver_activity_cost, test_maneuver_pursuit_duration))

# todo clean up the stuff above only after I have the full library of calculations done

#-----------------------------------------------------------------------------------------------------------------------
#
#
#   TESTING AND PLOTTING
#
#
#-----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

plot_fish_fork_length = 3.0
fine_max_dist = 5 * plot_fish_fork_length

ac_interp = maneuver_cost_interpolation("/Users/jason/Desktop/Maneuver Sheet Test/fl_3.0/v_1.0/activity_cost.csv")
pd_interp = maneuver_cost_interpolation("/Users/jason/Desktop/Maneuver Sheet Test/fl_3.0/v_1.0/pursuit_duration.csv")
# Create xy grid based on actual interpolation points
xs = ac_interp['x_values']
ys = ac_interp['y_values'][4:]  # removing mirrored y-values for now, will re-mirror whole half later
xg, yg = np.meshgrid(xs, ys)
zg_ac = np.empty(np.shape(xg))
zg_pd = np.empty(np.shape(xg))
# Create fine xy grid based on distance from fish within reasonable foraging zone (fine_max_dist)
xs_f = np.linspace(-fine_max_dist, fine_max_dist, 50)
ys_f = np.linspace(-fine_max_dist, fine_max_dist, 50)
xg_f, yg_f = np.meshgrid(xs_f, ys_f)
zg_ac_f = np.empty(np.shape(xg_f))
zg_pd_f = np.empty(np.shape(xg_f))

for i in range(len(xs)):
    for j in range(len(ys)):
        zg_ac[j, i] = interpolate_maneuver_cost((xs[i], ys[j], 0.0), ac_interp['interpolation'])
        zg_pd[j, i] = interpolate_maneuver_cost((xs[i], ys[j], 0.0), pd_interp['interpolation'])
xgd = np.vstack((xg, xg))
ygd = np.vstack((-np.flipud(yg), yg))
zgd_ac = np.vstack((np.flipud(zg_ac), zg_ac,))
zgd_pd = np.vstack((np.flipud(zg_pd), zg_pd))

for i in range(len(xs_f)):
    for j in range(len(ys_f)):
        zg_ac_f[j, i] = interpolate_maneuver_cost((xs_f[i], ys_f[j], 0.0), ac_interp['interpolation'])
        zg_pd_f[j, i] = interpolate_maneuver_cost((xs_f[i], ys_f[j], 0.0), pd_interp['interpolation'])

sns.set_style('white')
interpfig, ((ax_ac, ax_pd),(ax_ac_f, ax_pd_f)) = plt.subplots(2, 2, facecolor='w', figsize=(9, 7.5), dpi=132)
cf_ac = ax_ac.contourf(xgd, ygd, zgd_ac, 100, cmap='inferno')
cf_pd = ax_pd.contourf(xgd, ygd, zgd_pd, 100, cmap='viridis')
cf_ac_f = ax_ac_f.contourf(xg_f, yg_f, zg_ac_f, 100, cmap='inferno')
cf_pd_f = ax_pd_f.contourf(xg_f, yg_f, zg_pd_f, 100, cmap='viridis')
# ax_ac.scatter(xgd, ygd, s=0.25, c='k') # Can use these to overlay the grid of points from the interpolation table
# ax_pd.scatter(xgd, ygd, s=0.25, c='k')
interpfig.colorbar(cf_ac, ax=ax_ac, shrink=0.9)
interpfig.colorbar(cf_pd, ax=ax_pd, shrink=0.9)
interpfig.colorbar(cf_ac_f, ax=ax_ac_f, shrink=0.9)
interpfig.colorbar(cf_pd_f, ax=ax_pd_f, shrink=0.9)

ax_ac.set_title('Activity cost (J)')
ax_pd.set_title('Pursuit duration (s)')
ax_ac_f.set_title('Activity cost (J) closeup, rescaled')
ax_pd_f.set_title('Pursuit duration (s) closeup, rescaled')
ax_pd.text(0.15, 0.9, "Flow", ha="center", va="center", color='0.15', size=8, transform=ax_pd.transAxes)
ax_pd.arrow(0.27, 0.905, 0.15, 0.0, color='0.15', transform=ax_pd.transAxes, head_width=0.025)

for ax in (ax_ac, ax_pd, ax_ac_f, ax_pd_f):
    ax.plot((0, plot_fish_fork_length), (0, 0), marker=None, color=(0,1,0), linestyle='solid', linewidth=1)
    ax.set_aspect('equal')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')

#interpfig.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.95, wspace=0.5, hspace=0.5)
interpfig.tight_layout()
interpfig.show()