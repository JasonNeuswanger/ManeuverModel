import numpy as np
import pymysql
from maneuvermodel import maneuveringfish, optimize
from scipy.interpolate import RectBivariateSpline

# had to sys.path.append("/Users/Jason/Dropbox/Drift Model Project/Calculations/driftmodeldev/") when it stopped finding the module

#-----------------------------------------------------------------------------------------------------------------------
#                                                    BASIC SUMMARY
#-----------------------------------------------------------------------------------------------------------------------

# This file is designed to explore the relevant boundaries for the interpolation tables, erring on the side of making
# their coverage comfortably more expansive than needed. It includes justifications for the dimensions of the tables as
# well as which tables are available, and it places job descriptions for the table calculator in a database.

# Data will eventually be organized by subfolders by fork length and current speed, with files for response variable
# (activity cost and pursuit duration) within each. Within those files, rows represent the x direction (first column
# being x coordinate labels) and columns the y direction (first row being y labels). Tables are provided for the
# positive half of the x-y planes; maneuvers elsewhere ar found by rotating into that plane.

# We save activity cost and pursuit duration, which are the relevant numbers to plug into another foraging model,
# but this must be done carefully to make sure SMR is also accounted for when calculating net rate of energy intake.
# The most relevant "handling time" for a model is pursuit duration, under the assumption that wait time and
# the return stage don't count as "handling" because the fish can be detecting and potentially pursuing other prey
# during that time. Only the pursuit is time lost with regard to other

# todo figure out how SMR would be apportioned during an application based on pursuit duration only, and whether I should include anything else

# todo note somewhere that handling time would exclude wait time, if using pursuit duration


#-----------------------------------------------------------------------------------------------------------------------
#                                  MODELING CHOICES, INCLUDING FIXED SETTINGS
#-----------------------------------------------------------------------------------------------------------------------

# It isn't feasible to precalculate values for every conceivable combination of settings, so we instead choose the most
# likely applications and vary only the quantities mentioned in the Basic Summary.
#
# Wait times are enabled to give accurate energy costs for items detected far upstream. Our paper suggested that
# including wait times resulted in a worse fit to real data for the model, but that was in a test assuming the fish
# detected the item when it responded to it. We could not know when they actually first detected items or how long they
# really waited in such cases. However, from a theoretical standpoint, excluding wait time would result in excessively
# high predicted time/energy costs for items detected far upstream, which is probably not the case. To avoid penalizing
# more effective prey detection, we have to include wait time. Users wishing to exclude it can use this script to make
# their own, net set of tables.
#
# We use only energy cost and not total cost (including opportunity cost based on NREI), because it would require far
# more calculations to account for different levels of possible NREI, and we found that the model excluding opportunity
# costs was the best fit our maneuver data from real fish anyway.

#---------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                       BOUNDARY CALCULATIONS FOR THE TABLES
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# CURRENT SPEED
speeds_in_bodylengths = [attempt.mean_current_speed / attempt.fish.best_length for attempt in all_foraging_attempts]
max(speeds_in_bodylengths) # 6.83
np.percentile(speeds_in_bodylengths, 99) # 5.48
np.percentile(speeds_in_bodylengths, 95) # 4.04
np.percentile(speeds_in_bodylengths, 50) # 1.46

# Maneuvers at high current speeds in bodylengths/s were not uncommon.
# It wouldn't be too abnormal to have a 4 cm Chinook in 20 cm/s water (5 bodylengths)
# But it would be very abnormal to have a 50 cm Grayling in 150 cm/s water (3 bodylengths)


# 
# For distance:
# The absolute longest was 9.2 bodylengths, next longest 8.8, then 8.1, then a couple dozen.
# in the 7-6-5 range.  The 95th percentile is 2.9 bodylengths and 99th percentile is 4.9 bodylengths.
# We can probably assume some of the really long detections were extreme anomalies or misinterpretations of joint maneuvers.
#
#

# I'm putting the database query inside the second but not third level of the nested list, to strike the balance between
# query size and query count

# Size appropriate velocity max is designed to exceed realistic foraging for fish of a given size and then some, while
# still eliminating the work of calculating absurdly high velocities. max_velocity = 3.0 * fl + 50 # in cm/s and cm

fork_lengths = list(np.concatenate([np.arange(3,5,0.2),np.arange(5,10,0.5),np.arange(10,20,1),np.arange(20,57,2),np.arange(58,80,3)]))
time_per_number = 6.0   # seconds required to compute one cell in the spreadsheet
bytes_per_number = 9.8   # bytes required to store one cell in the spreadsheet
numbers_per_sheet = 999  # number of cells in each sheet, based on the resolution
max_instances = 25  # max number of virtual machines running calculations

queries = []
total_sheets = 0
total_bytes = 0
total_time = 0
all_velocities = list(np.concatenate([np.arange(1,19,2), np.arange(19, 40, 3), np.arange(40, 90, 5), np.arange(90,166,15)]))
for fl in fork_lengths:
    size_appropriate_velocities = [v for v in all_velocities if v < 3.0 * fl + 50]
    for v in size_appropriate_velocities:
        total_sheets += 1
        total_bytes += bytes_per_number * numbers_per_sheet
        total_time += time_per_number * numbers_per_sheet
        queries.append(f"INSERT INTO maneuver_model_tasks (fork_length, velocity) VALUES ({fl:.1f}, {v:.1f})")
total_bytes *= 2 # because there are 2 response variables
time_per_sheet = total_time / total_sheets
real_time = (total_time / 3600) / max_instances
print("Total calculation predicted to generate {0} sheets in {1:.1f} cpu-hours ({2:.1f} min/sheet, {4:.1f} hours for {5} instances) taking {3:.1f} mb of space.".format(total_sheets, total_time/3600.0, time_per_sheet/60.0, total_bytes/(1024.0*1024.0), real_time, max_instances))

# Actually generate the to-do list in the database -- ONLY DO THIS ONCE unless I am resetting the whole thing!
# If I do reset the whole thing, I need to do ALTER TABLE maneuver_model_tasks AUTO_INCREMENT = 1
db = pymysql.connect(host="troutnut.com", port=3306, user="jasonn5_calibtra", passwd="aVoUgLJyKo926", db="jasonn5_calibration_tracking", autocommit=True)
cursor = db.cursor()
for i in range(len(queries)):
   print("Running query {0} of {1}.".format(i,len(queries)))
   exq = cursor.execute(queries[i])
db.close()

#---------------------------------------------------------------------------------------------------------------------------------------------------------
#                           CHECKING ACCURACY WITH WHICH SPLINE PREDICTS DIRECT MODEL PREDICTIONS and optimizing spline parameters
#---------------------------------------------------------------------------------------------------------------------------------------------------------
# This code refers to all kinds of global variables from other sheets and will be a nuisance to reuse. It's not really meant for that. Just look at the 
# commented results below instead.

import random

# get xs, ys, ec from maneuver_spreadsheet_creation.py
spl_ec_665 = RectBivariateSpline(xs, ys, ec)
spl_ec_deg5 = RectBivariateSpline(xs, ys, ec, kx=5, ky=5) # kx, ky = spline degree... both 1 and 5 worked worse
spl_efc_smooth = RectBivariateSpline(xs, ys, ec, s=2) # s = smoothing... looked good on plots but bad for results


# worked very poorly with x=-12, y=5
errors_A = []
errors_B = []
for i in range(1,300):
    # Uniform test throughout the possible reaction distances or an inner subset thereof
    #distfact=1
    #testx = random.uniform(min(xs)/distfact,max(xs)/distfact)
    #testy = random.uniform(min(ys)/distfact,max(ys)/distfact)
    # Test weighted to actual reaction distances (in bodylengths) by choosing randomly from real fish data from all_foraging_attempts in Maneuver Paper Calculations.py
    attempt = random.choice(all_foraging_attempts)
    testx = fork_length * attempt.reaction_vector[0] / attempt.fish.best_length
    testy = fork_length * attempt.lateral_reaction_distance / attempt.fish.best_length
    test_energy_model = optimize.optimal_maneuver(fish, detection_point_3D = (testx, testy, 0.0), popsize=4, variant_scale=1.5, mixing_ratio=3.0, iterations=4500, use_starting_iterations=True, num_starting_populations=12, num_starting_iterations=500).dynamics.activity_cost
    errors_A.append(100 * abs(spl_ec_665.ev(testx, testy) - test_energy_model) / test_energy_model) # percent error in log spline model
    errors_B.append(100 * abs(spl_ec_416.ev(testx, testy) - test_energy_model) / test_energy_model) # percent error in lin spline model
print("Mean A (665) error is {0:.2f}, median {1:.2f}, 95th percentile {2:.2f}, max {3:.2f}. Mean B (416) error is {4:.2f}, median {5:.2f}, 95th percentile is {6:.2f}, max is {7:.2f}.".format(np.mean(errors_A), np.median(errors_A), percentile(errors_A,95), max(errors_A), np.mean(errors_B), np.median(errors_B), percentile(errors_B,95), max(errors_B)))
# Within the region closest to the fish (distfact=5, inner 20 % of interpolated range), logspline works better:
# Mean logspline error is 0.00897633464221, median 0.0044701166147, max 0.0864545485659. Mean linspline error is 0.0399583067863, median 0.0247217810159, max 0.125442996728.
# For the inner 50% of extrapolated range (distfact=2):
# Mean logspline error is 0.0484298747137, median 0.0206530488527, max 0.355140338201. Mean linspline error is 0.0270808114457, median 0.0120931308594, max 0.198272165424.
# For the overall extrapolated region:
# Mean logspline error is 0.128979262139, median 0.0599612297048, max 0.692983284836. Mean linspline error is 0.0114831437642, median 0.00342595840382, max 0.103078836909.
# AFTER changing spline degree to 1 (linear)
# distfact = 1
# Mean logspline error is 0.0218975898111, median 0.0113214302154, max 0.138234911992. Mean linspline error is 0.0125631421346, median 0.0035485055496, max 0.140522637627
# distfact = 2
# Mean logspline error is 0.0368812920421, median 0.0152525418311, max 0.198044035521. Mean linspline error is 0.0243640316147, median 0.0150866681404, max 0.147206404992.
# distfact = 5
# Mean logspline error is 0.0137129814445, median 0.00836141861284, max 0.105787513153. Mean linspline error is 0.0393639917989, median 0.0210805421877, max 0.176547064472.
# NEXT TEST: Increase resolution to 312 pts, and start using percent errors instead of regular errors, cover full realistic ranges
# distfact = 1
# Mean logspline error is 6.61230781457, median 2.52211224078, max 81.438706048. Mean linspline error is 2.70886911207, median 0.140402890291, max 44.1715921604.
# distfact = 2
# Mean logspline error is 5.13227646421, median 1.11771126799, max 64.1222831046. Mean linspline error is 15.9565636692, median 1.12503901849, max 632.523260832.
# distfact = 5
# Mean logspline error is 5.95691325441, median 3.00161632168, max 82.3370962708. Mean linspline error is 8.76483038921, median 3.49808649369, max 84.3056318184.
# Now using data from actual fish within the range where they're really doing stuff (N=300 for calculating these stats)
# Mean logspline error is 6.22782654855, median 3.72433730964, max 42.6818144461. Mean linspline error is 15.4806385555, median 4.82063948279, max 323.389534253
# That is, dare I say, tolerable... but let's see if increasing iterations helps.
# Now testing two models with either 312-point or 1248-point grids (both on log scale) used to create the spline
# Mean 1248-pt error is 5.32348294783, median 1.49117256539, max 47.2459070847. Mean 312-pt error is 6.28938249563, median 3.69462175484, max 38.4551959213.
# Not that much of an improvement. What if instead of increasing resolution so much, we increase iterations to smooth things out?
# Using 12x500+4500 (both for extrapolation and test)
# Switching from 6x500 to 12x500 for the preliminary solution really didn't help much.
# Now trying out smoothing on the interpolation in model 312bs (smoothed at s=2 in RectBivariateSpline)
# Mean 312b error is 5.9512430459, median 4.12127444535, max 66.6321313342. Mean 312bs error is 36.5312610831, median 22.2841087968, max 395.585373521.
# Okay, so the smoothing looks good on the plots, but it's actually awful.
# Next task is to see how much difference spline order makes, model 312bl, with linear instead of cubic spline interpolation:
# Mean 312b error is 5.41455284206, median 3.74412497518, max 40.8658187381. Mean 312bl error is 35.3688630676, median 23.6634535005, max 488.033873058.
# Okay, so the cubic spline is VASTLY better than linear interpolation.
# Next test is with better interpolation grid that isn't so heavily weighted toward zero (starting log scale at 1.0 instead of 0.1, but adding in some short ones)
# Mean 416 error is 6.49, median 3.00, 95th percentile 22.91, max 156.98. Mean 312 error is 6.54, median 4.08, max 20.75, 95th percentile 72.64.
# Also can't hurt to give quintic splines a try... okay, they're no good.
# Mean A (416) error is 5.12, median 2.07, 95th percentile 20.01, max 118.43. Mean B (416q) error is 18.64, median 7.24, max 50.52, 95th percentile 1172.86.
# Next result comparing 665 (beefing up resolution on the -x and +y quadrant) to the 416 result (which had fairly useless, beefed-up resolution in the +x +y quadrant)
# Mean A (665) error is 4.76, median 1.84, 95th percentile 16.38, max 96.79. Mean B (416) error is 5.23, median 2.11, 95th percentile is 18.74, max is 67.20.
# Calling that one good enough.

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
xnew, ynew = np.ogrid[xmin:xmax:1000j, ymin:ymax:1000j] #meshgrid(xx, yy)
znew_a = spl_ec_312(xnew, ynew)
znew_b = spl_ec_416(xnew, ynew)
znew_c = spl_ec_1248(xnew, ynew)
im1 = ax1.imshow(znew_a, cmap='viridis', extent=(ymin,ymax,xmin,xmax))
im2 = ax2.imshow(znew_b, cmap='viridis', extent=(ymin,ymax,xmin,xmax))
im3 = ax3.imshow(znew_c, cmap='viridis', extent=(ymin,ymax,xmin,xmax))
ax1.set_title('312')
ax2.set_title('416')
ax3.set_title('1248')
plt.show()


#---------------------------------------------------------------------------------------------------------------------------------------------------------
#                                       CHECKING EFFECT OF POINTS USED TO CONSTRUCT SPLINES ON SPLINE CALL SPEED (NO EFFECT)
#---------------------------------------------------------------------------------------------------------------------------------------------------------


# The following test, conducted using splines based on 25, 100, or 900 points, shows that the number of points used to construct the spline has 
# no detectable effect on very short amount of time required to get call the spline function (about 3.5 microseconds). This means the only limiting
# factor in the resolution of grid we're using to create the splines is how much computing time & storage space we have to calculate/hold the grids.
# Note that the real optimal energy cost directly calculated from the model with tons of iterations is about 0.6127 J.
#from timeit import timeit
#timeit("spl_ec25(-2.55, 16.38)", setup="from __main__ import spl_ec25", number = 100000)/100000
#timeit("spl_ec100(-2.55, 16.38)", setup="from __main__ import spl_ec100", number = 100000)/100000
#timeit("spl_ec900(-2.55, 16.38)", setup="from __main__ import spl_ec900", number = 100000)/100000



#---------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                      FINAL INTERPOLATION CODE (FROM CSV FILE)
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# Note: Using non-default options (for spline degree or smoothing) does not improve interpolation quality, as shown above.

def maneuver_cost_interpolation(file_to_interpolate):
    filedata = np.genfromtxt(file_to_interpolate, delimiter=',')
    xs = filedata[1:,0]      # In sample data, x should go from -24 to 15
    ys = filedata[0,1:-1]    # In sample data, y should go from .001 to 18
    data = filedata[1:,1:-1] # We have to trim off the last element of each row (nan) because of the trailing comma when saving
    return RectBivariateSpline(xs, ys, data)  
     
def interpolate_maneuver_cost(detection_point_3D, interpolation_function):
    x, y, z = detection_point_3D
    R = np.sqrt(y*y + z*z)
    matrix_2Dfrom3D = np.array([[1,0,0],[0,y/R,z/R],[0,-z/R,y/R]]) # matrix to rotate the 3-D detection point about the x-axis into the x-y plane
    (xrot, yrot) = matrix_2Dfrom3D.dot(np.array(detection_point_3D))[0:2] # 2-D detection point to use for the model, not yet sign-adjusted
    yrot = abs(yrot) # flip negative-y values to the positive half of the x-y plane for cost calculations
    return interpolation_function(xrot, yrot)[0,0]

testpt = (-15, 5, 3)        
interp_ec = maneuver_cost_interpolation("/Users/Jason/Dropbox/Drift Model Project/Calculations/driftmodeldev/maneuvermodel/sample_data/interpolation_sample_data_energy_cost.csv")
interpolate_maneuver_cost(testpt, interp_ec)

from timeit import timeit
timeit("interpolate_maneuver_cost(testpt, interp_ec)", setup="from __main__ import interpolate_maneuver_cost, testpt, interp_ec", number = 1000)/1000

# Could use Dill to serialize the interpolations, too
# https://stackoverflow.com/questions/23997431/is-there-a-way-to-pickle-a-scipy-interpolate-rbf-object

#---------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                      TRANSFER THIS CODE TO AMAZON INSTANCES
#---------------------------------------------------------------------------------------------------------------------------------------------------------
# Had to install ssh-askpass for this to work: https://github.com/theseal/ssh-askpass
# Make sure to use the .pem version of the keyfile straight from Amazon, not the Putty .ppk version
# Also have to chmod 400 the .pem file before it can be used.

#import os
#ec2_private_key = "'/Users/Jason/Dropbox/Amazon AWS/NeuswangerManeuverModelS3Instances.pem'"
#module_folder = "'/Users/Jason/Dropbox/Drift Model Project/Calculations/driftmodeldev/maneuvermodel'"
#creation_script = "'/Users/Jason/Dropbox/Drift Model Project/Calculations/driftmodeldev/maneuver_spreadsheet_creation.py'"
#ec2_server_address = "ec2-34-216-120-173.us-west-2.compute.amazonaws.com" # NEED TO FREQUENTLY UPDATE
#remote_folder = "~"
#os.system("scp -r -i {0} {1} ec2-user@{2}:{3}".format(ec2_private_key, module_folder, ec2_server_address, remote_folder))
#command2 = "scp -i {0} {1} ec2-user@{2}:{3}".format(ec2_private_key, creation_script, ec2_server_address, remote_folder)
##print(command2) # WHEN THE SERVER IP CHANGES, NEED TO RE-RUN PRINTED SCP COMMAND IN TERMINAL TO GET AROUND ssh-askpass ERROR
#os.system(command2)

#---------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                      CLEAN UP INTERRUPTED PROCESSES
#---------------------------------------------------------------------------------------------------------------------------------------------------------

db = pymysql.connect(host="troutnut.com", port=3306, user="jasonn5_calibtra", passwd="aVoUgLJyKo926", db="jasonn5_calibration_tracking", autocommit=True)
#db = pymysql.connect(host="maneuver-model-tasks.crtfph6ctn2x.us-west-2.rds.amazonaws.com", port=3306, user="manmoduser", passwd="x]%o4g28", db="maneuver_model_tasks", autocommit=True)
cursor = db.cursor()
cursor.execute("UPDATE maneuver_model_tasks SET start_time = NULL WHERE end_time IS NULL")
db.close()