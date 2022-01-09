# This file is meant to centralize and explain all the sensitivity analysis adjustments, so it's easy to
# verify when they're all off and the model is running normally, and so it's easy to play around with them
# for the manuscript. The effect of each one should be carefully described in the comments.

#---------------------------------------------------------------------------------------------------------------#
#------------------------------- Actual physical/biological model parameters -----------------------------------#
#---------------------------------------------------------------------------------------------------------------#

# Multiple by which the cost of steady swimming exceeds that of coasting
ALPHA = 4 # standard value 4
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------- Numerical algorithm settings -----------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

# Multiple by which maximum allowed turn radius exceeds the minimum allowed turn radius
# There is no physical limit to this maximum, so it's only used to constrain the search space for the optimal
# maneuver and keep the numerical algorithm more efficient (not wasting cycles evaluating unlikely solutions).
# Optimal turn radii average 1.2 to 3.9 times the minimum turn radius depending on fish size and which turn,
# but some situations were bumping up against the previous maximum of 15.
MAX_TURN_RADIUS_MULTIPLE = 50

# Similar to the max turn radius, this parameter exists only to constrain the numerical algorithm and avoid
# wasting CPU cycles on unlikely parts of the solution space.
MAX_FINAL_TURN_X_LENGTH_MULTIPLE = 10

# Value beyond which exp(value) overflows to inf in float64 arithmetic. I'm using using 650 to pad it, vs real
# limit near 750, so other operations that might increase the number don't exceed 750 and overflow.
EXP_OVERFLOW_THRESHOLD = 650

# Controls for the use of Newton's Method in calculating the final straight
MAX_NEWTON_ITERATIONS = 300 # normally only takes a few, but a high limit reduces failures to converge on odd maneuvers
CONVERGENCE_TOLERANCE_T_A_BOUNDS = 1e-7
CONVERGENCE_TOLERANCE_SEGMENT_B = 1e-7

# "Energy cost" of a candidate maneuver that fails to converge, so it's weeded out in the optimization function.
CONVERGENCE_FAILURE_COST = 99999999999

#---------------------------------------------------------------------------------------------------------------#
#--------------------------------------- Sensitivity analysis settings -----------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

# Multiplier on the amount by which turn_factor exceeds 1. Set to 0 for no turn multiplier, 0.5 to halve the
# effect, 2.0 to double it, etc. This doesn't half or double the factor itself, but its multiplicative effect.
SENSITIVITY_TURN_FACTOR_MULTIPLIER = 1.0

# Multiplier on the amount by which the unsteady swimming webb_factor exceeds 1, similar to above.
SENSITIVITY_WEBB_FACTOR_MULTIPLIER = 1.0 # todo tried 0.5, next 2.0

#---------------------------------------------------------------------------------------------------------------#
#--------------------------------------- Optimization Model Defaults-- -----------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

DEFAULT_OPT_N = 200
DEFAULT_OPT_ITERATIONS = 300

SLOW_OPT_N = 300
SLOW_OPT_ITERATIONS = 1000

#---------------------------------------------------------------------------------------------------------------#
#--------------------------------------- Sensitivity analysis to-do --------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

# - Test ALPHA of 3 and 5
# - Test SENSITIVITY_TURN_FACTOR_MULTIPLIER of 0.5 and 2.0
# - Test SENSITIVITY_WEBB_FACTOR_MULTIPLIER of 0.5 and 2.0
# - Test optimizing maneuver vs maneuver + opportunity costs
# - Test enabling wait time (wait times are ideally predicted near 0 because we base estimated detections on start of motion)
