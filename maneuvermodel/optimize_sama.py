# Cuckoo Search Optimizer
# Customized from the implementation in EvoloPy at https://github.com/7ossam81/EvoloPy/blob/master/CS.py
# Operates on numpy solution vectors scaled from 0 to 1

import math
import numpy as np
from .maneuver import maneuver_from_proportions
from .constants import CONVERGENCE_FAILURE_COST

def objective_function(fish, prey_velocity, xd, yd, p):
    maneuver = maneuver_from_proportions(fish, prey_velocity, xd, yd, p)
    return maneuver.fitness

##########################################################################

def SAMASearch(fish, xd, yd, **kwargs):
    """ Implements the self-learning antelopes migration algorithm of Lin et al 2019 """
    prey_velocity = kwargs.get('prey_velocity', fish.focal_velocity) # Prey velocity defaults to fish focal velocity if prey velocity not specified

    d = 12                                                  # Dimensions of the problem
    N = kwargs.get('N', 30)                                 # Number of antelopes in the herd
    M = kwargs.get('iterations', 1000)                      # Number of iterations
    alpha = kwargs.get('alpha', 0.95)
    beta = kwargs.get('beta', 1.05)
    gamma = kwargs.get('gamma', 0.04) # paper recommends 0.04
    N_ordinary_min = kwargs.get('N_ordinary_min', 10)        # Minimum number of ordinary antelopes & 1 - maximum number of scout antelopes
    N_ordinary_max = kwargs.get('N_ordinary_max', 20)       # Maximum number of ordinary antelopes & 1 - minimum number of scout antelopes
    assert N_ordinary_min < N
    assert N_ordinary_max < N

    R = np.full(M, np.nan)                                      # Ordinary antelope grazing radius
    sigma = np.full(M, np.nan)                                  # Scout antelope exploring amplitude
    mu = 0                                                      # Current running total of stagnation iterations
    N_ordinary = round((N_ordinary_min + N_ordinary_max) / 2)   # Starting number of ordinary antelopes

    X = np.full((M, d), np.nan)                 # Array holds the "grazing center" (best solution) at each iteration
    f = np.full(M, np.nan)                      # Array holds the fitness of the best solution at each iteration
    X[0] = np.random.rand(d)
    f[0] = objective_function(fish, prey_velocity, xd, yd, X[0])
    fEvals = 0                                  # Running total of objective function evaluations

    antelopes = np.random.rand(N, d)            # Initialize antelopes (ordinary/scout doesn't matter yet)

    for i in range(1, M):                                  # Begin main search loop
        R[i] = 1 if i < 2 else (R[i-1] * alpha if f[i-1] == f[i-2] else R[i-1] * beta)
        for j in range(N_ordinary):  # Ordinary antelope grazing
            antelopes[j] = X[i-1] + np.random.uniform(-R[i], R[i], d)
        sigma[i] = 1 if i < 2 else (sigma[i-1] * 0.5 if f[i-1] == f[i-2] else 1)  # note sigma[i] here is wrongly given as R[i] in original algorithm paper
        for j in range(N_ordinary, N):  # Scout antelope grazing
            antelopes[j] = X[i-1] + np.random.normal(0, sigma[i], d)
        # Use Numpy where shenanigans to replace parameter values outside the (0,1) range with random values inside the range
        bad_value_locations = np.where((antelopes < 0) | (antelopes > 1))
        antelopes[bad_value_locations] = np.random.rand(len(bad_value_locations[0]))
        # antelopes = np.clip(antelopes, 0, 1) # alternative way of trimming overruns, probably better for my problem, but the above is what's in the original algo paper
        # Evaluate the fitness of all antelopes and update the grazing center X[i] with the best fitness f[i]
        X[i] = X[i-1] # initialize grazing center with the previous one for comparison
        f[i] = f[i-1] # same for its fitness
        for k, antelope in enumerate(antelopes):
            fitness = objective_function(fish, prey_velocity, xd, yd, antelope)
            if fitness > f[i]:
                X[i] = antelope
                f[i] = fitness
                antelope_type = "ordinary" if k < N_ordinary else "scout"
                print("In iteration", i,"with N_o=", N_ordinary, ", R=",R[i],", sigma=",sigma[i],antelope_type,"antelope improved fitness f[i] to", f[i])#,"and antelope\nX[i]=", X[i])
        # Conduct self-learning organization to determine number of ordinary and scout antelopes
        if f[i] == f[i-1]:
            mu += 1
            P = 1 - np.exp(-(mu/(M*gamma))**2)
            if np.random.rand() < P and N_ordinary > N_ordinary_min:
                N_ordinary -= 1
        else:
            mu = 0
            if N_ordinary < N_ordinary_max:
                N_ordinary += 1
        # Conduct self-learning search by algorithm 5 lines 8-13
        if f[i] != f[i-1]:
            X_S = X[i] + (X[i] - X[i-1])  # self-learning search by equation (4) investigates further change in direction of most recent improvement
        else:
            X_S = X[i] + (X[i] - np.mean(antelopes, axis=0))  # self-learning search by equation (5), investigates reversing average tendency of unproductive search step
        X_S = np.clip(X_S, 0, 1) # not included in the paper but necessary to either clip or randomize perhaps
        f_s = objective_function(fish, prey_velocity, xd, yd, X_S)
        if f_s > f[i]:  # update the best solution if an improvement was found in this step
            X[i] = X_S
            f[i] = f_s
            print("In iteration", i, "with N_o=", N_ordinary, "and R=", R[i], "X_S learning improved fitness f[i] to", f[i])#, "and antelope\nX[i]=", X[i])
        fEvals += len(antelopes) + 1
    print("Evaluation count was ", fEvals)
    return maneuver_from_proportions(fish, prey_velocity, xd, yd, X[M-1])

def optimal_maneuver_SAMA(fish, **kwargs):
    '''The 'use_total_cost' parameter should be either False to optimize for the pure energetic cost of the maneuvering or True to incorporate
        the opportunity cost of not searching during the two pursuit segments, given an assumed NREI.'''
    fish.fEvals = 0
    d3D = kwargs.get('detection_point_3D', (-1 * fish.fork_length, 0.5*fish.fork_length, 0.5*fish.fork_length)) # default detection point given if not specified
    y3D, z3D = d3D[1:3]
    R = np.sqrt(y3D**2 + z3D**2)
    matrix_2Dfrom3D = np.array([[1,0,0],[0,y3D/R,z3D/R],[0,-z3D/R,y3D/R]]) # matrix to rotate the 3-D detection point about the x-axis into the x-y plane
    matrix_3Dfrom2D = matrix_2Dfrom3D.T                                    # because the inverse of this matrix is also its transpose
    (xd, yd) = matrix_2Dfrom3D.dot(np.array(d3D))[0:2]      # 2-D detection point to use for the model, not yet sign-adjusted
    ysign = np.sign(yd)
    yd *= ysign # Because of symmetry, we do maneuver calculations in the positive-y side of the x-y plane, saving the sign to convert back at the end
    fittest_maneuver = SAMASearch(fish, xd, yd, **kwargs)
    fittest_maneuver.matrix_3Dfrom2D = np.ascontiguousarray(matrix_3Dfrom2D) # Set attributes to allow the fittest solution to convert; the contiguous array typing prevents a silly warning about Numba execution speed in np.dot in maneuver.to_3D
    fittest_maneuver.ysign = ysign                     # results back into 3-D
    fittest_maneuver.calculate_summary_metrics() # calculate final summary quantities like average metabolic rate that are only needed for the optimal solution, not to evaluate fitness while finding it
    label = kwargs.get('label', "")
    iterations = kwargs.get("iterations")
    if not kwargs.get('suppress_output', False):
        if fittest_maneuver.energy_cost != CONVERGENCE_FAILURE_COST:
            print("Lowest energy cost after {0} CS iterations ({7:8d} evaluations) was {1:10.6f} joules. Mean speed {2:4.1f} cm/s, {3:5.2f} bodylengths/s. Metabolic rate {4:7.1f} mg O2/kg/hr ({5:4.1f}X SMR). {6}".format(iterations, fittest_maneuver.energy_cost, fittest_maneuver.mean_swimming_speed, fittest_maneuver.mean_swimming_speed_bodylengths, fittest_maneuver.mean_metabolic_rate, fittest_maneuver.mean_metabolic_rate_SMRs,label, fish.fEvals))
            if fittest_maneuver.dynamics.bad_thrust_b_penalty > 0:
                print("The best maneuver included a penalty for a bad thrust in stage b of the final straight, penalty factor {0:.3f}.".format(fittest_maneuver.dynamics.bad_thrust_b_penalty))
            if fittest_maneuver.dynamics.violates_acceleration_limit_penalty > 0:
                print("The best maneuver included a penalty for violating the acceleration limit, penalty factor {0:.3f}.".format(fittest_maneuver.dynamics.violates_acceleration_limit_penalty))
        else:
            print("Maneuver failed to converge in all possible paths/dynamics considered. Did not find an optimal maneuver.")
            if hasattr(fittest_maneuver, 'convergence_failure_code'):
                print("Convergence failure code in maneuver.py was {0}.".format(fittest_maneuver.convergence_failure_code))
                if fittest_maneuver.convergence_failure_code == 4:  # failure in final straight
                    print("Convergence failure code in finalstraight.py was {0}.".format(fittest_maneuver.dynamics.straight_3.convergence_failure_code))
    return fittest_maneuver
