# Cuckoo Search Optimizer
# Customized from the implementation in EvoloPy at https://github.com/7ossam81/EvoloPy/blob/master/CS.py
# Operates on numpy solution vectors scaled from 0 to 1

import math
import time
import numpy as np
from maneuvermodel.maneuver import maneuver_from_proportions
from maneuvermodel.constants import CONVERGENCE_FAILURE_COST

def objective_function(fish, prey_velocity, xd, yd, p):
    maneuver = maneuver_from_proportions(fish, prey_velocity, xd, yd, p)
    return -maneuver.fitness

def global_random_walk(nests, best, n):
    """ This function returns an entire set of nests consisting of Levy flight perturbations of the old nests. """
    temp_nests = np.array(nests) # np.empty(nests.shape)
    beta = 3/2
    sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    for j in range(n):
        s = nests[j, :]
        dim = len(s)
        u = np.random.randn(dim)*sigma
        v = np.random.randn(dim)
        step = u/abs(v)**(1/beta)
        stepsize = 0.01*(step*(s-best)) # step size is based on distance between this solution and the best one
        s += stepsize*np.random.randn(dim)
        np.clip(s, 0, 1, out=s)
        temp_nests[j,:] = s
    return temp_nests

def pairwise_best_nests(old_nests, new_nests, old_fitnesses, n, fish, prey_velocity, xd, yd):
    """ This function takes the main current nests (nests) with their fitness (fitness) and a proposed set of new nests,
        checks each new nest to see if its fitness is better than that of the corresponding old nest, and if so, replaces
        it. In other words, it compares each nest in new_nests against the corresponding element in old_nests and returns
        a set of nests with the best one out of each pairwise comparison."""
    temp_nests = np.copy(old_nests)
    for j in range(n):
        np.clip(new_nests[j, :], 0, 1, out=new_nests[j, :])
        fnew = objective_function(fish, prey_velocity, xd, yd, new_nests[j, :])
        if fnew <= old_fitnesses[j]:
           old_fitnesses[j] = fnew
           temp_nests[j,:] = new_nests[j, :]
    # Find the current best
    fmin = min(old_fitnesses)
    K = np.argmin(old_fitnesses)
    bestlocal = temp_nests[K,:]
    return fmin, bestlocal, temp_nests, old_fitnesses

def local_random_walk(nests, pa, n, dim):
    """ This function does not seem to correspond to the way the original algorithm emptied nests at all. It doesn't really get rid of entire nests and replace
        them; instead, it computes perturbations to them based on differences with other nests (rather than Levy flights) and applies those perturbations. This
        seems to fit the description of the local random walk step by Iztok et al 2013, which is describing the main Cuckoo algorithm, but it is very difficult
        to see this from the original Cuckoo paper."""
    # Create an array the same dimensions as the nests with (pa) elements being 0 and the rest being 1
    K = np.random.uniform(0, 1, (n, dim)) > pa
    # Get n random differences between nests, and scale them down by a random factor between 0 and 1
    stepsize = np.random.rand() * (nests[np.random.permutation(n), :] - nests[np.random.permutation(n), :])
    # then throw out (pa) proportion of those and add the remainder to the current nests
    return nests + stepsize * K

##########################################################################

def CuckooSearch(fish, xd, yd, **kwargs):
    prey_velocity = kwargs.get('prey_velocity', fish.focal_velocity) # Prey velocity defaults to fish focal velocity if prey velocity not specified
    n = kwargs.get('n', 50)                                 # Number of nests
    iterations = kwargs.get('iterations', 1000)             # Number of iterations
    pa = kwargs.get('p_a', 0.25)                            # Proportion of bad solutions rejected and replaced with random new ones each iteration ("alien eggs discovered")
    dim = 12

    nests = np.random.rand(n, dim)   # Initialize nests randomly
    new_nests = np.copy(nests)       # Create new_nests initially as a copy of the other ones
    fitnesses = np.full(n, np.inf)     # Initialize fitness vector with bad fitnesses

    fmin, best_nest, nests, fitnesses = pairwise_best_nests(nests, new_nests, fitnesses, n, fish, prey_velocity, xd, yd)        # Calculating initial fitnesses, using comparison function for convenience

    for _ in range(iterations):                                                                                                 # Begin main search loop
        new_nests = global_random_walk(nests, best_nest, n)                                                                     # Generate new solutions (but keep the current best)
        fnew, best, nests, fitnesses = pairwise_best_nests(nests, new_nests, fitnesses, n, fish, prey_velocity, xd, yd)         # Compare solutions pairwise vs levy perturbations and keep best of each pair
        new_nests = local_random_walk(new_nests, pa, n, dim)                                                                    # Create new solutions based on differences between current ones
        fnew, best, nests, fitnesses = pairwise_best_nests(nests, new_nests, fitnesses, n, fish, prey_velocity, xd, yd)         # Evaluate those pairing-based new solutions and find the best
        if fnew < fmin:
            fmin = fnew
            best_nest = best

    result = maneuver_from_proportions(fish, prey_velocity, xd, yd, best_nest)
    result.objective_function_evaluations = n * (2 * iterations + 1)
    return result

def optimal_maneuver_CS(fish, detection_point_3D, **kwargs):
    start_time = time.perf_counter()
    y3D, z3D = detection_point_3D[1:3]
    R = np.sqrt(y3D**2 + z3D**2)
    matrix_2Dfrom3D = np.array([[1,0,0],[0,y3D/R,z3D/R],[0,-z3D/R,y3D/R]]) # matrix to rotate the 3-D detection point about the x-axis into the x-y plane
    matrix_3Dfrom2D = matrix_2Dfrom3D.T                                    # because the inverse of this matrix is also its transpose
    (xd, yd) = matrix_2Dfrom3D.dot(np.array(detection_point_3D))[0:2]      # 2-D detection point to use for the model, not yet sign-adjusted
    ysign = np.sign(yd)
    yd *= ysign # Because of symmetry, we do maneuver calculations in the positive-y side of the x-y plane, saving the sign to convert back at the end
    fittest_maneuver = CuckooSearch(fish, xd, yd, **kwargs)
    fittest_maneuver.matrix_3Dfrom2D = np.ascontiguousarray(matrix_3Dfrom2D) # Set attributes to allow the fittest solution to convert; the contiguous array typing prevents a silly warning about Numba execution speed in np.dot in maneuver.to_3D
    fittest_maneuver.ysign = ysign                     # results back into 3-D
    end_time = time.perf_counter()
    time_cost_s = end_time - start_time
    fittest_maneuver.calculate_summary_metrics() # calculate final summary quantities like average metabolic rate that are only needed for the optimal solution, not to evaluate fitness while finding it
    label = kwargs.get('label', "")
    iterations = kwargs.get("iterations", 1000)
    if not kwargs.get('suppress_output', False):
        if fittest_maneuver.energy_cost != CONVERGENCE_FAILURE_COST:
            print("Lowest energy cost after {0} CS iterations ({7:8d} evaluations, {8:5.1f} s) was {1:10.6f} joules. Mean speed {2:4.1f} cm/s, {3:5.2f} bodylengths/s. Metabolic rate {4:7.1f} mg O2/kg/hr ({5:4.1f}X SMR). {6}".format(iterations, fittest_maneuver.energy_cost, fittest_maneuver.mean_swimming_speed, fittest_maneuver.mean_swimming_speed_bodylengths, fittest_maneuver.mean_metabolic_rate, fittest_maneuver.mean_metabolic_rate_SMRs,label, fittest_maneuver.objective_function_evaluations, time_cost_s))
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
