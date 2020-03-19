from __future__ import division # prevents integer divisions from truncating result to an integer, the default python 2.x behavior
import numpy as np
from functools import partial
from .maneuver import *
import os

#threaded = False  # Tells whether or not to thread the calculations over multiple cores. Not working right now.

MICROGA_MUTATION_RATE=0.35    # Controls the probability that any given characteristic mutates when a solution is mutated
MICROGA_MUTATION_SCALE=0.15   # Controls the variance of the size of the mutations ~N(0,scale) added to value proportions when mutating
MICROGA_MIXING_RATIO=0.5      # Controls the proportion of the value of the child's characteristics that comes from the fittest parent
MICROGA_SHUFFLE_INTERVAL=15   # How often to shuffle the microgenetic algorithm to bring in new solution genes

@jit(nopython=True)
def choose_parent_indices(fitnesses):
    popsize = len(fitnesses)
    match_1 = np.empty(2, dtype=np.int8)
    match_1[0] = int(np.random.uniform(0,popsize))                                 # Get 2 random indices about 3x faster than np.random.choice()
    match_1[1] = int(np.random.uniform(0,popsize))                                 # Get 2 random indices about 3x faster than np.random.choice()
    while match_1[1] == match_1[0]: match_1[1] = int(np.random.uniform(0,popsize)) # Make sure no solution is mated with itself
    winner_1 = match_1[0] if fitnesses[match_1[0]] > fitnesses[match_1[1]] else match_1[1] # Choose the parent index with greatest fitness in the first match
    # Second match proceeds exactly like the first
    match_2 = np.empty(2, dtype=np.int8)
    match_2[0] = int(np.random.uniform(0,popsize))                                 # Get 2 random indices about 3x faster than np.random.choice()
    match_2[1] = int(np.random.uniform(0,popsize))                                 # Get 2 random indices about 3x faster than np.random.choice()
    while match_2[1] == match_2[0]: match_2[1] = int(np.random.uniform(0,popsize)) # Make sure no solution is mated with itself
    winner_2 = match_2[0] if fitnesses[match_2[0]] > fitnesses[match_2[1]] else match_2[1] # Choose the parent index with greatest fitness in the first match
    return np.array([winner_1, winner_2]) if fitnesses[winner_1] > fitnesses[winner_2] else np.array([winner_2, winner_1]) # Return winning parent indices in order of fitness

def refine_solution_population(input_population, random_solution_partial, mate_solutions_partial, popsize, iterations):
    population = sorted(input_population, key = lambda x: -x.fitness) # sort ascending by fitness to start                       
    iterations_since_shuffling = 0
    i = 0
    while i < iterations:
        iterations_since_shuffling += 1
        best_fitness = population[0].fitness
        fittest_solution_index = 0
        for j in range(1,popsize):
            if population[j].fitness > best_fitness:
                best_fitness = population[j].fitness
                fittest_solution_index = j
        fittest_solution = population[fittest_solution_index]
        fittest_solution_variant = mutate_solution(fittest_solution, MICROGA_MUTATION_RATE, MICROGA_MUTATION_SCALE)
        fitnesses = [solution.fitness for solution in population]
        if (iterations_since_shuffling > MICROGA_SHUFFLE_INTERVAL):
        #if (iterations_since_shuffling > 30) or (abs(variation(fitnesses)) < 0.0001):
        #    if (abs(variation(fitnesses)) < 0.01): print("Shuffling for variation {0:.5f} after {1} iterations.".format(abs(variation(fitnesses)),iterations_since_shuffling))
            iterations_since_shuffling = 0
            population[2:popsize] = [random_solution_partial() for k in range(popsize-2)]
        else:
            parent_indices = [choose_parent_indices(fitnesses) for x in range(popsize-2)]
            fittest_parents = [population[p[0]] for p in parent_indices] 
            other_parents = [population[p[1]] for p in parent_indices] 
            children = [mate_solutions_partial(fittest_parents[j],other_parents[j]) for j in range(len(fittest_parents))]
            population[2:popsize] = children    
        population[0] = fittest_solution
        population[1] = fittest_solution_variant
        i += 1
    return fittest_solution


def optimal_maneuver(fish, detection_point_3D = None, popsize = 4, iterations = 3000, suppress_output = False, label="", use_starting_iterations = False, num_starting_populations = 6, num_starting_iterations = 500, allow_retries = True):
    '''The 'use_total_cost' parameter should be either False to optimize for the pure energetic cost of the maneuvering or True to incorporate 
        the opportunity cost of not searching during the two pursuit segments, given an assumed NREI.'''
    fish.fEvals = 0
    d3D = detection_point_3D if detection_point_3D is not None else (-1 * fish.fork_length, 0.5*fish.fork_length, 0.5*fish.fork_length)         
    (xd, yd) = d3D[0:2]
    y3D, z3D = d3D[1:3]
    R = np.sqrt(y3D**2 + z3D**2)
    matrix_2Dfrom3D = np.array([[1,0,0],[0,y3D/R,z3D/R],[0,-z3D/R,y3D/R]]) # matrix to rotate the 3-D detection point about the x-axis into the x-y plane
    matrix_3Dfrom2D = matrix_2Dfrom3D.T                                    # because the inverse of this matrix is also its transpose
    (xd, yd) = matrix_2Dfrom3D.dot(np.array(d3D))[0:2]      # 2-D detection point to use for the model, not yet sign-adjusted
    ysign = np.sign(yd)
    yd *= ysign # Because of symmetry, we do maneuver calculations in the positive-y side of the x-y plane, saving the sign to convert back at the end
    random_solution_partial = partial(random_maneuver, fish = fish, xd = xd, yd = yd)
    mate_solutions_partial = partial(mate_solutions, mixing_ratio = MICROGA_MIXING_RATIO)
    if use_starting_iterations and num_starting_populations >= popsize:
        starting_populations = [[random_solution_partial() for i in range(popsize)] for j in range(num_starting_populations)]   
        refined_solutions = [refine_solution_population(population, random_solution_partial, mate_solutions_partial, popsize, num_starting_iterations) for population in starting_populations]
        solutions_for_final_refinement = sorted(refined_solutions, key = lambda sol: -sol.fitness)[0:popsize]
    elif use_starting_iterations and num_starting_populations < popsize:
        starting_populations = [[random_solution_partial() for i in range(popsize)] for j in range(num_starting_populations)]   
        refined_solutions = [refine_solution_population(population, random_solution_partial, mate_solutions_partial, popsize, num_starting_iterations) for population in starting_populations]
        random_solutions = [random_solution_partial() for i in range(popsize - num_starting_populations)]
        solutions_for_final_refinement = refined_solutions + random_solutions
    else:
        solutions_for_final_refinement = [random_solution_partial() for i in range(popsize)]
    fittest_solution = refine_solution_population(solutions_for_final_refinement, random_solution_partial, mate_solutions_partial, popsize, iterations)
    fittest_solution.matrix_3Dfrom2D = matrix_3Dfrom2D # Set attributes to allow the fittest solution to convert
    fittest_solution.ysign = ysign                     # results back into 3-D
    fittest_solution.calculate_summary_metrics() # calculate final summary quantities like average metabolic rate that are only needed for the optimal solution, not to evaluate fitness while finding it
    if not suppress_output:
        # 'r' for random, 'c' for crossover (mating), 'm' for mutant
        origin = {1: 'random', 2: 'crossover', 3: 'mutation'}[fittest_solution.origin]
        print("Lowest energy cost after {3}x{4}+{0} uGA iterations ({10:8d} evaluations) was {1:10.6f} joules (origin: {9:9s}). Mean speed {7:4.1f} cm/s, {6:5.2f} bodylengths/s. Metabolic rate {5:7.1f} mg O2/kg/hr ({8:4.1f}X SMR). {2}".format(iterations, -fittest_solution.fitness, label, (num_starting_populations if use_starting_iterations else 0), (num_starting_iterations if use_starting_iterations else 0),fittest_solution.mean_metabolic_rate,fittest_solution.mean_swimming_speed_bodylengths,fittest_solution.mean_swimming_speed,fittest_solution.mean_metabolic_rate_SMRs,origin,fish.fEvals))
    if allow_retries and fittest_solution.mean_metabolic_rate > fish.SMR * 125 and num_starting_populations < 50:
        os.system('say "Retrying a solution with an excessive metabolic rate using a larger population."')
        print("Lowest-energy solution exceeded 125X SMR. Assuming correct solution was not found and retrying.")  
        return optimal_maneuver(fish, detection_point_3D, popsize, 2*iterations, suppress_output, "(RETRY {0})".format(label), True, 2*num_starting_populations, 2*num_starting_iterations, allow_retries)
    else:
        return fittest_solution



