# ------------------------------------------------------------------------------------------------------%
#       Heavily adapted from class 'OriginalSARO' in SARO.py in the Mealpy package                      %
#                                                                                                       %
#       Mealpy version created by "Thieu Nguyen" at 11:16, 18/03/2020                                   %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#                                                                                                       %
#       Based on the paper: Search And Rescue Optimization (SAR) (A New Optimization Algorithm Based    %
#                           on Search and Rescue Operations)                                            %
#       Link:               https://doi.org/10.1155/2019/2482543                                        %
#                                                                                                       %
#       HEAVILY modified for just-in-time compiled operation by Jason Neuswanger on 11/27/2021          %
#       The purpose of this is modification to trim the version of the algorithm included in the large  %
#       optimization package into a bare-bones, faster version compiled and stripped of diagnostics     %
#       so it can be used more efficiently on a very large number of optimizations.                     %
#       This modification also hard-codes the bounds of the variables at [0, 1].                        %
#       It has also my objective function baked in (reliant on another jitted class).                   %
#       And it feeds back into a "solution" object thrust values that can change when calculating its   %
#       "fitness" in order to ensure convergence (see needs_to_slow_down). It also optionally tracks    %
#       many quantities relevant to evaluating convergence performance.                                 %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from .maneuveringfish import ManeuveringFish
from .maneuver import maneuver_from_proportions
from .constants import CONVERGENCE_FAILURE_COST
from numba import jit, float64, uint8, uint64, boolean, optional, types
from numba.experimental import jitclass

@jitclass([('position', float64[:]), ('fitness', float64), ('energy_cost', float64), ('pursuit_duration', float64), ('capture_x', float64), ('was_slowed_down', boolean), ('origin', uint8)])
class SolutionSARO:

    def __init__(self, position, fitness, energy_cost, pursuit_duration, capture_x, was_slowed_down, origin):
        self.position = position
        self.fitness = fitness
        self.energy_cost = energy_cost
        self.pursuit_duration = pursuit_duration
        self.capture_x = capture_x
        self.was_slowed_down = was_slowed_down
        self.origin = origin # 0 for random, 1 for social phase, 2 for individual phase

fish_type = ManeuveringFish.class_type.instance_type
solution_type = SolutionSARO.class_type.instance_type

@jit(solution_type(solution_type), nopython=True)
def copy_solution(s):
    return SolutionSARO(s.position, s.fitness, s.energy_cost, s.pursuit_duration, s.capture_x, s.was_slowed_down, s.origin)

saro_spec = [
    ('fish', fish_type),
    ('prey_velocity', float64),
    ('det_x', float64),                        # detection position x coord
    ('det_y', float64),                        # detection position y coord
    ('epoch', uint64),
    ('nfe', uint64),
    ('nfe_nonconvergent', uint64),                     # tracks the number of objective function evaluations (solutions tried) that weren't convergent
    ('nfe_slowed_down', uint64),                     # tracks the number of objective function evaluations (solutions tried) that had to be slowed down
    ('pop_size', uint64),
    ('dims', uint8),
    ('problem_lb', uint8[:]),
    ('problem_ub', uint8[:]),
    ('mu', uint64),
    ('se', float64),
    ('dyn_USN', uint64[:]),
    ('solution', optional(solution_type)),
    ('tracked', boolean),
    ('tracked_nfe', optional(uint64[:])),
    ('tracked_nfe_nonconvergent', optional(uint64[:])),
    ('tracked_nfe_slowed_down', optional(uint64[:])),
    ('tracked_convergence_failure_codes', optional(types.string)),
    ('tracked_energy_cost', optional(float64[:])),
    ('tracked_pursuit_duration', optional(float64[:])),
    ('tracked_capture_x', optional(float64[:])),
    ('tracked_position', optional(float64[:,:])),
    ('tracked_best_was_slowed_down', optional(boolean[:])),
    ('tracked_best_origin', optional(uint8[:])),
    ('tracked_number_slowed_down', optional(uint64[:])),
    ('label', optional(types.string)) # allows external code to label this model run for diagnostic testing
]

@jitclass(saro_spec)
class CompiledSARO:

    def __init__(self, fish, prey_velocity, det_x, det_y, epoch=1000, pop_size=100, se=0.5, mu=50, dims=12, tracked=False):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 50
        """
        self.fish = fish
        self.prey_velocity = prey_velocity
        self.det_x = det_x
        self.det_y = det_y

        self.dims = dims
        self.problem_lb = np.zeros(dims, dtype=uint8)
        self.problem_ub = np.ones(dims, dtype=uint8)

        self.epoch = epoch
        self.pop_size = pop_size
        self.nfe = 0
        self.nfe_nonconvergent = 0
        self.nfe_slowed_down = 0
        self.se = se
        self.mu = mu

        self.dyn_USN = np.zeros(self.pop_size, dtype=uint64)
        self.tracked = tracked
        if tracked:
            self.tracked_nfe = np.zeros(self.epoch+1, dtype=uint64)
            self.tracked_nfe_nonconvergent = np.zeros(self.epoch+1, dtype=uint64)
            self.tracked_nfe_slowed_down = np.zeros(self.epoch+1, dtype=uint64)
            self.tracked_convergence_failure_codes = ''
            self.tracked_energy_cost = np.zeros(self.epoch+1, dtype=float64)
            self.tracked_pursuit_duration = np.zeros(self.epoch+1, dtype=float64)
            self.tracked_capture_x = np.zeros(self.epoch+1, dtype=float64)
            self.tracked_position = np.zeros((self.epoch+1, self.dims), dtype=float64)
            self.tracked_best_was_slowed_down = np.zeros(self.epoch+1, dtype=boolean)
            self.tracked_best_origin = np.zeros(self.epoch+1, dtype=uint8)
            self.tracked_number_slowed_down = np.zeros(self.epoch+1, dtype=uint64)

    def solution_with_evaluated_fitness(self, p, origin):
        self.nfe += 1
        maneuver = maneuver_from_proportions(self.fish, self.prey_velocity, self.det_x, self.det_y, p)
        new_p = maneuver.proportions()[:self.dims] # this saves the changes made to thrusts within the maneuver to arrive from downstream of the focal point at the end
        if maneuver.dynamics.energy_cost == CONVERGENCE_FAILURE_COST:
            self.nfe_nonconvergent += 1
            if self.tracked:
                self.tracked_convergence_failure_codes += '_' + maneuver.convergence_failure_code
        if maneuver.dynamics.was_slowed_down:
            self.nfe_slowed_down += 1
        return SolutionSARO(new_p, maneuver.fitness, maneuver.energy_cost, maneuver.pursuit_duration, maneuver.capture_x, maneuver.dynamics.was_slowed_down, origin)

    def evolve(self, master_pop):
        pop_x = master_pop[:self.pop_size].copy()
        pop_m = master_pop[self.pop_size:].copy()
        pop_new = []
        for i in range(self.pop_size):
            ## Social Phase
            k = i
            while k == i: # choose a random k not equal to i; just repeat the choice if k == i is chosen
                k = np.random.randint(0, 2 * self.pop_size)
            k = round(k)  # inexplicable numba quirk requires this line here (and not in the line above) to make k an int instead of float
            sd = pop_x[i].position - master_pop[k].position # use it to establish the search direction and distance
            j_rand = np.random.randint(0, self.dims) # at least one dimension of new solution to move in the search direction
            r1 = np.random.uniform(-1, 1) # distance to move in the search direction
            pos_new = pop_x[i].position.copy()
            for j in range(0, self.dims):
                if np.random.uniform(0, 1) < self.se or j == j_rand:
                    if master_pop[k].fitness > pop_x[i].fitness:
                        pos_new[j] = master_pop[k].position[j] + r1 * sd[j]
                    else:
                        pos_new[j] = pop_x[i].position[j] + r1 * sd[j]
                if pos_new[j] < 0:
                    pos_new[j] = (pop_x[i].position[j]) / 2
                if pos_new[j] > 1:
                    pos_new[j] = (pop_x[i].position[j] + 1) / 2
            pop_new.append(self.solution_with_evaluated_fitness(pos_new, 1))
        for i in range(self.pop_size):
            if pop_new[i].fitness > pop_x[i].fitness:
                pop_m[np.random.randint(0, self.pop_size)] = copy_solution(pop_x[i]) # SolutionSARO(pop_x[i].position, pop_x[i].fitness)
                pop_x[i] = copy_solution(pop_new[i]) # SolutionSARO(pop_new[i].position, pop_new[i].fitness)
                self.dyn_USN[i] = 0
            else:
                self.dyn_USN[i] += 1
        ## Individual phase
        pop = pop_x + pop_m
        pop_new = []
        for i in range(0, self.pop_size):
            choices = np.array([x for x in range(0, 2 * self.pop_size) if x != i], dtype=uint64)
            k, m = np.random.choice(choices, 2, replace=False)
            pos_new = pop_x[i].position + np.random.uniform(0, 1) * (pop[k].position - pop[m].position)
            for j in range(0, self.dims):
                if pos_new[j] < 0:
                    pos_new[j] = (pop_x[i].position[j]) / 2
                if pos_new[j] > 1:
                    pos_new[j] = (pop_x[i].position[j] + 1) / 2
            pop_new.append(self.solution_with_evaluated_fitness(pos_new, 2))
        for i in range(self.pop_size):
            if pop_new[i].fitness > pop_x[i].fitness:
                pop_m[np.random.randint(0, self.pop_size)] = pop_x[i]
                pop_x[i] = copy_solution(pop_new[i]) # SolutionSARO(pop_new[i].position, pop_new[i].fitness)
                self.dyn_USN[i] = 0
            else:
                self.dyn_USN[i] += 1
            if self.dyn_USN[i] > self.mu:
                pop_x[i] = self.create_random_solution()
                self.dyn_USN[i] = 0
        new_master_pop = pop_x + pop_m # because no addition is defined for typed.List
        return sorted(new_master_pop, key=lambda agent: -agent.fitness)

    def solve(self, verbose=False):
        pop = [self.create_random_solution() for _ in range(0, 2 * self.pop_size)]
        master_pop = sorted(pop, key=lambda agent: -agent.fitness)
        if self.tracked:
            self.tracked_nfe[0] = self.nfe
            self.tracked_nfe_nonconvergent[0] = self.nfe_nonconvergent
            self.tracked_nfe_slowed_down[0] = self.nfe_slowed_down
            self.tracked_energy_cost[0] = master_pop[0].energy_cost
            self.tracked_pursuit_duration[0] = master_pop[0].pursuit_duration
            self.tracked_capture_x[0] = master_pop[0].capture_x
            self.tracked_position[0] = master_pop[0].position
        for epoch in range(0, self.epoch):
            master_pop = self.evolve(master_pop)
            if self.tracked:
                self.tracked_nfe[epoch+1] = self.nfe
                self.tracked_nfe_nonconvergent[epoch+1] = self.nfe_nonconvergent
                self.tracked_nfe_slowed_down[epoch+1] = self.nfe_slowed_down
                self.tracked_energy_cost[epoch+1] = master_pop[0].energy_cost
                self.tracked_pursuit_duration[epoch+1] = master_pop[0].pursuit_duration
                self.tracked_capture_x[epoch+1] = master_pop[0].capture_x
                self.tracked_position[epoch+1] = master_pop[0].position
                self.tracked_best_was_slowed_down[epoch+1] = master_pop[0].was_slowed_down
                self.tracked_best_origin[epoch+1] = master_pop[0].origin
                self.tracked_number_slowed_down[epoch + 1] = len([sol for sol in master_pop if sol.was_slowed_down])
        self.solution = master_pop[0]
        if verbose:
            print("Lowest energy cost was", self.solution.energy_cost, "J after", self.nfe,"evals  (", self.epoch,"x", self.pop_size,"; se = ", self.se,"; mu = ", self.mu, ").")
        return self.solution

    def create_random_solution(self):
        position = np.random.uniform(0, 1, self.dims)
        return self.solution_with_evaluated_fitness(position, 0)

























