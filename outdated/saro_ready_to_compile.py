# ------------------------------------------------------------------------------------------------------%
#       Adapted from class 'OriginalSARO' in SARO.py in the Mealpy package                              %
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
#       Heavily modified for just-in-time compiled operation by Jason R Neuswanger on 11/25/2021        %
#       The purpose of this is modification to trim the version of the algorithm included in the large  %
#       optimization package into a bare-bones, faster version compiled and stripped of diagnostics     %
#       so it can be used more efficiently on a very large number of optimizations.                     %
#       This modification also hard-codes the bounds of the variables at [0, 1].                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np

class CompiledSARO:

    ## Assumption the A solution with format: [position, [target, [obj1, obj2, ...]]]
    ID_POS = 0  # Index of position/location of solution/agent
    ID_FIT = 1  # Index of fitness value of solution/agent

    def __init__(self, obj_func, epoch=10000, pop_size=100, se=0.5, mu=50, problem_n_dims=12, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 50
        """
        self.epoch, self.pop_size, self.solution, self.pop = None, None, None, None

        self.objective_function = obj_func
        self.problem_n_dims = problem_n_dims
        self.problem_lb = [0, ] * problem_n_dims
        self.problem_ub = [1, ] * problem_n_dims

        self.nfe_per_epoch = 2 * pop_size
        self.epoch = epoch
        self.pop_size = pop_size
        self.se = se
        self.mu = mu

        ## Dynamic variable
        self.dyn_USN = np.zeros(self.pop_size)

    def evolve(self):
        """ Args: epoch (int): The current iteration """
        pop_x = self.pop[:self.pop_size].copy()
        pop_m = self.pop[self.pop_size:].copy()

        pop_new = []
        for i in range(self.pop_size):
            ## Social Phase
            k = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {i})) # randomly select a solution from main pop or memory
            sd = pop_x[i][self.ID_POS] - self.pop[k][self.ID_POS] # use it to establish the search direction and distance
            j_rand = np.random.randint(0, self.problem_n_dims) # at least one dimension of new solution to move in the search direction
            r1 = np.random.uniform(-1, 1) # distance to move in the search direction
            pos_new = pop_x[i][self.ID_POS].copy()
            for j in range(0, self.problem_n_dims):
                if np.random.uniform() < self.se or j == j_rand:
                    if self.pop[k][self.ID_FIT] > pop_x[i][self.ID_FIT]:
                        pos_new[j] = self.pop[k][self.ID_POS][j] + r1 * sd[j]
                    else:
                        pos_new[j] = pop_x[i][self.ID_POS][j] + r1 * sd[j]
                if pos_new[j] < 0:
                    pos_new[j] = (pop_x[i][self.ID_POS][j]) / 2
                if pos_new[j] > 1:
                    pos_new[j] = (pop_x[i][self.ID_POS][j] + 1) / 2
            # pos_new = np.clip(pos_new, self.problem_lb, self.problem_ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        for i in range(self.pop_size):
            if pop_new[i][self.ID_FIT] > pop_x[i][self.ID_FIT]:
                # print("Found a better solution in social phase, fitness {0} > {1}.".format(pop_new[i][self.ID_FIT],pop_x[i][self.ID_FIT]))
                pop_m[np.random.randint(0, self.pop_size)] = pop_x[i].copy()
                pop_x[i] = pop_new[i].copy()
                self.dyn_USN[i] = 0
            else:
                self.dyn_USN[i] += 1

        ## Individual phase
        pop = pop_x + pop_m # pop_x.copy() + pop_m.copy()
        pop_new = []
        for i in range(0, self.pop_size):
            k, m = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {i}), 2, replace=False)
            pos_new = pop_x[i][self.ID_POS] + np.random.uniform() * (pop[k][self.ID_POS] - pop[m][self.ID_POS])
            for j in range(0, self.problem_n_dims):
                if pos_new[j] < 0:
                    pos_new[j] = (pop_x[i][self.ID_POS][j]) / 2
                if pos_new[j] > 1:
                    pos_new[j] = (pop_x[i][self.ID_POS][j] + 1) / 2
            # pos_new = np.clip(pos_new, self.problem_lb, self.problem_ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        for i in range(self.pop_size):
            if pop_new[i][self.ID_FIT] > pop_x[i][self.ID_FIT]:
                # print("Found a better solution in individual phase, fitness {0} > {1}.".format(pop_new[i][self.ID_FIT],pop_x[i][self.ID_FIT]))
                pop_m[np.random.randint(0, self.pop_size)] = pop_x[i]
                pop_x[i] = pop_new[i].copy()
                self.dyn_USN[i] = 0
            else:
                self.dyn_USN[i] += 1
            if self.dyn_USN[i] > self.mu:
                # print("replacing stagnant solution", i)
                pop_x[i] = self.create_solution()
                self.dyn_USN[i] = 0
        self.pop = pop_x + pop_m

    def solve(self):
        pop = [self.create_solution() for _ in range(0, 2 * self.pop_size)]
        self.pop = sorted(pop, key=lambda agent: -agent[self.ID_FIT])  # Already returned a new sorted list

        for epoch in range(0, self.epoch):
            self.evolve()
            self.pop = sorted(self.pop, key=lambda agent: -agent[self.ID_FIT])
            # print("Best fitness at epoch ", epoch," is ", self.pop[-1][self.ID_FIT])
        return self.pop[-1]  # returns [position, fitness value]

    def create_solution(self):
        position = np.random.uniform(self.problem_lb, self.problem_ub)
        fitness = self.objective_function(position)
        return [position, fitness]

    def update_fitness_population(self, pop):
        for i, agent in enumerate(pop):
            pop[i][self.ID_FIT] = self.objective_function(agent[self.ID_POS])
        return pop
