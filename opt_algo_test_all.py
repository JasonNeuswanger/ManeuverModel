# Test different optimization algorithms

import numpy as np

from maneuvermodel import maneuveringfish
from maneuvermodel import maneuver

typical = {'Chinook Salmon': {'fork_length': 4.6, 'focal_velocity': 10, 'prey_velocity': 11, 'mass': 0.85, 'temperature': 9, 'max_thrust': 62, 'NREI': 0.017, 'detection_distance': 8, 'SMR': 226},
           'Dolly Varden': {'fork_length': 18, 'focal_velocity': 28, 'prey_velocity': 29, 'mass': 51, 'temperature': 10, 'max_thrust': 94, 'NREI': 0.017, 'detection_distance': 17, 'SMR': 52},
           'Arctic Grayling': {'fork_length': 43, 'focal_velocity': 42, 'prey_velocity': 48, 'mass': 920, 'temperature': 6, 'max_thrust': 159, 'NREI': 0.017, 'detection_distance': 35, 'SMR': 40}}
species = 'Dolly Varden'
fish = maneuveringfish.ManeuveringFish(fork_length = typical[species]['fork_length'],
                                       focal_velocity = typical[species]['focal_velocity'],
                                       mass = typical[species]['mass'],
                                       temperature = typical[species]['temperature'],
                                       SMR = typical[species]['SMR'],
                                       max_thrust = typical[species]['max_thrust'],
                                       NREI = typical[species]['NREI'],
                                       use_total_cost = False,
                                       disable_wait_time = False)
xd, yd = (-typical[species]['detection_distance'] / 1.414, typical[species]['detection_distance'] / 1.414)
prey_velocity = typical[species]['prey_velocity']

def objective_function(p):
    return maneuver.maneuver_from_proportions(fish, prey_velocity, xd, yd, p).fitness

def problem_description(verbose=True):
    return {
        "obj_func": objective_function,
        "lb": [0, ] * 12,
        "ub": [1, ] * 12,
        "minmax": "max",
        "verbose": verbose,
    }

def process_completion(model, plot=False):
    lowest_energy_cost = -model.solution[1][0]
    n_function_evals = model.nfe_per_epoch * model.epoch
    print("Lowest energy cost was {0} J after {1} fEvals.".format(lowest_energy_cost, n_function_evals))
    if plot:
        model.history.save_global_objectives_chart(filename="hello/goc")
        model.history.save_local_objectives_chart(filename="hello/loc")
        model.history.save_global_best_fitness_chart(filename="hello/gbfc")
        model.history.save_local_best_fitness_chart(filename="hello/lbfc")
        model.history.save_runtime_chart(filename="hello/rtc")
        model.history.save_exploration_exploitation_chart(filename="hello/eec")
        model.history.save_diversity_chart(filename="hello/dc")
        model.history.save_trajectory_chart(list_agent_idx=[3, 5], list_dimensions=[3], filename="hello/tc")

# BENCHMARK: Previous best with 15 million function evaluations was 0.146549 joules
# Best current value possible is 0.14628342129912647 with 4.5 million evaluations from Self Adaptive Differential Evolution
# SADE at one point found -0.146466 with only 100k evals, beating my old Cuckoo algo's 15 million
# SARO found 0.14628342250169188 at only 200k evals -- matches the best out to 8 digits!

# -------------------------------------------------------------------------------------------------------------------------
# BIO_BASED: Algorithms from mealpy.bio_based (note most animal algorithms are in swarm_based)
# -------------------------------------------------------------------------------------------------------------------------

from mealpy.bio_based import SMA
model = SMA.BaseSMA(problem_description(True), epoch=1000, pop_size=100, pr=0.03)
model.solve()
process_completion(model) # 0.149, 0.153, 0.152 @ 100k

# from mealpy.bio_based import EOA # Earthworm optimization algorithm
# model = EOA.BaseEOA(problem_description(True), epoch=350, pop_size=100, p_c=0.9, p_m=0.01, n_best=2, alpha=0.98, beta=1, gamma=0.9)
# model.solve()
# process_completion(model) #  0.19, 0.24 @ 100k, 0.22 @ 285k

# from mealpy.bio_based import IWO # Invasive Weed Optimization
# model = IWO.OriginalIWO(problem_description(True), epoch=1000, pop_size=100, seeds=[2, 10], exponent=2, sigma=[0.5, 0.001])
# model.solve()
# process_completion(model) #  0.84 @ 100k

from mealpy.bio_based import SBO # Satin Bowerbird Optimizer
model = SBO.BaseSBO(problem_description(True), epoch=1000, pop_size=100, alpha=0.94, pm=0.05, psw=0.02)
model.solve()
process_completion(model) #  0.16, 0.17, 0.19 @ 100k

# from mealpy.bio_based import VCS # Virus Colony Search
# model = VCS.BaseVCS(problem_description(True), epoch=1000, pop_size=33, lamda=0.5, xichma=0.3)
# model.solve()
# process_completion(model) # 0.25 @ 100k, 0.21 @ 300k

# from mealpy.bio_based import WHO # Wildebeest Herd Optimization
# model = WHO.BaseWHO(problem_description(True), epoch=1000, pop_size=100, n_s=3, n_e=3, eta=0.15, local_move=(0.9, 0.3), global_move=(0.2, 0.8), p_hi=0.9, delta=(2.0, 2.0))
# model.solve()
# process_completion(model) #  0.38 @ 553k

# from mealpy.bio_based import BBO # Biogeography-based optimization
# model = BBO.OriginalBBO(problem_description(True), epoch=1000, pop_size=100, p_m=0.01, elites=2)
# model.solve()
# process_completion(model) # 0.74, 0.71 @ 100k

# -------------------------------------------------------------------------------------------------------------------------
# EVOLUTIONARY_BASED: Algorithms from mealpy.evolutionary_based (note most animal algorithms are in swarm_based)
# -------------------------------------------------------------------------------------------------------------------------

from mealpy.evolutionary_based import CRO # Coral Reef Optimization
model = CRO.BaseCRO(problem_description(True), epoch=1800, pop_size=100, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.1, G=[0.02, 0.2], GCR=0.1, n_trials=3)
model.solve()
process_completion(model) # 0.159 @ 55k,  0.157, 0.168 @ 100k


from mealpy.evolutionary_based import CRO # Coral Reef Optimization with Opposition-based learning
model = CRO.OCRO(problem_description(True), epoch=1800, pop_size=100, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.1, G=(0.02, 0.2), GCR=0.1, n_trials=3, restart_count=55)
model.solve()
process_completion(model) # 0.168, 0.162, 0.165 @ 100k

from mealpy.evolutionary_based import DE # Differential Evolution
model = DE.BaseDE(problem_description(True), epoch=1000, pop_size=100)
model.solve()
process_completion(model) # 0.160, 0.160, 0.167 @ 100k

# from mealpy.evolutionary_based import DE # Adaptive Differential Evolution with Optional External Archive
# model = DE.JADE(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.165, 0.167 @ 100k

# NOTE: This one has best performance, but pretty slow per iteration!
from mealpy.evolutionary_based import DE # Self-adaptive differential evolution algorithm for numerical optimization
model = DE.SADE(problem_description(True), epoch=1000, pop_size=100)
model.solve(mode='sequential') # sequential is the default, switching mode to 'thread' doesn't make things faster
process_completion(model) # 0.148, 0.148, 0.147, 0.147, 0.147, 0.147, 0.148, 0.147 @ 100k, 0.14628342129912647 @ 4.5 mil (15000x300)

# from mealpy.evolutionary_based import DE # Success-History Based Parameter Adaptation for Differential Evolution
# model = DE.SHADE(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # Gets to around epoch 530 three times with fitness near 0.16 and then ValueError: In solution.py, value_from_proportion got min_value > max_value.

# from mealpy.evolutionary_based import DE # Improving the Search Performance of SHADE Using Linear Population Size Reduction
# model = DE.L_SHADE(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.166, 0.163, 0.160 @ 100k

# from mealpy.evolutionary_based import DE #  Exploring dynamic self-adaptive populations in differential evolution
# model = DE.SAP_DE(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.41 @ 100k

# from mealpy.evolutionary_based import EP #  Evolutionary Programming
# model = EP.BaseEP(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.23, 0.20 @ 100k

# from mealpy.evolutionary_based import EP #  Evolutionary Programming, Levy flight version
# model = EP.LevyEP(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.23, 0.22 @ 100k

# from mealpy.evolutionary_based import ES #  Evolutionary Strategies
# model = ES.BaseES(problem_description(True), epoch=1000, pop_size=130)
# model.solve()
# process_completion(model) # 0.99, 0.95 @ 100k

# from mealpy.evolutionary_based import FPA #  Flower Pollination Algorithm
# model = FPA.BaseFPA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.28, 0.25 @ 100k

# from mealpy.evolutionary_based import GA # Genetic algorithm
# model = GA.BaseGA(problem_description(True), epoch=1000, pop_size=100, pc=0.85, pm=0.05)
# model.solve()
# process_completion(model) # 0.31, 0.32 @ 100k

# from mealpy.evolutionary_based import MA # Memetic algorithm
# model = MA.BaseMA(problem_description(True), epoch=100, pop_size=100)
# model.solve()
# process_completion(model) # 0.31 @ 100k, also way slow, probalby way more evals than reported

# -------------------------------------------------------------------------------------------------------------------------
# HUMAN_BASED: Algorithms from mealpy.human_based
# -------------------------------------------------------------------------------------------------------------------------

# from mealpy.human_based import BRO # Battle Royale Optimization
# model = BRO.BaseBRO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.24 @ 145k

# from mealpy.human_based import BRO # Battle Royale Optimization
# model = BRO.OriginalBRO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.23 @ 100k

# from mealpy.human_based import BSO # Brain Storm Optimization
# model = BSO.BaseBSO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.22 @ 100k

# from mealpy.human_based import BSO # Brain Storm Optimization
# model = BSO.ImprovedBSO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.28 @ 100k

# from mealpy.human_based import CA # Culture Algorithm
# model = CA.OriginalCA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.83 @ 100k

# from mealpy.human_based import CHIO # Coronavirus Herd Immunity Optimization
# model = CHIO.BaseCHIO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # solution_from_proportion errors

# from mealpy.human_based import CHIO # Coronavirus Herd Immunity Optimization
# model = CHIO.OriginalCHIO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # solution_from_proportion errors

from mealpy.human_based import FBIO # Forensic Based Investigation Optimization
model = FBIO.BaseFBIO(problem_description(True), epoch=1000, pop_size=25)
model.solve()
process_completion(model) # 0.148, 0.152, 0.151, 0.148 @ 100k (250x100), 0.157, 0.157, 0.148, 0.151  @ 100k (1000x25), 0.1467 @ 400k

from mealpy.human_based import FBIO # Forensic Based Investigation Optimization
model = FBIO.OriginalFBIO(problem_description(True), epoch=250, pop_size=100)
model.solve()
process_completion(model) # 0.152, 0.149  @ 100k (250x100), 0.1470 @ 400k

# from mealpy.human_based import GSKA # Gaining Sharing Knowledge-based Algorithm
# model = GSKA.BaseGSKA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.185 @ 100k

# from mealpy.human_based import GSKA # Gaining Sharing Knowledge-based Algorithm
# model = GSKA.OriginalGSKA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.182 @ 100k

# from mealpy.human_based import ICA # Imperialist Competition Algorithm
# model = ICA.BaseICA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.164, 0.18 @ 100k

# from mealpy.human_based import LCO # Life Choice-based Optimization
# model = LCO.BaseLCO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.161, 0.17 @ 100k

# from mealpy.human_based import LCO # Life Choice-based Optimization
# model = LCO.OriginalLCO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # @ 0.20 100k

# from mealpy.human_based import LCO # Life Choice-based Optimization
# model = LCO.ImprovedLCO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # @ value_from_proportion errors

from mealpy.human_based import QSA
model = QSA.BaseQSA(problem_description(True), epoch=1000, pop_size=33)
model.solve()
process_completion(model) # 0.1467, 0.155, 0.157, 0.148 @ 100k (333x100), 0.14633, 0.14725, 0.14646 @ 100k (1000x33), 0.145289 @ 300k

from mealpy.human_based import QSA # Queuing Search Algorithm
model = QSA.OppoQSA(problem_description(True), epoch=1000, pop_size=25)
model.solve()
process_completion(model) # 0.14837, 0.15479 @ 100k (1000x25), 0.1495 @ 400k

from mealpy.human_based import QSA # Queuing Search Algorithm
model = QSA.LevyQSA(problem_description(True), epoch=333, pop_size=100)
model.solve()
process_completion(model) # 0.155, 0.162 @ 100k (333x100), 0.157 @ 100k (1000x33) 0.1464 @ 300k

from mealpy.human_based import QSA # Queuing Search Algorithm (Improved)
model = QSA.ImprovedQSA(problem_description(True), epoch=250, pop_size=100)
model.solve()
process_completion(model) # 0.149, 0.147, 0.149, 0.149 @ 100k (250x100), 0.1465 @ 400k

from mealpy.human_based import QSA # Queuing Search Algorithm
model = QSA.OriginalQSA(problem_description(True), epoch=333, pop_size=100)
model.solve()
process_completion(model) # 0.148, 0.151, 0.149 @ 100k (333x100)

# from mealpy.human_based import SARO # Search and Rescue Optimization
# model = SARO.BaseSARO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.160 @ 200k

from mealpy.human_based import SARO # Search and Rescue Optimization (Original) -- best yet??
model = SARO.OriginalSARO(problem_description(True), epoch=500, pop_size=100)
model.solve()
process_completion(model) # 0.1462874499455973, 0.1464892747988185, 0.14629692591064095 @ 100k (500x100), 0.14628342250169188 @ 200k

# from mealpy.human_based import SSDO # Social Ski-Driver Optimization
# model = SSDO.BaseSSDO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.52 @ 100k

from mealpy.human_based import TLO # Teaching-Learning-based Optimization
model = TLO.BaseTLO(problem_description(True), epoch=500, pop_size=100)
model.solve()
process_completion(model) # 0.1488820, 0.1567, 0.1470  @ 100k (500x100), 0.1464876 @ 200k

from mealpy.human_based import TLO # Teaching-Learning-based Optimization (original)
model = TLO.OriginalTLO(problem_description(True), epoch=500, pop_size=100)
model.solve()
process_completion(model) # 0.1469, 0.1561 @ 100k (500x100), 0.14677 @ 200k

# from mealpy.human_based import TLO # Teaching-Learning-based Optimization (improved)
# model = TLO.ITLO(problem_description(True), epoch=500, pop_size=100)
# model.solve()
# process_completion(model) # 0.23, 0.19 @ 100k (500x100)

# -------------------------------------------------------------------------------------------------------------------------
# MUSIC_BASED: Algorithms from mealpy.music_based
# -------------------------------------------------------------------------------------------------------------------------

# from mealpy.music_based import HS # Harmony Search
# model = HS.BaseHS(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.52 @ 100k
#
# from mealpy.music_based import HS # Harmony Search
# model = HS.OriginalHS(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) #0.68 @ 100k

# -------------------------------------------------------------------------------------------------------------------------
# MATH_BASED: Algorithms from mealpy.math_based
# -------------------------------------------------------------------------------------------------------------------------

# from mealpy.math_based import AOA # Arithmetic Optimization Algorithm
# model = AOA.OriginalAOA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.149, 0.161, 0.166 @ 100k

# from mealpy.math_based import HC # Hill Climbing
# model = HC.BaseHC(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.157 @ 100k but insanely slow

# from mealpy.math_based import HC # Hill Climbing
# model = HC.OriginalHC(problem_description(True), epoch=2000, pop_size=100)
# model.solve()
# process_completion(model) # 0.161, 0.166, 0.163 @ 100k

# from mealpy.math_based import SCA # Sine Cosine Algorithm
# model = SCA.BaseSCA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.155, 0.163, 0.157 @ 100k

# from mealpy.math_based import SCA # Sine Cosine Algorithm
# model = SCA.OriginalSCA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19, 0.18 @ 100k

# -------------------------------------------------------------------------------------------------------------------------
# PHYSICS_BASED: Algorithms from mealpy.physics_based
# -------------------------------------------------------------------------------------------------------------------------

# from mealpy.physics_based import ASO # Atom Search Optimization
# model = ASO.BaseASO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # AttributeError: module 'numpy' has no attribute 'randint'

# from mealpy.physics_based import ArchOA # Archimedes Optimization
# model = ArchOA.OriginalArchOA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.39 @ 100k

from mealpy.physics_based import EFO # Electromagnetic Field Optimization
model = EFO.BaseEFO(problem_description(True), epoch=1000, pop_size=100)
model.solve()
process_completion(model) # 0.159, 0.157, 0.1467, 0.161, 0.161 @ 100k

from mealpy.physics_based import EO # Equilibrium Optimization
model = EO.BaseEO(problem_description(True), epoch=1000, pop_size=100)
model.solve()
process_completion(model) # 0.151, 0.151, 0.153, 0.152, 0.151 @ 100k

# from mealpy.physics_based import EO # Equilibrium Optimization
# model = EO.ModifiedEO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.154, 0.152, 0.152 @ 200k

# from mealpy.physics_based import EO # Equilibrium Optimization
# model = EO.AdaptiveEO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.155, 0.155, 0.155 @ 100k

# from mealpy.physics_based import HGSO # Henry Gas Solubility Optimizer
# model = HGSO.BaseHGSO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.23 @ 115k

# from mealpy.physics_based import MVO # Multi-Verse Optimizer
# model = MVO.BaseMVO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.166, 0.158 @ 100k

from mealpy.physics_based import MVO # Multi-Verse Optimizer
model = MVO.OriginalMVO(problem_description(True), epoch=1000, pop_size=100)
model.solve()
process_completion(model) # 0.152, 0.161, 0.153, 0.151 @ 100k

# from mealpy.physics_based import NRO # Nuclear Reaction Optimizer
# model = NRO.BaseNRO(problem_description(True), epoch=1000, pop_size=33)
# model.solve()
# process_completion(model) # 0.162, 0.151, 0.153 @ 300k

# from mealpy.physics_based import SA # Simulated Annealing
# model = SA.BaseSA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # slow, convergence failing

# from mealpy.physics_based import TWO # Tug of War Optimization
# model = TWO.BaseTWO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.36 @ 100k slow

# from mealpy.physics_based import TWO # Tug of War Optimization (Oppo)
# model = TWO.OppoTWO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.38 @ 100k

# from mealpy.physics_based import TWO # Tug of War Optimization (Levy)
# model = TWO.LevyTWO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # @ proportion error

# from mealpy.physics_based import TWO # Tug of War Optimization (Improved)
# model = TWO.ImprovedTWO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.29 @ 100k

# from mealpy.physics_based import WDO # Wind Driven Optimization
# model = WDO.BaseWDO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.17, 0.18 @ 100k

# -------------------------------------------------------------------------------------------------------------------------
# PROBABILISTIC_BASED: Algorithms from mealpy.probabilistic_based
# -------------------------------------------------------------------------------------------------------------------------

# from mealpy.probabilistic_based import CEM # Cross-Entropy Method
# model = CEM.BaseCEM(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.35 @ 100k

# -------------------------------------------------------------------------------------------------------------------------
# SYSTEM_BASED: Algorithms from mealpy.physics_based
# -------------------------------------------------------------------------------------------------------------------------

# from mealpy.system_based import AEO # Artificial Ecosystem-based Optimization
# model = AEO.OriginalAEO(problem_description(True), epoch=1000, pop_size=50)
# model.solve()
# process_completion(model) # 0.21 @ 100k, 0.20 @ 200k
#
# from mealpy.system_based import AEO # Artificial Ecosystem-based Optimization
# model = AEO.ImprovedAEO(problem_description(True), epoch=1000, pop_size=50)
# model.solve()
# process_completion(model) # 0.28 @ 100k

# from mealpy.system_based import AEO # Artificial Ecosystem-based Optimization
# model = AEO.EnhancedAEO(problem_description(True), epoch=1000, pop_size=50)
# model.solve()
# process_completion(model) # error crashed

# from mealpy.system_based import AEO # Artificial Ecosystem-based Optimization
# model = AEO.ModifiedAEO(problem_description(True), epoch=1000, pop_size=50)
# model.solve()
# process_completion(model) # error crashed

# from mealpy.system_based import AEO # Artificial Ecosystem-based Optimization
# model = AEO.AdaptiveAEO(problem_description(True), epoch=1000, pop_size=50)
# model.solve()
# process_completion(model) # error crashed

from mealpy.system_based import GCO # Germinal Center Optimization
model = GCO.BaseGCO(problem_description(True), epoch=1000, pop_size=100)
model.solve()
process_completion(model) # 0.149, 0.151, 0.150, 0.152, 0.151  @ 100k

# from mealpy.system_based import WCA # Water Cycle Algorithm
# model = WCA.BaseWCA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.156, 0.160, 0.182  @ 100k

# -------------------------------------------------------------------------------------------------------------------------
# SWARM_BASED: Algorithms from mealpy.human_based
# -------------------------------------------------------------------------------------------------------------------------

# from mealpy.swarm_based import ABC  # Artificial Bee Colony
# model = ABC.BaseABC(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19, 0.20 @ 100k

# from mealpy.swarm_based import ALO  # Ant Lion Optimizer Original
# model = ALO.OriginalALO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.56 @ 100k

# from mealpy.swarm_based import ALO  # Ant Lion Optimizer Base
# model = ALO.BaseALO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.81 @ 100k

# from mealpy.swarm_based import AO  # Aquila Optimization
# model = AO.OriginalAO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19, 0.18 @ 100k

# from mealpy.swarm_based import BA  # Bat-inspired Algorithm (basic)
# model = BA.BasicBA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.41 @ 100k

# from mealpy.swarm_based import BA  # Bat-inspired Algorithm (original)
# model = BA.OriginalBA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.54 @ 100k

# from mealpy.swarm_based import BA  # Bat-inspired Algorithm (base)
# model = BA.BaseBA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.164, 0.22  @ 100k

# from mealpy.swarm_based import BES  # Bald Eagle Search
# model = BES.BaseBES(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19, 0.18 @ 300k -- gets close fast and then stalls

# from mealpy.swarm_based import BFO  # Bacterial Foraging Optimization
# model = BFO.OriginalBFO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # unusably slow

# from mealpy.swarm_based import BFO  # Adaptive Bacterial Foraging Optimization
# model = BFO.ABFO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.21 @ 417k

# from mealpy.swarm_based import BSA  # Bird Swarm Algorithm
# model = BSA.BaseBSA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19, 0.29 @ 100k

# from mealpy.swarm_based import BeesA  # Bees Algorithm
# model = BeesA.BaseBeesA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19 @ 750k

# from mealpy.swarm_based import BeesA  # Bees Algorithm (probabilistic form)
# model = BeesA.ProbBeesA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19 @ 1 million evals

# from mealpy.swarm_based import COA  # Coyote Optimization Algorithm
# model = COA.BaseCOA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.27 @ 100k

from mealpy.swarm_based import CSA  # Cuckoo Search Algorithm (original)
model = CSA.BaseCSA(problem_description(True), epoch=1000, pop_size=100, p_a=0.3)
model.solve()
process_completion(model) # 0.23, 0.26 @ 130k

# from mealpy.swarm_based import CSO  # Cat Swarm Optimization
# model = CSO.BaseCSO(problem_description(True), epoch=1000, pop_size=100, mixture_ratio=0.15, smp=5, spc=False, cdc=0.8, srd=0.15, c1=0.4, w_minmax=(0.4, 0.9), selected_strategy=1)
# model.solve()
# process_completion(model) # 0.21 @ lots of function evals, doesn't produce good eval estimate

# from mealpy.swarm_based import DO  # Dragonfly Optimization
# model = DO.BaseDO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.39 @ 200k very slow iters

# from mealpy.swarm_based import EHO  # Elephant Herding Optimization
# model = EHO.BaseEHO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.32, 0.31 @ 100k

# from mealpy.swarm_based import FOA  # Fruit-fly Optimization Algorithm (original)
# model = FOA.OriginalFOA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 1.16 @ 100k

# from mealpy.swarm_based import FOA  # Fruit-fly Optimization Algorithm (base)
# model = FOA.BaseFOA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.88 @ 100k

# from mealpy.swarm_based import FOA  # Whale Fruit-fly Optimization Algorithm
# model = FOA.WhaleFOA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.25 @ 100k

# from mealpy.swarm_based import FireflyA  # Firefly Algorithm
# model = FireflyA.BaseFireflyA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # instantly gives ValueError: In solution.py, value_from_proportion got min_value > max_value.

# from mealpy.swarm_based import GOA  # Grasshopper Optimization Algorithm
# model = GOA.BaseGOA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.21 @ 100k but VERY slow evals

# from mealpy.swarm_based import GWO  # Gray Wolf Optimizer
# model = GWO.BaseGWO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.178, 0.189, 0.27 @ 100k

# from mealpy.swarm_based import GWO  # Random Walk Gray Wolf Optimizer
# model = GWO.RW_GWO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.27, 0.35 @ 100k

# from mealpy.swarm_based import HGS  # Hunger Games Search
# model = HGS.OriginalHGS(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) 0.23 # @ 100k

# from mealpy.swarm_based import HHO  # Harris Hawks Optimization
# model = HHO.BaseHHO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.20, 0.26, @ 150k

# from mealpy.swarm_based import JA  # Jaya Algorithm (base) -- Gets close very fast (100-200 epochs) but stalls out. Has no hyperparameters. Same with variants below.
# model = JA.BaseJA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.17, 0.19, 0.18, 0.20 @ 100k
#
# from mealpy.swarm_based import JA  # Jaya Algorithm (original)
# model = JA.OriginalJA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.17, 0.17, 0.19, 0.16 @ 100k
#
# from mealpy.swarm_based import JA  # Jaya Algorithm (levy)
# model = JA.LevyJA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.17, 0.20, 0.31 @ 100k

from mealpy.swarm_based import MFO  # Moth Flame Optimizer (base) -- works great, but modifications in package not from a paper?
model = MFO.BaseMFO(problem_description(True), epoch=1000, pop_size=100)
model.solve()
process_completion(model) # 0.148, 0.150, 0.149, 0.148, 0.148  @ 100k

# from mealpy.swarm_based import MFO  # Moth Flame Optimizer (original)
# model = MFO.OriginalMFO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.44, 0.31 @ 100k

# from mealpy.swarm_based import MRFO  # Manta Ray Foraging Optimization
# model = MRFO.BaseMRFO(problem_description(True), epoch=500, pop_size=100)
# model.solve()
# process_completion(model) # 0.21 @ 100k, 0.18 @ 200k

# from mealpy.swarm_based import MSA  # Moth Search Algorithm
# model = MSA.BaseMSA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.23 @ 100k

# from mealpy.swarm_based import NMRA # Naked Mole Rat Algorithm
# model = NMRA.BaseNMR(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.166, 0.158, 0.183 @ 100k
#
# from mealpy.swarm_based import NMRA # Naked Mole Rat Algorithm (improved)
# model = NMRA.ImprovedNMR(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.182, 0.26 @ 100k

# from mealpy.swarm_based import PFA # Pathfinder Algorithm
# model = PFA.BasePFA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.53 @ 100k, absolutely no convergence, broken?

# from mealpy.swarm_based import PSO # Particle Swarm Optimization
# model = PSO.BasePSO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19, 0.20 @ 100k
#
# from mealpy.swarm_based import PSO # Phasor Particle Swarm Optimization
# model = PSO.PPSO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19, 0.20 @ 100k

from mealpy.swarm_based import PSO # Self-organising Hierarchical PSO with Time-Varying Acceleration Coefficients
model = PSO.HPSO_TVAC(problem_description(True), epoch=1000, pop_size=100)
model.solve()
process_completion(model) # 0.157, 0.149, 0.151, 0.155, 0.150 @ 100k

# from mealpy.swarm_based import PSO # Chaos Particle Swarm Optimization
# model = PSO.C_PSO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # did not run, errors in value_from_proportion

# from mealpy.swarm_based import PSO # Comprehensive Learning Particle Swarm Optimization
# model = PSO.CL_PSO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.17, 0.166, 0.17 @ 100k

# from mealpy.swarm_based import SFO # Sailfish Optimizer
# model = SFO.BaseSFO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # @ did not run, errors

# from mealpy.swarm_based import SFO # Sailfish Optimizer (improved)
# model = SFO.ImprovedSFO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.23 @ 100k

# from mealpy.swarm_based import SHO # Spotted Hyena Optimizer
# model = SHO.BaseSHO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.99 @ 100k, no convergence at all

# from mealpy.swarm_based import SLO # Sea Lion Optimizer (base)
# model = SLO.BaseSLO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19, 0.21 @ 100k

# from mealpy.swarm_based import SLO # Sea Lion Optimizer (modified)
# model = SLO.ModifiedSLO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19, 0.23 @ 100k

from mealpy.swarm_based import SLO # Sea Lion Optimizer (improved)
model = SLO.ISLO(problem_description(True), epoch=1000, pop_size=100)
model.solve()
process_completion(model) # 0.151, 0.160, 0.164, 0.178 @ 100k

from mealpy.swarm_based import SRSR # Swarm Robotics Search and Rescue
model = SRSR.BaseSRSR(problem_description(True), epoch=1000, pop_size=50)
model.solve()
process_completion(model) # 0.151, 0.157, 0.180 @ 100k, 0.163 @ 200k

# from mealpy.swarm_based import SSA # Sparrow Search Algorithm (base)
# model = SSA.BaseSSA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.21 @ 100k

# from mealpy.swarm_based import SSA # Sparrow Search Algorithm (original)
# model = SSA.OriginalSSA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.19 @ 190k

# from mealpy.swarm_based import SSO # Salp Swarm Optimizer (base)
# model = SSO.BaseSSO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # did not run, proportion errors

# from mealpy.swarm_based import SSpiderA # Social Spider Algorithm
# model = SSpiderA.BaseSSpiderA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.24 @ 100k

# from mealpy.swarm_based import SSpiderO # Social Spider Algorithm (original)
# model = SSpiderO.BaseSSpiderO(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.18 @ 1.8 mil

# from mealpy.swarm_based import WOA # Whale Optimization Algorithm
# model = WOA.BaseWOA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.20, 0.21 @ 100k

# from mealpy.swarm_based import WOA # Hybrid Improved Whale Optimization Algorithm
# model = WOA.HI_WOA(problem_description(True), epoch=1000, pop_size=100)
# model.solve()
# process_completion(model) # 0.22 @ 100k
