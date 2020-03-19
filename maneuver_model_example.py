from maneuvermodel import maneuveringfish, optimize, visualize

focal_current_speed = 29
mass = 45 # default 45
fork_length = 0.1 * (mass / 4.8e-06)**(1/3.1) # reverse of the dolly varden length-mass regression used in Fish.py
NREI = 0.08
focal_return_tolerance = 0.1 # 1 mm
detection_vector = (-12.0, 12.0, 0.0)
disable_wait_time = True # want to be false for environmental relationships, true for spatial relationships
fix_u_ms_at_opt_temp = False
temperature = 10
max_thrust = 97

use_starting_iterations = False
num_starting_populations = 15
num_starting_iterations = 1000
num_iterations = 4500
num_tests = 20
allow_retries = True

popsize = 4 # initially 6 
variant_scale = 1.5 # initially 10
mixing_ratio = 3.0 # initially 3.0

fish = maneuveringfish.ManeuveringFish(fork_length, focal_current_speed, temperature, mass, max_thrust, 0.0, False, 0.1, disable_wait_time, fix_u_ms_at_opt_temp)
fittest_solution = optimize.optimal_maneuver(fish, detection_point_3D = detection_vector, popsize = popsize, variant_scale = variant_scale, mixing_ratio = mixing_ratio, iterations = num_iterations, use_starting_iterations = use_starting_iterations, num_starting_populations=num_starting_populations, num_starting_iterations=num_starting_iterations, allow_retries=allow_retries)
visualize.summarize_solution(fittest_solution, display = True, title = None, should_print_dynamics = True)

