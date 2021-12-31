import json

from maneuvermodel.maneuveringfish import ManeuveringFish
from maneuvermodel.optimize import optimal_maneuver
from maneuvermodel.constants import DEFAULT_OPT_N, DEFAULT_OPT_ITERATIONS

def optimal_maneuver_from_saved_setup(saved_setup_file, **kwargs):
    """ Retrieves the inputs specifying the maneuver setup (velocity, detection position, etc), not the
        specific attributes (radius, thrusts, etc) of the particular maneuver object saved. Calculates those anew
        instead. Any maneuver input can be altered by passing it as a keyword argument.
         """
    with open(saved_setup_file) as f:
        loaded = json.load(f)

    fish = ManeuveringFish(kwargs.get('fork_length', loaded['fish_fork_length']),
                           kwargs.get('focal_velocity', loaded['fish_focal_velocity']),
                           kwargs.get('base_mass', loaded['fish_base_mass']),
                           kwargs.get('temperature', loaded['fish_temperature']),
                           kwargs.get('SMR', loaded['fish_SMR']),
                           kwargs.get('max_thrust', loaded['fish_max_thrust']),
                           kwargs.get('NREI', loaded['fish_NREI']),
                           kwargs.get('use_total_cost', loaded['fish_use_total_cost']),
                           kwargs.get('disable_wait_time', loaded['fish_disable_wait_time'])
                           )
    return optimal_maneuver(fish,
                            prey_velocity=kwargs.get('prey_velocity', loaded['prey_velocity']),
                            n=kwargs.get('n', DEFAULT_OPT_N),
                            max_iterations=kwargs.get('max_iterations', DEFAULT_OPT_ITERATIONS),
                            detection_point_3D=kwargs.get('detection_point_3D', (loaded['det_x'], loaded['det_y'], 0)),
                            tracked=True,
                            return_optimization_model=True
                            )

def save_setup_inputs(maneuver, destination):
    """ Saves the inputs specifying the maneuver setup (velocity, detection position, etc), not the
        specific attributes (radius, thrusts, etc) of the particular maneuver object passed.
        'destination' should specify a json file """
    inputs = {'prey_velocity': maneuver.prey_velocity,
              'det_x': maneuver.det_x,
              'det_y': maneuver.det_y,
              'fish_fork_length': maneuver.fish.fork_length,
              'fish_focal_velocity': maneuver.fish.focal_velocity,
              'fish_temperature': maneuver.fish.temperature,
              'fish_base_mass': maneuver.fish.base_mass,
              'fish_SMR': maneuver.fish.SMR,
              'fish_max_thrust': maneuver.fish.max_thrust,
              'fish_NREI': maneuver.fish.NREI,
              'fish_use_total_cost': maneuver.fish.use_total_cost,
              'fish_disable_wait_time': maneuver.fish.disable_wait_time
              }
    with open(destination, 'w', encoding='utf-8') as f:
        json.dump(inputs, f, ensure_ascii=False, indent=4)
        print("Saved inputs to ", destination)
