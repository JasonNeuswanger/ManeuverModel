import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.colors as colors
import seaborn as sns
from matplotlib.collections import LineCollection
from .constants import *
from .maneuver import maneuver_from_proportions

smallestfontsize = 8
smallfontsize = 10
bigfontsize = 12

param_labels = ['Thrust (turn 1)', 'Thrust (straight 1)', 'Thrust (turn 2)', 'Thrust (straight 2)',
                'Thrust (turn 3)', 'Radius (turn 1)', 'Radius (turn 2)', 'Radius (turn 3)',
                'Thrust (straight 3A)', 'X (turn 3)', 'Wait time']

def summarize_solution(solution, display=True, title=None, export_path=None, should_print_dynamics=True, detailed=False, add_text_panel=False, plot_dpi=132):
    if solution.dynamics.energy_cost >= CONVERGENCE_FAILURE_COST:
        print("Cannot summarize solution in which the final straight failed to converge.")
        return
    plt.ioff()  # set interactive mode to off so the plot doesn't display until show() is called
    sns.set_style('ticks')
    figsize = (7.5+(2.6 if add_text_panel else 0), 5.0) if detailed else (6.5+(2.6 if add_text_panel else 0), 4.75)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, facecolor='w', figsize=figsize, dpi=plot_dpi)
    dynamicstuple = solution.dynamics.plottable_segments(solution)
    if should_print_dynamics: print_dynamics(dynamicstuple, solution.dynamics, solution)
    plot_spatial_path_2D(ax1, dynamicstuple, solution, detailed, False)  # water coords plot
    plot_spatial_path_2D(ax2, dynamicstuple, solution, detailed, True)  # ground coords plot
    plot_dynamics(ax3, dynamicstuple, solution, detailed)
    plot_swimming_costs(ax4, dynamicstuple, solution, detailed)
    if title is not None and detailed:
        fig.suptitle(title, fontsize=bigfontsize, weight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.24, left=0.09, right=(0.90 if not add_text_panel else 0.69), hspace=0.6, wspace=0.6)
    else:
        fig.subplots_adjust(top=0.95, bottom=0.24, left=0.09, right=(0.96 if not add_text_panel else 0.65), hspace=0.5, wspace=0.3)

    line_segtrans = lines.Line2D([], [], color='k', label='Segment transition', linewidth=0.5, linestyle='dotted', alpha=0.4)
    line_focvel = lines.Line2D([], [], color='k', label='Focal velocity', linewidth=0.5, linestyle='-', dashes=[0.5, 0.5, 2, 0.5, 0.5, 0.5, 2, 0.5], alpha=0.4)
    line_steady = lines.Line2D([], [], color='k', label='Steady swimming cost', linewidth=0.5, linestyle='dashed', alpha=0.4)
    line_SMR = lines.Line2D([], [], color='k', label='Standard metabolic rate', linewidth=0.5, linestyle='-', dashes=[0.5, 0.5], alpha=0.4)
    dot_focal_point = lines.Line2D([], [], color='k', label='Focal point', marker='s', markersize=3, linestyle='')
    dot_capture_point = lines.Line2D([], [], color='r', label='Capture point', marker='o', markersize=5, linestyle='')
    plt.legend(handles=[dot_focal_point, dot_capture_point, line_segtrans, line_focvel, line_steady, line_SMR], ncol=3, loc=(-1.35, -0.82), fontsize=smallfontsize)
    for ax in (ax1, ax2, ax3, ax4):
        sns.despine(ax=ax, top=True, right=True)
        ax.set(adjustable='datalim')
    if add_text_panel:
        p = solution.proportions()
        dyn = solution.dynamics
        text = "Energy cost:     {0:12.8f}\n".format(solution.energy_cost)
        text += "Pursuit duration:{0:12.8f}\n".format(solution.pursuit_duration)
        text += "Return duration: {0:12.8f}\n".format(solution.return_duration)
        text += "Total duration:  {0:12.8f}\n".format(solution.duration)
        text += "Fitness:         {0:12.8f}\n".format(solution.fitness)
        text += "Total distance:   {0:12.8f}\n".format(solution.path.total_length)
        text += "Mean speed        {0:12.8f}\n".format(solution.dynamics.mean_speed)
        text += "Final_turn_x adjusted? {0}\n".format("Yes" if solution.had_final_turn_adjusted else "No")

        text += "\n"
        text += "Parameters\n"
        text += "Turn 1 thrust: {0:7.3f} | {1:.3f}p\n".format(dyn.turn_1.u_thrust, p[0])
        text += "Strt 1 thrust: {0:7.3f} | {1:.3f}p\n".format(dyn.straight_1.u_thrust, p[1])
        text += "Turn 2 thrust: {0:7.3f} | {1:.3f}p\n".format(dyn.turn_2.u_thrust, p[2])
        text += "Strt 2 thrust: {0:7.3f} | {1:.3f}p\n".format(dyn.straight_2.u_thrust, p[3])
        text += "Turn 3 thrust: {0:7.3f} | {1:.3f}p\n".format(dyn.turn_1.u_thrust, p[4])
        text += "Strt 3 thru_a: {0:7.3f} | {1:.3f}p\n".format(dyn.straight_3.thrust_a, p[8])
        text += "Strt 3 thru_b: {0:7.3f} | (n/a)\n".format(dyn.straight_3.thrust_b)
        text += "           r1: {0:7.3f} | {1:.3f}p\n".format(solution.r1, p[5])
        text += "           r2: {0:7.3f} | {1:.3f}p\n".format(solution.r2, p[6])
        text += "           r3: {0:7.3f} | {1:.3f}p\n".format(solution.r3, p[7])
        text += " final_turn_x: {0:7.3f} | {1:.3f}p\n".format(solution.final_turn_x, p[9])
        text += "    wait time: {0:7.3f} | {1:.3f}p\n".format(solution.wait_time, p[10])
        text += "\n"
        text += "Stage    Length   Cost       Time   EndSpd\n"
        text += "Turn 1  {0:7.3f} {1:10.6f} {2:7.3f} {3:7.3f}\n".format(dyn.turn_1.length, dyn.turn_1.cost, dyn.turn_1.duration, dyn.turn_1.final_speed)
        text += "Strt 1  {0:7.3f} {1:10.6f} {2:7.3f} {3:7.3f}\n".format(dyn.straight_1.length, dyn.straight_1.cost, dyn.straight_1.duration, dyn.straight_1.final_speed)
        text += "Turn 2  {0:7.3f} {1:10.6f} {2:7.3f} {3:7.3f}\n".format(dyn.turn_2.length, dyn.turn_2.cost, dyn.turn_2.duration, dyn.turn_2.final_speed)
        text += "Strt 2  {0:7.3f} {1:10.6f} {2:7.3f} {3:7.3f}\n".format(dyn.straight_2.length, dyn.straight_2.cost, dyn.straight_2.duration, dyn.straight_2.final_speed)
        text += "Turn 3  {0:7.3f} {1:10.6f} {2:7.3f} {3:7.3f}\n".format(dyn.turn_3.length, dyn.turn_3.cost, dyn.turn_3.duration, dyn.turn_3.final_speed)
        text += "Str 3A  {0:7.3f} {1:10.6f} {2:7.3f} {3:7.3f}\n".format(dyn.straight_3.length_a, dyn.straight_3.cost_a, dyn.straight_3.duration_a, dyn.straight_3.final_speed_a)
        text += "Str 3B  {0:7.3f} {1:10.6f} {2:7.3f} {3:7.3f}\n".format(dyn.straight_3.length_b, dyn.straight_3.cost_b, dyn.straight_3.duration_b, solution.mean_water_velocity)
        text += "\n"
        text += "Prey velocity: {0:5.3f}\n".format(solution.prey_velocity)
        text += "Focal velocity: {0:5.3f}\n".format(solution.fish.focal_velocity)
        text += "Detection point 2D: {0:5.3f}, {1:5.3f}\n".format(solution.det_x, solution.det_y)
        fig.text(0.75, 0.93, text, transform=fig.transFigure, fontsize=7, family='monospace', va='top')
    if display:
        fig.show()
    if export_path is not None:
        fig.savefig(export_path)
        if not display: plt.close(fig)
    sns.set_style('darkgrid')


def preyarrow(ax, pt1, pt2, color, head_width, overhang=0.4):
    ax.arrow(pt1[0], pt1[1], pt2[0] - pt1[0], pt2[1] - pt1[1], color=color, length_includes_head=True, head_width=head_width, overhang=overhang)

def build_spatial_path(dynamicstuple, solution, v):
    # Pass this v = 0 for water coords, v = maneuver.mean_water_velocity for ground coords
    # This should take the parametric (time, position relative to length of path) coords from each path and convert into (x,y)[t] coords
    # based on the known starting times/positions of each leg of the path and referenced from a single starting t = 0 for the whole maneuver,
    # not for each leg. Then use time to convert the x-positions into ground coords. This returns data for a 2-D ground coordinates path
    # in the plane of the maneuver.
    (turn_1, straight_1, turn_2, straight_2, turn_3, straight_3) = dynamicstuple
    path = solution.path
    turn_1_times = turn_1.plottimes
    straight_1_times = straight_1.plottimes + turn_1.duration
    turn_2_times = turn_2.plottimes + turn_1.duration + straight_1.duration
    straight_2_times = straight_2.plottimes + turn_1.duration + straight_1.duration + turn_2.duration
    turn_3_times = turn_3.plottimes + turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration
    straight_3_times = straight_3.plottimes + turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration + turn_3.duration

    def turn_1_coords(p):  # 'p' is the proportion of t1 completed at the returned coordinates; so p = 0.2 returns the position 20 % of the way through turn 1
        # sign = 1
        # angle = path.turn_1_angles[0] + sign * p * path.theta_1
        angle = path.turn_1_angles[0] + (path.turn_1_angles[1] - path.turn_1_angles[0]) * p
        return path.turn_1_center + path.turn_1_radius * np.array([np.cos(angle), np.sin(angle)])

    def turn_2_coords(p):
        # dir2 is -1
        # sign = 1 # path.turn_2_direction * np.sign(path.turn_2_center[1])
        # angle = path.turn_2_angles[0] + sign * p * path.theta_2
        angle = path.turn_2_angles[0] + (path.turn_2_angles[1] - path.turn_2_angles[0]) * p
        return path.turn_2_center + path.turn_2_radius * np.array([np.cos(angle), np.sin(angle)])

    def turn_3_coords(p):
        angle = path.turn_3_angles[0] + (path.turn_3_angles[1] - path.turn_3_angles[0]) * p
        return path.turn_3_center + path.turn_3_radius * np.array([np.cos(angle), np.sin(angle)])

    def straight_1_coords(p):
        return path.tangent_point_B + p * (path.capture_point - path.tangent_point_B)

    def straight_2_coords(p):
        return path.tangent_point_D + p * (path.tangent_point_E - path.tangent_point_D)

    def straight_3_coords(p):
        return path.tangent_point_F + p * (path.focal_return_point - path.tangent_point_F)

    # Assign ground coordinate positions (starting with water coordinates in this assignment)
    turn_1_groundcoord_positions = [turn_1_coords(p) for p in turn_1.plotpositions]
    straight_1_groundcoord_positions = [straight_1_coords(p) for p in straight_1.plotpositions]
    turn_2_groundcoord_positions = [turn_2_coords(p) for p in turn_2.plotpositions]
    straight_2_groundcoord_positions = [straight_2_coords(p) for p in straight_2.plotpositions]
    turn_3_groundcoord_positions = [turn_3_coords(p) for p in turn_3.plotpositions]
    straight_3_groundcoord_positions = [straight_3_coords(p) for p in straight_3.plotpositions]
    # Now convert them from water coordinates to ground coordinates
    for i in range(len(turn_1_times)): turn_1_groundcoord_positions[i][0] += turn_1_times[i] * v
    for i in range(len(straight_1_times)): straight_1_groundcoord_positions[i][0] += straight_1_times[i] * v
    for i in range(len(turn_2_times)): turn_2_groundcoord_positions[i][0] += turn_2_times[i] * v
    for i in range(len(straight_2_times)): straight_2_groundcoord_positions[i][0] += straight_2_times[i] * v
    for i in range(len(turn_3_times)): turn_3_groundcoord_positions[i][0] += turn_3_times[i] * v
    for i in range(len(straight_3_times)): straight_3_groundcoord_positions[i][0] += straight_3_times[i] * v
    x, y = np.array(turn_1_groundcoord_positions + straight_1_groundcoord_positions + turn_2_groundcoord_positions + straight_2_groundcoord_positions + turn_3_groundcoord_positions + straight_3_groundcoord_positions).T
    s = np.concatenate((turn_1.plotspeeds, straight_1.plotspeeds, turn_2.plotspeeds, straight_2.plotspeeds, turn_3.plotspeeds, straight_3.plotspeeds)).T
    t = np.concatenate((turn_1_times, straight_1_times, turn_2_times, straight_2_times, turn_3_times, straight_3_times)).T
    num_arrows = 3
    arrow_times = np.linspace(t[0], t[-1], num_arrows + 2)[1:-1]  # preferred times for arrows at equal intervals throughout the maneuver
    arrow_indices = [(np.abs(t - arrow_time)).argmin() for arrow_time in arrow_times]  # find the indices for those preferred times
    capture_index = int(len(turn_1_times) + len(straight_1_times))
    return (x, y, t, s, arrow_indices, capture_index)


def draw_ground_coords_path_3D(dynamicstuple, solution, figure, actual_capture_point=None, multiplier=10, colorbar=False, predicted_capture_point_color=(1, 1, 0), actual_capture_point_color=(0, 1, 0)):
    from mayavi import mlab
    from driftmodel.graphics import lines
    (turn_1, straight_1, turn_2, straight_2, turn_3, straight_3) = dynamicstuple
    (x2d, y2d, t, s, arrow_indices, capture_index) = build_spatial_path(dynamicstuple, solution, solution.fish.water_velocity)
    path = solution.path
    # reaction_point = path.capture_point # commented 2 lines here replaced by the one below, not deleting until double-checking
    # predicted_capture_point = solution.to_3D(reaction_point + (solution.fish.water_velocity * (turn_1.duration + straight_1.duration), 0))
    predicted_capture_point = solution.predicted_capture_point_3D_groundcoords()
    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []
    arrx = []
    arry = []
    arrz = []
    arru = []
    arrv = []
    arrw = []
    prevsol = []

    def ar(x):
        return multiplier * np.array(x)  # multiplier here for difference in units between maneuver model and data (I use 10 to convert modeled cm -> measured mm)

    cmap = 'RdYlBu'  # other options: RdYlBu, cool, spectral
    max_colorbar_speed = 60.0
    cnorm = colors.Normalize(vmin=0.0, vmax=max_colorbar_speed)  # scale factor for colors stays in cm/s so the colorbar does too
    for i in range(len(x2d)):
        sol = solution.to_3D(np.array((x2d[i], y2d[i])))
        if i > 0:
            x1.append(prevsol[0])
            y1.append(prevsol[1])
            z1.append(prevsol[2])
            x2.append(sol[0])
            y2.append(sol[1])
            z2.append(sol[2])
        if i in arrow_indices:
            dirvec = ((sol - prevsol) / np.linalg.norm(sol - prevsol)) * 0.01 * solution.fish.fork_length
            arrx = np.array((prevsol[0]))
            arry = np.array((prevsol[1]))
            arrz = np.array((prevsol[2]))
            arru = np.array((dirvec[0]))
            arrv = np.array((dirvec[1]))
            arrw = np.array((dirvec[2]))
            conecolor = plt.get_cmap(cmap)(cnorm(s[i]))[:3]
            mlab.quiver3d(ar(arrx), ar(arry), ar(arrz), ar(arru), ar(arrv), ar(arrw), figure=figure, line_width=1, mode='cone', color=conecolor, scale_factor=4.0)
        prevsol = sol
        # Note I tried reversing the colormap so blue = slow and red = fast, but then it shows the slow spots as fast and vice versa; this is the only way to not mess that up, it seems.
    mlab.points3d(ar((predicted_capture_point[0])), ar((predicted_capture_point[1])), ar((predicted_capture_point[2])), figure=figure, color=predicted_capture_point_color, scale_factor=5.0)  # previously used 5, going to 10 for grayling demo
    if actual_capture_point is not None:
        mlab.points3d(np.array((actual_capture_point[0])), np.array((actual_capture_point[1])), np.array((actual_capture_point[2])), figure=figure, color=actual_capture_point_color, scale_factor=5.0)
        mlab.plot3d(np.array((multiplier * predicted_capture_point[0], actual_capture_point[0])), np.array((multiplier * predicted_capture_point[1], actual_capture_point[1])), np.array((multiplier * predicted_capture_point[2], actual_capture_point[2])), figure=figure, color=(1, 1, 1),
                    tube_radius=0.7, opacity=0.5)

    lines(ar(x1), ar(y1), ar(z1), ar(x2), ar(y2), ar(z2), s1=np.array(s[:(len(x2d) - 1)]), figure=figure, line_width=2, colormap='RdYlBu', vmin=0, vmax=max_colorbar_speed, reverse_colormap=False, colorbar=False)
    # lines(ar(x1), ar(y1), ar(z1), ar(x2), ar(y2), ar(z2), s1 = np.array(s[:(len(x2d)-1)]), figure = figure, line_width = 2, colormap = 'RdYlBu', vmin=0, vmax=max_colorbar_speed, reverse_colormap = False, colorbar = colorbar, bartitle='Speed')


def plot_spatial_path_2D(ax, dynamicstuple, solution, detailed, use_groundcoords):
    if use_groundcoords:
        title = "Ground coordinates"
        v = solution.mean_water_velocity
        ax.text(-0.26, 1.06, 'b', transform=ax.transAxes, size=bigfontsize, weight='bold')
    else:
        title = "Water coordinates"
        v = 0
        ax.text(-0.21, 1.06, 'a', transform=ax.transAxes, size=bigfontsize, weight='bold')
    (turn_1, straight_1, turn_2, straight_2, turn_3, straight_3) = dynamicstuple
    (x, y, t, s, arrow_indices, capture_index) = build_spatial_path(dynamicstuple, solution, v)
    path = solution.path
    detection_point = path.detection_point
    reaction_point = path.capture_point
    capture_point = reaction_point + (v * (turn_1.duration + straight_1.duration), 0)
    ax.set_aspect('equal')
    padding = max(max(x) - min(x), max(y) - min(y)) / 30
    ax.set_ylim(ymin=min(y) - padding, ymax=max(y) + padding)
    ax.set_xlim(xmin=min(x) - padding, xmax=max(x) + padding)
    arrowsize = max(abs(ax.get_xlim()[1] - ax.get_xlim()[0]), abs(ax.get_ylim()[1] - ax.get_ylim()[0])) / 30
    cmap = 'inferno'
    cnorm = colors.Normalize(vmin=min(s), vmax=1.2 * max(s))

    def arrow_at_index(i):
        ax.arrow(x[i], y[i], x[i + 1] - x[i], y[i + 1] - y[i], color=plt.get_cmap(cmap)(cnorm(s[i])), head_width=arrowsize, overhang=0.4)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=cnorm)
    lc.set_array(s)
    lc.set_linewidth(1.5)
    ax.add_collection(lc)
    if use_groundcoords:
        axcb = plt.colorbar(lc, ax=ax)
        axcb.set_label('Speed (cm/s)')
        flowarrow_ycoord = 0.15 if detailed else 0.86
        ax.text(0.12, flowarrow_ycoord + 0.04, 'Flow', transform=ax.transAxes, size=smallestfontsize)
        ax.arrow(0.1, flowarrow_ycoord, 0.15, 0.0, transform=ax.transAxes, color='k', head_width=0.03, linewidth=0.5, overhang=0.4)
        if detailed: preyarrow(ax, reaction_point, capture_point, 'k', head_width=0.8 * arrowsize, overhang=0.4)

    for ind in arrow_indices: arrow_at_index(ind)
    if abs(detection_point[0] - reaction_point[0]) > 0.001: preyarrow(ax, detection_point, reaction_point, 'g', head_width=0.8 * arrowsize)
    ax.plot(capture_point[0], capture_point[1], marker='o', ls='', color='r', markersize=5, zorder=10)
    ax.plot(0, 0, marker='s', ls='', color='k', markersize=3, zorder=10)
    ax.set_xlabel('x (cm)', fontsize=smallfontsize)
    ax.set_ylabel('y (cm)', fontsize=smallfontsize)
    ax.set_title(title, fontsize=smallfontsize)


def plot_dynamics(ax, dynamicstuple, maneuver, detailed):
    (turn_1, straight_1, turn_2, straight_2, turn_3, straight_3) = dynamicstuple
    ax.axvline(x=turn_1.duration, linestyle='dotted', color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(x=turn_1.duration + straight_1.duration, linestyle='dotted', color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(x=turn_1.duration + straight_1.duration + turn_2.duration, linestyle='dotted', color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(x=turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration, linestyle='dotted', color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(x=turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration + turn_3.duration, linestyle='dotted', color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(x=turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration + turn_3.duration + straight_3.duration_a, linestyle='dotted', color='k', linewidth=0.5, alpha=0.4)
    ax.axhline(y=maneuver.mean_water_velocity, linestyle='-', dashes=[0.5, 0.5, 2, 0.5, 0.5, 0.5, 2, 0.5], color='k', linewidth=0.5, alpha=0.4)
    turn_1_times = turn_1.plottimes
    straight_1_times = straight_1.plottimes + turn_1.duration
    turn_2_times = turn_2.plottimes + turn_1.duration + straight_1.duration
    straight_2_times = straight_2.plottimes + turn_1.duration + straight_1.duration + turn_2.duration
    turn_3_times = turn_3.plottimes + turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration
    straight_3_times = straight_3.plottimes + turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration + turn_3.duration
    ax.margins(0.02)
    times = np.concatenate((turn_1_times, straight_1_times, turn_2_times, straight_2_times, turn_3_times, straight_3_times))
    speeds = np.concatenate((turn_1.plotspeeds, straight_1.plotspeeds, turn_2.plotspeeds, straight_2.plotspeeds, turn_3.plotspeeds, straight_3.plotspeeds))
    accelerations = np.concatenate((turn_1.plotaccelerations, straight_1.plotaccelerations, turn_2.plotaccelerations, straight_2.plotaccelerations, turn_3.plotaccelerations, straight_3.plotaccelerations))

    ax.plot(times, speeds, 'b', linewidth=1.5)
    if detailed:
        ax2 = ax.twinx()
        ax2.plot(times[1:len(times)], 0.01 * accelerations[1:len(accelerations)], 'g', alpha=0.2)  # skipping first element (no acceleration defined yet)
        ax2.set_ylabel('Acceleration (m/s' + r'$^2$' + ')', fontsize=smallfontsize)
        ax2.set_ylim(ymin=0.01 * 1.03 * min(accelerations), ymax=0.01 * 1.03 * max(accelerations))
        # sns.despine(ax=ax2, top=True, left=True) # this would be nice, but despine seems to have a bug and messes up the twinning, moving the axis to the left side instead

    ax.set_xlabel('Time (s)', fontsize=smallfontsize)
    ax.set_ylabel('Speed (cm/s)', fontsize=smallfontsize)
    ymax = 1.03 * max([max(turn_1.plotspeeds), max(turn_2.plotspeeds), max(turn_3.plotspeeds), max(straight_1.plotspeeds), max(straight_2.plotspeeds), max(straight_3.plotspeeds)])
    ax.set_ylim(ymin=0, ymax=ymax)
    ax.set_xlim(xmin=-0.02 * abs(max(straight_3_times) - min(turn_1_times)), xmax=max(straight_3_times))
    ax.text(-0.21, 0.95, 'c', transform=ax.transAxes, size=bigfontsize, weight='bold')
    if detailed:
        ax.set_title('Speed (mean = {0:.1f} cm/s)'.format(maneuver.dynamics.mean_speed), fontsize=smallfontsize)


def plot_swimming_costs(ax, dynamicstuple, solution, detailed):
    (turn_1, straight_1, turn_2, straight_2, turn_3, straight_3) = dynamicstuple
    ax.axvline(x=turn_1.duration, linestyle='dotted', color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(x=turn_1.duration + straight_1.duration, linestyle='dotted', color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(x=turn_1.duration + straight_1.duration + turn_2.duration, linestyle='dotted', linewidth=0.5, color='k', alpha=0.4)
    ax.axvline(x=turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration, linestyle='dotted', color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(x=turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration + turn_3.duration, linestyle='dotted', color='k', linewidth=0.5, alpha=0.4)
    ax.axhline(y=solution.fish.focal_swimming_cost_including_SMR, linestyle='dashed', color='k', linewidth=0.5, alpha=0.5)
    ax.axhline(y=solution.fish.coasting_cost_including_SMR, linestyle='-', dashes=[0.5, 0.5], color='k', linewidth=0.5, alpha=0.5)
    turn_1_times = turn_1.plottimes
    straight_1_times = straight_1.plottimes + turn_1.duration
    turn_2_times = turn_2.plottimes + turn_1.duration + straight_1.duration
    straight_2_times = straight_2.plottimes + turn_1.duration + straight_1.duration + turn_2.duration
    turn_3_times = turn_3.plottimes + turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration
    straight_3_times = straight_3.plottimes + turn_1.duration + straight_1.duration + turn_2.duration + straight_2.duration + turn_3.duration
    ax.margins(0.02)
    times = np.concatenate((turn_1_times, straight_1_times, turn_2_times, straight_2_times, turn_3_times, straight_3_times))
    SMR = solution.fish.SMR_J_per_s
    costs = SMR + np.concatenate((turn_1.plotcosts, straight_1.plotcosts, turn_2.plotcosts, straight_2.plotcosts, turn_3.plotcosts, straight_3.plotcosts))
    thrusts = np.concatenate((turn_1.plotthrusts, straight_1.plotthrusts, turn_2.plotthrusts, straight_2.plotthrusts, turn_3.plotthrusts, straight_3.plotthrusts))
    ax.plot(times, costs, 'b', linewidth=1.5)
    if detailed:
        ax2 = ax.twinx()
        ax2.plot(times, thrusts, 'g', alpha=0.2)
        ax2.set_ylim(ymin=0, ymax=1.05 * max(thrusts))
        ax2.set_ylabel('Webb-adjusted thrust (cm/s)', fontsize=smallfontsize)
    ax.set_xlabel('Time (s)', fontsize=smallfontsize)
    ax.set_ylabel('Energy cost (J/s)', fontsize=smallfontsize)
    ax.set_ylim(ymin=0, ymax=1.05 * max(costs))
    ax.set_xlim(xmin=-0.02 * abs(max(straight_3_times) - min(turn_1_times)), xmax=max(straight_3_times))
    ax.text(-0.235, 0.95, 'd', transform=ax.transAxes, size=bigfontsize, weight='bold')
    if detailed:
        ax.set_title('Cost ({0:.5f} J + {1:.5f} J from SMR)'.format(solution.dynamics.energy_cost, SMR * solution.dynamics.duration), fontsize=smallfontsize)

def plot_parameter_sensitivity(opt, display=True, export_path=None):
    opt_proportions = opt.proportions()
    plt.ioff()
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    def replace_element(proportions, index, new_value):
        new_proportions = proportions.copy()
        new_proportions[index] = new_value
        return new_proportions
    for i, ax in enumerate(axes.reshape(-1)):
        if i < len(opt_proportions):
            p = opt_proportions[i]
            x = np.unique(np.clip(np.linspace(p - 0.05, p + 0.05, 301), 0, 1)) # go +/- 5 % from optimal param proportional value, but stopping at 0 or 1; always choose odd #
            y = [-maneuver_from_proportions(opt.fish, opt.prey_velocity, opt.det_x, opt.det_y, replace_element(opt_proportions, i, x_value)).fitness for x_value in x]
            final_turn_x_adjustments = [x_value for x_value in x if maneuver_from_proportions(opt.fish, opt.prey_velocity, opt.det_x, opt.det_y, replace_element(opt_proportions, i, x_value)).had_final_turn_adjusted]
            sns.lineplot(x=x, y=y, ax=ax)
            sns.rugplot(final_turn_x_adjustments, ax=ax)
            ax.set_ylabel("Maneuver cost (J)")
            ax.set_xlabel("Proportional " + param_labels[i])
            default_ymin, default_ymax = ax.get_ylim()
            ax.set_ylim(ymin=max(-0.9*opt.fitness, default_ymin), ymax=min(-2.0*opt.fitness, default_ymax))
            ax.axvline(x=p, ls='dotted', color='0.0', label='Global Optimum')
    fig.suptitle("Proportional parameters (dotted line = optimum, orange ticks = had to slow down)")
    fig.tight_layout()
    if export_path is not None:
        fig.savefig(export_path)
    if display:
        fig.show()
    else:
        plt.close(fig)

def print_dynamics(dynamicstuple, dynamics, solution):
    (turn_1, straight_1, turn_2, straight_2, turn_3, straight_3) = dynamicstuple

    def print_segment_dynamics(label, dyn):
        print("Dynamics of segment {0:17s} of length {1:6.2f}: thrust={2:5.3f}, duration={3:5.3f}, final_speed={4:7.3f},      cost: {5:11.7f}.".format(label, dyn.length, dyn.u_thrust, dyn.duration, dyn.final_speed, dyn.cost))

    print("----------------------------------------------------------------------")
    print_segment_dynamics('pursuit turn', turn_1)
    print_segment_dynamics('pursuit straight', straight_1)
    print_segment_dynamics('return turn', turn_2)
    print_segment_dynamics('return straight', straight_2)
    print_segment_dynamics('final turn', turn_3)
    print("Dynamics of segment {0:17s} of length {1:6.2f}: duration_a={2:5.3f}, duration_b={3:5.3f}, thrusts=({5:6.2f}, {6:6.2f}), cost: {4:11.7f}.".format('final straight', straight_3.length, straight_3.duration_a, straight_3.duration_b, straight_3.cost, straight_3.thrust_a, straight_3.thrust_b))
    print("----------------------------------------------------------------------")
    energy_cost_with_SMR = solution.fish.maneuver_energy_cost_including_SMR(solution)
    total_cost_with_SMR = solution.fish.maneuver_energy_cost_including_SMR(solution) + dynamics.opportunity_cost
    print("Energy cost = {0:7.4f} J ({1:7.4f} J without SMR, {2:7.4f} J w/opportunity cost of {3:7.4f} J).".format(energy_cost_with_SMR, dynamics.energy_cost, total_cost_with_SMR, dynamics.opportunity_cost))
    print("Total duration = {0:8.6f} s (wait time {1:5.3f} s, pursuit {2:.3f} s). Traveled distance {3:6.1f} cm at mean speed {4:4.1f} cm/s".format(dynamics.duration, dynamics.wait_duration, dynamics.pursuit_duration, solution.path.total_length, dynamics.mean_speed))

