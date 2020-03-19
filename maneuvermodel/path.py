import numpy as np
from numba import int64, float64, jitclass, boolean

def print_path(path): # A non-compiled function to print the characteristics of a path for diagnostic purposes.
    print("Inputs for my Mathematica code to plot this maneuver")
    print("----------------------------------------------------")
    mathematica_string = """{{HxM, HyM}} = {{0, {0}}};
{{KxM, KyM}} = {{{1}, {2}}};
r2 = {3};
dir2 = {4};
{{GxM, GyM}} = {{{5}, 0}};
{{Cx, Cy}} = {{{6}, {7}}};""".format(path.turn_1_center[1], path.turn_3_center[0],path.turn_3_center[1],path.turn_2_radius,path.turn_2_direction,path.focal_return_point[0],path.capture_point[0],path.capture_point[1])
    print(mathematica_string)
    print("----------------------------------------------------")
    print("Turn arc angles are theta1={0}, theta2={1}, theta3={2}.".format(path.theta_1, path.theta_2, path.theta_3))
    print("Turn directions are dir2={0}, dir3={1}.".format(path.turn_2_direction, path.turn_3_direction))
    print("Turn 1 center (point H) is ({0}, {1}).".format(path.turn_1_center[0], path.turn_1_center[1]))
    print("Turn 2 center (point J) is ({0}, {1}).".format(path.turn_2_center[0], path.turn_2_center[1]))
    print("Turn 3 center (point K) is ({0}, {1}).".format(path.turn_3_center[0], path.turn_3_center[1]))
    print("Tangent point B is ({0}, {1}).".format(path.tangent_point_B[0], path.tangent_point_B[1]))
    print("Tangent point D is ({0}, {1}).".format(path.tangent_point_D[0], path.tangent_point_D[1]))
    print("Tangent point E is ({0}, {1}).".format(path.tangent_point_E[0], path.tangent_point_E[1]))
    print("Tangent point F is ({0}, {1}).".format(path.tangent_point_F[0], path.tangent_point_F[1]))

maneuver_path_spec = [
    ('creation_succeeded', boolean),
    ('min_turn_radius', float64),
    ('turn_1_center', float64[:]),
    ('turn_2_center', float64[:]),
    ('turn_3_center', float64[:]),
    ('turn_1_radius', float64),
    ('turn_2_radius', float64),
    ('turn_3_radius', float64),
    ('turn_1_angles', float64[:]),
    ('turn_2_angles', float64[:]),
    ('turn_3_angles', float64[:]),
    ('turn_2_direction', int64),
    ('turn_3_direction', int64),
    ('capture_point', float64[:]),
    ('detection_point', float64[:]),
    ('tangent_point_B',float64[:]),
    ('tangent_point_D', float64[:]),
    ('tangent_point_E', float64[:]),
    ('tangent_point_F', float64[:]),
    ('focal_return_point', float64[:]), # Not set until after dynamics are calculated
    ('turn_1_length', float64),
    ('turn_2_length',float64),
    ('turn_3_length',float64),
    ('wait_length',float64),
    ('straight_1_length',float64),
    ('straight_2_length',float64),
    ('straight_3_length',float64),      # Not set until after dynamics are calculated
    ('total_length',float64),           # Not set until after dynamics are calculated because it requires straight_3_length
    ('theta_1',float64),
    ('theta_2',float64),
    ('theta_3',float64)
]

@jitclass(maneuver_path_spec)
class ManeuverPath(object):
    
    def __init__(self, maneuver):
        ''' This function computes the path the fish follows in water coordinates as given its provided choices of the
            pursuit turn radius (r1), return turn radius (r2), clockwise or counterclockwise direction of the second turn (dir2 = +/- 1), 
            and time taken for the whole maneuver (t), in combination with the fixed capture position (xc, yc) (which is also the 
            position of the particle when pursuit begins, in water coords), and the water velocity. Note that I'm using multiple variables
            with the same value here (e.g., both Kx and r3 are equal to solution.r3) to improve readability with respect to the conceptual
            diagram of the maneuver.'''
        self.creation_succeeded = False # can't raise exceptions in Numba, so __init__ will return with this attribute as False if geometry is bad
        twopi = 2*np.pi # used more than once so might as well save a couple cycles
        Cy = maneuver.det_y
        Cx = maneuver.det_x + maneuver.mean_water_velocity * maneuver.wait_time
        Hx = 0
        Hy = maneuver.r1
        r1 = maneuver.r1
        r2 = maneuver.r2
        r3 = maneuver.r3
        eps = 2.2e-16 # actual minimum float is 2.2204460492503131e-16, found by np.finfo(float).eps, which can't be called in numba
        if Cx == -r1 or Cx == 0: Cx += eps # avoid some discontinuities
        # Calculate the first-turn tangent point (Bx, By) and angle theta1 between that and the focal point
        if Cx < 0:
            Bx = (Cx**2*r1**2 + (Cy - r1)*np.sqrt(Cx**2*(Cx**2 + Cy*(Cy - 2*r1))*r1**2))/(Cx*(Cx**2 + (Cy - r1)**2))
            By = (Cx**2*r1 + Cy*(Cy - r1)*r1 - np.sqrt(Cx**2*r1**2*(Cx**2 + Cy**2 - 2*Cy*r1)))/(Cx**2 + (Cy - r1)**2)
        else:
            Bx = (Cx**2*r1**2 + np.sqrt(Cx**2*(Cx**2 + Cy*(Cy - 2*r1))*r1**2)*(-Cy + r1))/(Cx*(Cx**2 + (Cy - r1)**2))
            By = (Cx**2*r1 + Cy*(Cy - r1)*r1 + np.sqrt(Cx**2*(Cx**2 + Cy*(Cy - 2*r1))*r1**2))/(Cx**2 + (Cy - r1)**2)
        # Set dir2 such that a fish heading back toward the focal line x=0 during the pursuit straight will always turn toward after capture, not away from it
        dir2 = -1 if Cy > By else 1
        # Calculate the arc angle for the first (pursuit) turn
        theta1 = np.arctan2(Bx, r1 - By)
        while theta1 > 0: theta1 -= twopi
        # Calculate the center of the second turn, (Jx, Jy)
        m = Bx / (Hy - By) # the slope of the first straight, ie the negative inverse of the slope of the radial lines between the turn centers and tangent points for straight 1
        sign1 = 1 if (Cx < -r1) or (-r1 < Cx and Cx < r1 and Cy < r1) else -1 # corrects for the slope of m1 when deciding which side of the tangent point circle 2 is on
        Jx = Cx - (dir2*m*r2**2*sign1)/np.sqrt((1 + m**2)*r2**2)
        Jy = Cy + (dir2*r2**2*sign1)/np.sqrt((1 + m**2)*r2**2)
        dir3 = -1 if (dir2 == 1 and Jy - r2 < 0) else 1
        Kx = maneuver.final_turn_x
        Ky = dir3 * maneuver.r3
        distJtoK = np.sqrt((Jx - Kx)**2 + (Jy - Ky)**2)
        # Validate the geometry now that point J is known, along with its distance to point K, and return with creation_succeeded = False if circles 2 and 3 aren't compatible
        if dir2 == -1: # counterclockwise -- circles 2 and 3 cannot overlap at all
            if r2 + r3 > distJtoK: return
        else: # circle 2 is clockwise
            if dir3 == -1:
                if r2 + r3 > distJtoK: return # circle 3 is counterclockwise -- circles 2 and 3 cannot overlap at all 
            else:
                if r3 - r2 > distJtoK: return # circle 3 is clockwise -- it can overlap with but not completely contain circle 2
        # Calculate the tangent points between the second and third circle
        psi = np.arctan2(Jy - Ky, Jx - Kx)
        distD2toE2 = np.sqrt(distJtoK**2 - (r3 - r2)**2)
        if dir2*dir3 == -1: # inner tangent points D1, E1 in Mathematica
            omega = np.arccos((r2 + r3) / distJtoK) - dir3 * psi
            Dx = Jx - r2 * np.cos(omega)
            Dy = Jy + dir3 * r2 * np.sin(omega)
            Ex = Kx + r3 * np.cos(omega)
            Ey = Ky - dir3 * r3 * np.sin(omega)
        else: # outer tangent points D2, E2 in Mathematica
            omega = np.arctan2(distD2toE2, r3 - r2) - dir3 * psi
            Dx = Jx + r2 * np.cos(omega)
            Dy = Jy - dir3 * r2 * np.sin(omega)
            Ex = Kx + r3 * np.cos(omega)
            Ey = Ky - dir3 * r3 * np.sin(omega)
        # Calculate the arc angle for the second (return) turn
        phi = np.arctan2(Cy - Jy, Cx - Jx)
        if phi < 0: phi = phi + twopi
        if dir2 == 1:
            if dir3 == 1:
                theta2 = -twopi - omega - phi
            else:
                theta2 = -np.pi + omega - phi
            while theta2 < -twopi: theta2 += twopi
        else:
            theta2 = np.pi - omega - phi
            while theta2 < 0: theta2 += twopi
        # Calculate arc angle for the third (final) turn
        theta3 = 0.5 * np.pi - omega
        while theta3 < 0: theta3 += twopi
        # Now store all the relevant results in well-named variables    
        self.min_turn_radius = maneuver.fish.min_turn_radius # used for plotting or otherwise displaying turn radii relative to the fish's minimum
        self.turn_1_center = np.array([Hx, Hy])
        self.turn_2_center = np.array([Jx, Jy])
        self.turn_3_center = np.array([Kx, Ky])
        self.turn_1_radius = r1
        self.turn_2_radius = r2
        self.turn_3_radius = r3
        self.turn_1_angles = np.array([1.5 * np.pi, 1.5 * np.pi + theta1])
        self.turn_2_angles = np.array([phi, phi + theta2])
        self.turn_3_angles = np.array([-dir3 * omega, -dir3 * (omega + theta3)])
        self.turn_2_direction = dir2
        self.turn_3_direction = dir3
        self.capture_point = np.array([Cx, Cy])
        self.detection_point = np.array([maneuver.det_x, maneuver.det_y])
        self.tangent_point_B = np.array([Bx, By])
        self.tangent_point_D = np.array([Dx, Dy])
        self.tangent_point_E = np.array([Ex, Ey])
        self.tangent_point_F = np.array([maneuver.final_turn_x, 0])
        self.focal_return_point = np.array([np.nan, 0])                # Undefined until after dynamics are calculated
        self.wait_length = abs(maneuver.det_x - Cx)                    # distance particle "moved" / distance fish swam vs water while waiting
        self.turn_1_length = abs(theta1 * r1)
        self.turn_2_length = abs(theta2 * r2)
        self.turn_3_length = abs(theta3 * r3)
        self.straight_1_length = np.sqrt((Cx-Bx)**2 + (Cy-By)**2)
        self.straight_2_length = np.sqrt((Ex-Dx)**2 + (Ey-Dy)**2)
        self.straight_3_length = np.nan                                # Undefined until after dynamics are calculated
        self.total_length = np.nan                                     # Undefined until after dynamics are calculated
        self.theta_1 = theta1
        self.theta_2 = theta2
        self.theta_3 = theta3
        self.creation_succeeded = True

    def update_with_straight_3_length(self, straight_3_length):
        self.straight_3_length = straight_3_length
        self.total_length = self.turn_1_length + self.turn_2_length + self.turn_3_length + self.straight_1_length + self.straight_2_length + straight_3_length
        self.focal_return_point = np.array([self.tangent_point_F[0] - straight_3_length, 0])
        