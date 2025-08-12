import math
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p

class GraspPlanner:

    def __init__(self, floats, arm):
        self.floats = floats
        self.arm = arm
        self.robotId = arm.bodyUniqueId


    def relative_pose(self, box_pos, box_R, cyl_pos, cyl_R):
        """
        Computes the relative pose of a box with respect to a cylinder.
        :param box_pos: Position of the box [x, y, z]
        :param box_R: Orientation of the box as a quaternion [x, y, z, w]
        :param cyl_pos: Position of the cylinder [x, y, z]
        :param cyl_R: Orientation of the cylinder as a quaternion [x, y, z, w]
        :return: Tuple containing:
                - rel_R: Relative rotation matrix (3x3)
                - rel_t: Relative translation vector (3,)
                - rel_quat: Relative orientation as a quaternion [x, y, z, w]
                - rel_euler: Relative orientation as Euler angles [roll, pitch, yaw] in degrees
        """
        box_pos = np.asarray(box_pos)[:3]
        cyl_pos = np.asarray(cyl_pos)[:3]
        box_R = R.from_quat(np.asarray(box_R).flatten()).as_matrix()
        cyl_R = R.from_quat(np.asarray(cyl_R).flatten()).as_matrix()

        t_world = box_pos - cyl_pos
        rel_R = cyl_R.T @ box_R
        rel_t = cyl_R.T @ t_world
        rel_quat = rel_euler = None

        r = R.from_matrix(rel_R)    
        rel_quat  = r.as_quat()                   
        rel_euler = r.as_euler('xyz', degrees=True)
        return rel_R, rel_t, rel_quat, rel_euler

    def bending_axis_towards_box(self, rel_t, eps=1e-8):
        """
        Computes the bending axis that points towards a box relative to a cylinder.
        :param rel_t: Relative translation vector from cylinder to box [x, y, z]
        :param eps: Small threshold to avoid division by zero when the box lies on the cylinder axis
        :return: Unit vector representing the bending axis
        """
        radial = np.array([rel_t[0], rel_t[1], 0.0])
        norm = np.linalg.norm(radial)
        if norm < eps:
            raise ValueError("Box center lies on the cylinder axis; bending axis undefined.")
        radial_unit = radial / norm
        k = np.array([0.0, 0.0, 1.0])
        axis = np.cross(k, radial_unit)
        axis /= np.linalg.norm(axis)
        return axis

    def bending_axis_components(self, rel_t, eps=1e-8):
        """
        Computes the XY components of the bending axis towards a box relative to a cylinder.
        :param rel_t: Relative translation vector from cylinder to box [x, y, z]
        :param eps: Small threshold to avoid division by zero when the box is near the cylinder axis
        :return: Tuple (x_component, y_component) of the bending axis in the XY-plane
        """
        radial = np.array([rel_t[0], rel_t[1], 0.0])
        norm_r = np.linalg.norm(radial)
        if norm_r < eps:
            raise ValueError("Box center is (near) the cylinder axis; bending axis undefined.")
        radial_unit = radial / norm_r
        axis_xy = np.array([-radial_unit[1], radial_unit[0]])
        return axis_xy[0], axis_xy[1]


    def curvature_and_plane_angle(self, a: float, b: float, c: float, y: float):
        """
        Computes the curvature (kappa) and bending plane angle (phi) of a soft robotic segment based on actuator length changes.

        :param a: Length of actuator 0
        :param b: Length of actuator 1
        :param c: Length of actuator 2
        :param y: Diameter of the soft robot
        :return: Tuple (kappa, phi) where:
                - kappa: Curvature magnitude
                - phi: Angle of the bending plane in radians
        """
        kappa = (2.0 / (y * np.sqrt(3))) * np.sqrt((a-b)**2 + (b-c)**2 + (c-a)**2)
        phi = np.arctan2(np.sqrt(3)*(b-c), 2*a - b - c)
        return kappa, phi


    def r_of_s(self, s: float, kappa: float, phi: float):
        """
        Computes the 3D position along a constant-curvature arc segment.
        The formula is based on the constant curvature model often used for continuum
        or soft robotic segments, where curvature (kappa) and bending plane angle (phi)
        remain constant along the segment.

        :param s: Arc length along the segment
        :param kappa: Curvature of the segment
        :param phi: Bending plane angle in radians
        :return: NumPy array [x, y, z] representing the Cartesian coordinates of the point
        """
        return np.array([
            (1/kappa)*(1 - np.cos(kappa*s))*np.cos(phi),
            (1/kappa)*(1 - np.cos(kappa*s))*np.sin(phi),
            (1/kappa)*np.sin(kappa*s)
        ])


    def t_of_s(self, s: float, kappa: float, phi: float):
        """
        Computes the unit tangent vector at a given arc length along a constant-curvature segment.
        The tangent direction is derived from the constant curvature model, where the bending 
        plane angle (phi) and curvature (kappa) remain constant along the segment.

        :param s: Arc length along the segment
        :param kappa: Curvature of the segment
        :param phi: Bending plane angle in radians
        :return: NumPy array [tx, ty, tz] representing the unit tangent vector
        """
        return np.array([
            np.cos(phi)*np.cos(kappa*s),
            np.sin(phi)*np.cos(kappa*s),
            np.sin(kappa*s)
        ])

    def n_of_s(self, s, kappa, phi):
        """
        Computes the unit normal vector at a given arc length along a constant-curvature segment.
        The normal vector is perpendicular to the tangent vector and points toward the center of curvature,
        based on the constant curvature model where curvature (kappa) and bending plane angle (phi) are constant.

        :param s: Arc length along the segment
        :param kappa: Curvature of the segment
        :param phi: Bending plane angle in radians
        :return: NumPy array [nx, ny, nz] representing the unit normal vector
        """
        return np.array([
            -np.cos(phi)*np.sin(kappa*s),
            -np.sin(phi)*np.sin(kappa*s),
            np.cos(kappa*s)
        ])


    def b_of_s(self, phi):
        """
        Computes the binormal vector for a constant-curvature segment.
        For constant curvature, the binormal vector b(s) does not depend on the arc length s.
        It is perpendicular to both the tangent and normal vectors and lies in the plane 
        orthogonal to the bending direction.

        :param phi: Bending plane angle in radians
        :return: NumPy array [bx, by, bz] representing the binormal vector
        """
        return np.array([np.sin(phi), -np.cos(phi), 0.0])

    def compute_end_frame(self, a, b, c, y, x):
        """
        Computes the end position and orientation frame of a constant-curvature segment.
        :param a: Length of actuator 0
        :param b: Length of actuator 1
        :param c: Length of actuator 2
        :param y: Diameter of the soft robot
        :param x: Arc length along the segment
        :return: Tuple (P_end, R_end, kappa, phi) where:
                - P_end: End position vector [x, y, z]
                - R_end: 3x3 rotation matrix of the end frame
                - kappa: Curvature magnitude
                - phi: Bending plane angle in radians
        """
        kappa, phi = self.curvature_and_plane_angle(a, b, c, y)
        P_end  = self.r_of_s(x, kappa, phi)
        t_end  = self.t_of_s(x, kappa, phi)
        n_end  = self.n_of_s(x, kappa, phi)
        b_end  = self.b_of_s(phi)
        R_end  = np.column_stack([n_end, b_end, t_end])
        return P_end, R_end, kappa, phi


    def compute_grasping_point(self, module_center, a, b, c, x, y, l, center_link_idx, total_link_number):
        """
        Calculates the world coordinates of the grasping point for a given center link.

        :param module_center: World coordinates of the module center [x, y, z]
        :param a: Length of actuator 0
        :param b: Length of actuator 1
        :param c: Length of actuator 2
        :param x: Arc length of the manipulator segment
        :param y: Diameter of the soft robot
        :param l: Desired offset distance from the object along the grasping direction
        :param center_link_idx: Index of the center link in the manipulator
        :param total_link_number: Total number of links in the manipulator
        :return: NumPy array [x, y, z] representing the grasping point in world coordinates
        """
        P_end, R_end, kappa, phi = self.compute_end_frame(a, b, c, y, x)
        s = x * (center_link_idx / total_link_number)
        grasping_point = module_center - l * R_end[:, 2] - (self.r_of_s(x, kappa, phi) - self.r_of_s(s, kappa, phi))
        return grasping_point

    
    
    def grasp(self, robotId, center_link_idx, objectId, tau, time):
        """
        Performs a grasping motion by bending the manipulator towards a target object.
        :param robotId: PyBullet body ID of the manipulator
        :param center_link_idx: Index of the manipulator's central link used as the bending reference
        :param objectId: PyBullet body ID of the object to grasp
        :param tau: Maximum torque value applied during the bending motion
        :param time: Number of simulation steps over which to ramp up the torque
        """
        
        link_state = p.getLinkState(robotId, center_link_idx)
        cyl_R = np.array(link_state[5])
        cyl_pos= np.array(link_state[4])
        box_pos, box_R = p.getBasePositionAndOrientation(objectId)
        # Compute the relative pose of the object with respect to the manipulator's center link
        rel_R, rel_t, rel_quat, rel_euler = self.relative_pose(box_pos, box_R, cyl_pos, cyl_R)
        # Determine the bending axis in 3D space that points towards the object
        axis3d = self.bending_axis_towards_box(rel_t)
        # Compute the X and Y components of the bending axis in the XY-plane
        x_share, y_share = self.bending_axis_components(rel_t)

        for j in range(time):                        
            torques = [x_share*(j/time)*tau,    y_share*(j/time)*tau,  x_share*(j/time)*tau,  y_share*(j/time)*tau, x_share*(j/time)*tau,    y_share*(j/time)*tau]
            self.arm.apply_actuation_torques(
                actuator_nrs=[0, 0, 1, 1, 2, 2],
                axis_nrs=[0, 1, 0, 1, 0, 1],
                actuation_torques=torques
            )
            p.stepSimulation()

        # set damping and friction and stiffness parameters
        for l in range(29): 
            p.changeDynamics(robotId, l, linearDamping=0.5, angularDamping=0.5, contactStiffness=1000, contactDamping=1000)

        p.changeDynamics(objectId, -1, linearDamping=0.5, angularDamping=0.5, contactStiffness=1000, contactDamping=1000)
        
        for l in range(29): 
            p.changeDynamics(robotId, l, rollingFriction=1000)  

        p.changeDynamics(objectId, -1, rollingFriction=1000)  


    def plan_grasps(self, objectId, object_name, grasp_force=300, grasp_duration=2000, center_link_idx=14):
        """
        Plans the grasps for the specified mission object using different approach directions.
        
        :param mission_object: Name of the object to grasp
        :param start_pos: Starting position for the object (default is [0, 0, 0])
        :param start_rot: Starting rotation for the object (default is [0, 0, 0])
        :param grasp_force: Maximum grasp force to apply (default is 300)
        :param grasp_duration: Duration of the grasp (default is 2000 simulation steps)
        """
        approach_directions = [
            ("left",   [-1, 0, 0], False),
            ("right",  [ 1, 0, 0], False),
            ("front",  [ 0, 1, 0], False),
            ("back",   [ 0,-1, 0], False),
            ("top",    [ 0, 0, 1], True),
            ("bottom", [ 0, 0,-1], True),
        ]

        # Load the environment, manipulator, and object
        robotId = self.robotId     
        obj_pos, obj_rot = p.getBasePositionAndOrientation(objectId)

        out_dir = os.path.join("../grasping", object_name + "_tests")
        os.makedirs(out_dir, exist_ok=True)

        # Loop through approach directions
        for direction_name, approach_vector, rotate_x in approach_directions:
            distance = 1.0
            min_distance = 0.2
            step_size = 0.2

            norm = math.sqrt(sum([x**2 for x in approach_vector]))
            approach_vector = [x / norm for x in approach_vector]

            while distance >= min_distance:
                p.resetBasePositionAndOrientation(objectId, obj_pos, obj_rot)

                offset = [x * distance for x in approach_vector]
                robot_pos = [obj_pos[i] + offset[i] for i in range(3)]

                if rotate_x:
                    robot_rot = p.getQuaternionFromEuler([math.pi/2, 0, 0])
                else:
                    robot_rot = p.getQuaternionFromEuler([0, 0, 0])

                # Apply local offset
                local_offset = np.array([0, 0, 0.75])
                rot_matrix = np.array(p.getMatrixFromQuaternion(robot_rot)).reshape(3, 3)
                world_offset = rot_matrix @ local_offset
                robot_pos = [robot_pos[i] - world_offset[i] for i in range(3)]

                self.floats.reset_robot(robotId, robot_pos, robot_rot)

                # Perform the grasp
                self.grasp(robotId, center_link_idx, objectId, grasp_force, grasp_duration)
                p.stepSimulation()

                # Check if grasp was successful
                if self.check_grasp_success(robotId, objectId):
                    center_link_pos = p.getLinkState(robotId, center_link_idx)[4]
                    center_link_rot = p.getLinkState(robotId, center_link_idx)[5]
                    grasp_data = {
                        "position": list(center_link_pos),
                        "orientation": list(center_link_rot),
                        "grasp force": grasp_force,
                        "grasp duration": grasp_duration
                    }
                    filename = f"grasp_data_{direction_name}_at_{distance}.json"
                    with open(os.path.join(out_dir, filename), "w") as f:
                        json.dump(grasp_data, f, indent=4)
                    print(f"Successfully grasped {object_name} from {direction_name} at {distance} m")
                    self.floats.save_camera_image(object_name, direction_name, distance)
                else:
                    print(f"Failed to grasp {object_name} from {direction_name} at {distance} m")

                distance -= step_size



    def check_grasp_success(self, robotId, objectId) -> bool:
        """
        Checks whether the robot is currently in contact with the specified object.
        :param robotId: PyBullet body ID of the robot
        :param objectId: PyBullet body ID of the object
        :return: True if at least one contact point exists, False otherwise
        """
        contacts = p.getContactPoints(bodyA=robotId, bodyB=objectId)
        return len(contacts) > 0
