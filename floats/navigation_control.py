import os
import json
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

class NavigationControl:

    def __init__(self, floats, k_p=0.05, k_i=0.01, k_d=0.02):
        self.floats = floats
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self._integral_error = 0.0
        self._prev_error = 0.0

    def get_distance_vector(self, robotId, center_link_idx, target_pos):
        """
        Calculates the distance vector from the manipulator's specified link to a target position.
        :param robotId: PyBullet body ID of the manipulator
        :param center_link_idx: Index of the link used for position tracking
        :param target_pos: Target position as [x, y, z]
        :return: Tuple (axis_idx, distance) where:
                - axis_idx: Index of the axis with the largest displacement (0=x, 1=y, 2=z)
                - distance: NumPy array representing the distance vector to the target
        """
        link_state = p.getLinkState(robotId, center_link_idx, computeLinkVelocity=True)
        base_pos = np.array(link_state[4])
        distance = np.array(target_pos) - np.array(base_pos)
        axis_idx = int(np.argmax(np.abs(distance)))
        return axis_idx, distance

    def move(self, robotId, center_link_idx, F_max, distance, dt=1./240.):
        """
        Applies a PID-controlled external force to move the manipulator's center toward a target.

        :param F_max: Maximum allowable force magnitude
        :param distance: Distance vector from the current position to the target
        :param axis_idx: Index of the primary axis for movement (0=x, 1=y, 2=z)
        :param vel: Current linear velocity of the controlled link 
        :param rot_vel: Current angular velocity of the controlled link
        :param base_pos: Current base position of the manipulator
        :param center_link_idx: Index of the end-effector link to which the force is applied
        :param dt: Simulation time step (default: 1/240)
        :return: The computed force vector applied to the link
        """
        I_max = F_max / self.k_i if self.k_i != 0 else F_max
        error = float(np.linalg.norm(distance))
        self._integral_error += error * dt
        self._integral_error = float(np.clip(self._integral_error, -I_max, I_max))
        derivative = (error - self._prev_error) / dt
        self._prev_error = error
        F = self.k_p * error + self.k_i * self._integral_error - self.k_d * derivative
        F = float(np.clip(F, -F_max, F_max))
        direction = distance / error if error > 1e-6 else np.zeros_like(distance)
        force_vec = F * direction
        link_state = p.getLinkState(robotId, center_link_idx)
        link_pos   = np.array(link_state[4])
        p.applyExternalForce(robotId, center_link_idx, force_vec.tolist(), link_pos.tolist(), p.WORLD_FRAME)
        return force_vec

    def follow_path(self, robotId, center_link_idx, points, input_F=1):
        """
        Moves the manipulator along a sequence of target waypoints.
        :param robotId: PyBullet body ID of the manipulator
        :param center_link_idx: Index of the link used for position tracking and control
        :param points: Sequence of target waypoints
        :param arm: The SMContinuumManipulator instance
        :param input_F: Force magnitude applied for movement
        """
        time_step = 0.001
        p.setTimeStep(time_step)
        for point in np.array(points):
            reached = False
            target_pos = point
            while not reached:
                axis_idx, distance = self.get_distance_vector(robotId, center_link_idx, target_pos)
                link_state = p.getLinkState(robotId, center_link_idx, computeLinkVelocity=True)
                lin_vel = np.array(link_state[6]); rot_vel = np.array(link_state[7])
                if np.linalg.norm(distance) >= 0.1:
                    if (np.abs(lin_vel) <= 0.2).any():
                        self.move(robotId, center_link_idx, input_F, distance)
                elif (np.linalg.norm(distance) < 0.1 and np.linalg.norm(distance) >= 0.01):
                    lin_percent=85; rot_percent=50
                    p.resetBaseVelocity(
                        objectUniqueId=robotId,
                        linearVelocity=(lin_vel/100*lin_percent).tolist(),
                        angularVelocity=(rot_vel/100*rot_percent).tolist(),
                    )
                    if (np.abs(lin_vel) < 0.01).all():
                        reached = True
                p.stepSimulation()

    def compute_point_before_grasing(self, global_path, best_grasp):
        """
        Determines an intermediate approach point before the grasping position.

        :param global_path: List of waypoints representing the robot's global path
        :param best_grasp: Dictionary containing grasp information with a "position" key [x, y, z]
        :return: List [x, y, z] representing the point before grasping
        """
        pos = global_path[-1]

        best_grasp_pos = best_grasp["position"]

        total_length = 0.0
        path = [pos, best_grasp_pos]
        for i in range(1, len(path)):
            p1 = np.array(path[i - 1])
            p2 = np.array(path[i])
            delta_x = np.linalg.norm(p2[0] - p1[0])
            delta_y = np.linalg.norm(p2[1] - p1[1])
            delta_z = np.linalg.norm(p2[2] - p1[2])

        dist = {"X": delta_x, "Y": delta_y, "Z": delta_z}

        direction, distanz = max(dist.items(), key=lambda kv: kv[1])

        if direction == "X": 
            point_before_grasping = [pos[0], best_grasp_pos[1], best_grasp_pos[2]]

        elif direction == "Y":
            point_before_grasping = [best_grasp_pos[0], pos[1], best_grasp_pos[2]]

        elif direction == "Z":
            point_before_grasping = [best_grasp_pos[0], best_grasp_pos[1], pos[2]]
        
        return point_before_grasping


    

    def get_approach_pose(self, robotId, current_object_pos, grasp_folder, train_obj_pos):
        """
        Selects the best stored grasp, adjusts it to the current object position,
        and returns the transformed base pose.
        :param robotId: PyBullet body ID of the manipulator
        :param current_object_pos: Current object position [x, y, z]
        :param grasp_folder: Path to the folder containing stored grasp data
        :return: Dictionary with keys:
                - "position": Transformed grasp position [x, y, z]
                - "orientation": Transformed grasp orientation as a quaternion [x, y, z, w]
                - "filename": Name of the file from which the grasp was loaded
        """
        current_base_pos, current_base_orn = p.getBasePositionAndOrientation(robotId)

        cur_pos = np.array(current_base_pos)
        cur_rot = R.from_quat(current_base_orn)
        cur_obj_pos = np.array(current_object_pos)

        best_score = float("inf")
        best_grasp = None

        for filename in os.listdir(grasp_folder):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(grasp_folder, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            try:
                train_base_pos = np.array(data["position"])
                train_base_orn = np.array(data["orientation"])
            except KeyError:
                print(f"Warning: File {filename} contains incomplete data – skipped.")
                continue

            # Calculate object displacement
            obj_delta = cur_obj_pos - train_obj_pos

            # New transformed base_position
            new_base_pos = train_base_pos + obj_delta
            new_base_orn = train_base_orn  # bleibt gleich

            # Generate homogeneous transformation matrix for current pose
            T_current = np.eye(4)
            T_current[:3, :3] = cur_rot.as_matrix()
            T_current[:3, 3] = cur_pos

            # Create homogeneous matrix for transformed stored handle
            grasp_rot = R.from_quat(new_base_orn)
            T_grasp = np.eye(4)
            T_grasp[:3, :3] = grasp_rot.as_matrix()
            T_grasp[:3, 3] = new_base_pos

            # Relative transformation error (between the two poses)
            T_error = np.linalg.inv(T_current) @ T_grasp

            # Extract position difference
            pos_error = T_error[:3, 3]
            pos_dist = np.linalg.norm(pos_error)

            # Extract orientation errors (as angles in radians)
            rot_error = R.from_matrix(T_error[:3, :3])
            rot_angle = rot_error.magnitude()

            score = rot_angle

            if score < best_score:
                best_score = score
                best_grasp = {
                    "position": new_base_pos.tolist(),
                    "orientation": new_base_orn,
                    "name": filename
                }

        if best_grasp is None:
            raise ValueError(f"No valid grasps found in '{grasp_folder}'.")

        return best_grasp

        