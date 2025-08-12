import os
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import copy
import sorotraj
from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.sm_continuum_manipulator import SMContinuumManipulator

modules = {
    "kibo_elm_ps": [0, 100, 40],
    "kibo": [0, 63, 0],
    "harmony": [0, 0, 0],
    "columbus": [0, -52, 0],
    "destiny": [-85, 0, 0],
    "unity": [-151, 0, 0],
    "quest": [-151, -55, 0],
    "tranquility": [-151, 65, 0],
    "beam": [-188, 65, 0],
    "leonardo": [-90, 65, 0],
    "cupola": [-151, 65, -24]
}

connections = {
    "unity": ["destiny", "quest", "tranquility"],
    "destiny": ["unity", "harmony"],
    "quest": ["unity"],
    "tranquility": ["unity", "cupola", "beam", "leonardo"],
    "harmony": ["destiny", "columbus", "kibo"],
    "columbus": ["harmony"],
    "kibo": ["harmony", "kibo_elm_ps"],
    "kibo_elm_ps": ["kibo"],
    "cupola": ["tranquility"],
    "beam": ["tranquility"],
    "leonardo": ["tranquility"]
}

urdf_files = {
    "camera": "../models/camera.urdf",
    "box": "../models/cargo_transfer_box_new_scale.urdf",
    "box_inhand": "../models/cargo_transfer_box.urdf",
    "crew_bag": "../models/crew_bag.urdf",
    "grease_gun": "../models/grease_gun.urdf",
    "rover_wheel": "../models/rover_wheel.urdf",
    "sample_tube": "../models/sample_tube.urdf",
    "ctb": "../models/ctb.urdf",
    "sphere": "../models/sphere.urdf",
    "astronaut": "../models/astronaut/astronaut.urdf"
}


class Floats:

    def __init__(self):
        self.modules = modules 
        self.connections = connections
        self.urdf_files = urdf_files
        opt_str = "--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0"
        cam_width, cam_height = 1920, 1640
        if cam_width is not None and cam_height is not None:
            opt_str += " --width=%d --height=%d" % (cam_width, cam_height)

        self.physicsClient = p.connect(p.GUI, options=opt_str)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=6.5, cameraYaw=30.0, cameraPitch=-30.0, cameraTargetPosition=[0, 0, 0])
        p.setGravity(0, 0, 0)
        p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setRealTimeSimulation(0)

    def set_timestep(self, time_step):
        p.setTimeStep(time_step)


    def load_environment(self, filename):
        """
        Loads the ISS model into the PyBullet simulation environment.
        :return: The PyBullet body ID of the loaded ISS model
        """
        print("filename", filename)
        issId = p.loadURDF(filename, useFixedBase=True, globalScaling=1)
        p.changeDynamics(issId, -1, lateralFriction=1)
        return issId

    def load_object(self, object_name, start_pos=[0, 0, 0], start_rot=[0, 0, 0], useFixedBase=False, inhand=False):
        """
        Loads a URDF object into the PyBullet simulation.
        :param objectStartPos: Starting position as [x, y, z]
        :param objectStartOr: Starting orientation as a quaternion 
        :param path: File path to the URDF model
        :param useFixedBase: Whether the object should be immovable (default: False)
        :return: The PyBullet body ID of the loaded object
        """
        path = urdf_files[object_name]

        objectId = p.loadURDF(
            path,
            start_pos,
            p.getQuaternionFromEuler(start_rot),
            useFixedBase=useFixedBase,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
            globalScaling=1,
        )
        if not useFixedBase:
            p.changeDynamics(objectId, -1, lateralFriction=1000, mass=0.1)
        if inhand: 
            p.changeDynamics(objectId, -1, lateralFriction=1, mass=15.0)
        return objectId

    def load_manipulator(self, arm_config, start_pos=[0, 0, 0], start_rot=[0, 0, 0]):
        """
        Loads a soft continuum manipulator into the PyBullet simulation.
        :param start_pos: Starting position as [x, y, z]
        :param start_rot: Starting orientation as a quaternion
        :return: Tuple containing the PyBullet body ID of the manipulator and the SMContinuumManipulator instance
        """
        arm_manipulator_def = SMManipulatorDefinition.from_file(arm_config)
        arm = SMContinuumManipulator(arm_manipulator_def)
        arm.load_to_pybullet(
            baseStartPos=start_pos,
            baseStartOrn=p.getQuaternionFromEuler(start_rot),
            baseConstraint="free",
            physicsClient=self.physicsClient,
        )
        robotId = arm.bodyUniqueId
        contact_properties = {"lateralFriction": 1000}
        arm.set_contact_property(contact_properties)
        return robotId, arm


    def load_trajectory(self, trajectory_file):
        """
        Loads a trajectory definition from file and returns an actuation function.
        :param trajectory_file: Path to the trajectory definition file
        :return: Interpolation function mapping time to actuation commands
        """
        traj = sorotraj.TrajBuilder(graph=False)
        traj.load_traj_def(trajectory_file)
        trajectory = traj.get_trajectory()
        interp = sorotraj.Interpolator(trajectory)
        actuation_fn = interp.get_interp_function(
            num_reps=1, speed_factor=1.2, invert_direction=False, as_list=False
        )
        return actuation_fn

    def run_trajectory(self, arm, actuation_fn, time_step, n_steps):
        for i in range(n_steps):
            torques = actuation_fn(i * time_step)
            arm.apply_actuation_torques(
                actuator_nrs=[0, 0, 1, 1, 2,2],
                axis_nrs=[0, 1, 0, 1, 0, 1],
                actuation_torques=torques.tolist()
            )
            p.stepSimulation()

    
    def reset_robot(self, robotId, robot_pos, robot_rot, joint_positions=None):
        """
        Resets the robot's base position, orientation, velocities, and optionally its joint states.
        :param robotId: PyBullet body ID of the robot
        :param robot_pos: Base position as [x, y, z]
        :param robot_rot: Base orientation as a quaternion [x, y, z, w]
        :param joint_positions: Optional list of joint positions; if None, joints reset to 0.0
        """
        p.resetBasePositionAndOrientation(robotId, robot_pos, robot_rot)
        p.resetBaseVelocity(robotId, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
        num_joints = p.getNumJoints(robotId)
        for i in range(num_joints):
            pos = 0.0 if (joint_positions is None or i >= len(joint_positions)) else joint_positions[i]
            p.resetJointState(robotId, i, targetValue=pos, targetVelocity=0.0)

    def save_camera_image(self, object_name, direction, distance, output_folder="screenshots"):
        """
        Captures and saves a screenshot from the PyBullet simulation using matplotlib.
        :param object_name: Name of the object
        :param direction: String describing the camera viewing direction
        :param distance: Camera distance from the object
        :param output_folder: Directory where the screenshot will be saved (default: "screenshots")
        """
        os.makedirs(output_folder, exist_ok=True)

        # Get RGBA image from PyBullet
        width, height, rgba, _, _ = p.getCameraImage(
            width=640,
            height=480,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )[:5]

        # Convert to NumPy array and normalize
        img = np.reshape(rgba, (height, width, 4)).astype(np.uint8)

        # Create filename
        filename = f"{object_name}_{direction}_{distance:.2f}m.png"
        filepath = os.path.join(output_folder, filename)

        # Save image using matplotlib (remove axes and borders)
        plt.imsave(filepath, img)
        print(f"Saved screenshot to {filepath}")

    def disconnect(self):
        p.disconnect()

    def get_robot_parameters(self): 

        robot_parameters = {
            "act_1_len": 0.7,
            "act_2_len": 0.3,
            "act_3_len": 0.7,
            "link_height": 0.07,
            "link_len": 0.2,
            "gripper_len": 0.3,
            "center_link_idx": 14,
            "total_link_number": 29
        }

        return robot_parameters
