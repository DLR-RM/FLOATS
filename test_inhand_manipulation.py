import floats

if __name__ == "__main__":
    f = floats.Floats()
    issId = f.load_environment("models/iss/iss_composed.urdf")  
    robotId, arm = f.load_manipulator("definitions/soft_robot_inhand_manipulation.yaml")
    objectId = f.load_object("box_inhand", [-1.5, 0.0, 2.25], inhand=True)

    # Specify time steps
    time_step = 0.001
    f.set_timestep(time_step)
    n_steps = 60000

    actuation_fn = f.load_trajectory("trajectory_inhand_manipulation")
    f.run_trajectory(arm, actuation_fn, time_step, n_steps)

    f.disconnect()
