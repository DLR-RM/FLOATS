import floats

if __name__ == "__main__":
    f = floats.Floats()
    pp = floats.PathPlanner(f, step_size=1.0)
    nav = floats.NavigationControl(f)
    mp = floats.MissionPlanner(f)

    issId = f.load_environment("models/iss/iss_composed.urdf")  
    astronautId = f.load_object("astronaut", [0, -40, 0], [1, 0, 0], useFixedBase=True)
    
    robot_parameters = f.get_robot_parameters()
    center_link_idx = robot_parameters["center_link_idx"]

    # change mission_object, mission start and mission goal here
    mission_object = "camera"
    waypoints = mp.plan_mission(mission_object, "harmony", "columbus")

    global_path = pp.compute_path(waypoints)

    target = f.modules[waypoints[-1]]
    objectId = f.load_object(mission_object, target, [0, 0, 0])
    robotId, arm = f.load_manipulator("definitions/soft_robot.yaml", f.modules[waypoints[0]], [0,0,0])
    
    nav.follow_path(robotId, center_link_idx, global_path, input_F=2)
    best_grasp = nav.get_approach_pose(robotId, target, "grasping/" + mission_object + "/", [0, 0, 0])
    pbg = nav.compute_point_before_grasing(global_path, best_grasp)
    end_path = pp.interpolate([global_path[-1], pbg, best_grasp["position"]], num=10, kind='quadratic').tolist()
    pp.plot_path(end_path)
    nav.follow_path(robotId, center_link_idx, end_path, input_F=0.5)

    gp = floats.GraspPlanner(f, arm)
    gp.grasp(robotId, center_link_idx, objectId, 850, 2000)
    
    back_path = pp.interpolate([best_grasp["position"], global_path[-1]], num=10, kind='linear').tolist()
    nav.follow_path(robotId, center_link_idx, back_path, input_F=0.5)
    nav.follow_path(robotId, center_link_idx, global_path[::-1], input_F=0.5)

    f.disconnect()

