import floats

if __name__ == "__main__":
    f = floats.Floats()
    issId = f.load_environment("models/iss/iss_composed.urdf")  
    robotId, arm = f.load_manipulator("definitions/soft_robot.yaml", [0, 0, 2])
    
    gp = floats.GraspPlanner(f, arm)
    
    mission_object = "camera"
    startPos = [0, 0, 0]
    startRot = [0, 0, 0]
    grasp_force = 300
    grasp_duration = 2000

    objectId = f.load_object(mission_object, startPos, startRot, useFixedBase=False)
    gp.plan_grasps(objectId, object_name=mission_object, grasp_force=grasp_force, grasp_duration=grasp_duration)

    f.disconnect()
