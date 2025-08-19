# FLOATS: Flexible Light-weight On-orbit Actuator Simulation

A simulation framework for free-flying soft robots in microgravity.

---

## About the Project
FLOATS facilitates realistic simulation of locomotion, manipulation, and planning for deformable on-orbit systems.
We demonstrate grasping, free-floating object reconfiguration, and mobile manipulation tasks as a baseline for research on soft robotics planning and control in microgravity.

![Hero Image](./images/hero_image.png)

## Features
- **PyBullet Simulation Environment**:
  - Loads ISS environment and mission objects.
- **Manipulator Control**:
  - Load and configure a soft continuum manipulator.
  - Apply torque control and set contact properties.
- **Path Planning Algorithms**:
  - RRT
  - RRT*
  - RRT-Connect
  - A* search 
- **Grasping Strategies**:
  - Apply hard-coded trajectories for inhand-manipulation, 
  - Execute torque-based grasps with dynamic adjustments.


## Getting Started

  ### Test inhand manipulation 
  You can perform inhand manipulation by grasping a Cargo Transder Box and rotating it by 180 or 360 degrees. Therefor, we use a trajectory defined in "trajectory_inhand_manipulation", that you can       adapt to also manipulate different objects like cameras, rover wheels, sample tubes, grease guns etc..

  | ![Inhand Manipulation - First Orientation](./assets/inhand_manipulation_1.png)   | ![Inhand Manipulation - Second Orientation](./assets/inhand_manipulation_2.png)   |
  
  ```bash
  python test_inhand_manipulation.py
  ```

  ### Test grasping 
  ```bash
  python test_grasping.py
  ```

  ### Test path planning 
  ```bash
  python test_planning.py
  ```


### Prerequisites
```bash
SOMO: https://github.com/GrauleM/somo.git

Ubuntu 16.04 and Ubuntu 18.04 with Python 3.6.9
Ubuntu 20.04 with Python 3.6.9, 3.7.9 and 3.8.2
Windows 10 with Python 3.7 through Anaconda

```
