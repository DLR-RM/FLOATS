
import copy
import numpy as np
from collections import deque


class MissionPlanner:
    def __init__(self, floats):
        self.floats = floats


    def compute_path_length(self, path, M=np.identity(3)):
        """
        Computes the total length of a path in 3D space, optionally using a metric matrix.
        :param path: List of waypoints, each as [x, y, z]
        :param M: 3x3 metric matrix (default: identity matrix for standard Euclidean distance)
        :return: Total path length as a float
        """
        total_length = 0.0
        for i in range(1, len(path)):
            p1 = np.array(path[i - 1])
            p2 = np.array(path[i])

            delta = p2 - p1
            dist = np.sqrt(delta.T @ M @ delta)
            total_length += dist
        return total_length


    def find_path(self, start, end):
        """
        Finds the shortest path between two nodes in a graph using BFS.
        """
        queue = deque([(start, [start])]); visited = set()
        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            visited.add(current)
            for nb in self.floats.connections.get(current, []):
                if nb not in visited:
                    queue.append((nb, path + [nb]))
        return None


    def plan_mission(self, mission_object, start, end):
        """
        Plans a mission to move from a start module to an end module and place the mission object.
        :param mission_object: Name of the mission object
        :param start: Name of the starting module
        :param end: Name of the target module
        :return: Tuple containing:
                - objectId: PyBullet body ID of the loaded mission object
                - waypoints: List of module names along the optimal path
                - mission_object: The mission object name
        """
        waypoints = self.find_path(start, end)

        waypoint_path = []
        for w in waypoints:
            waypoint_path.append(self.floats.modules.get(w))

        total_length = self.compute_path_length(waypoint_path, M=np.identity(3))
        print("optimal path is",total_length)

        

        return waypoints