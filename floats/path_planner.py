import time
import random
import math
import copy
import heapq
from scipy.interpolate import interp1d
from typing import Callable, List, Optional
import pybullet as p
import numpy as np



class Node:

    def __init__(self, point: np.ndarray, parent: Optional["Node"]=None):
        self.point = np.asarray(point, dtype=float)
        self.parent = parent
        self.cost = 0.0
        self.g = 0.0
        self.f = 0.0

    def nearest_node(self, nodes: List["Node"]) -> "Node":
        """
        Returns the nearest node to this node from a list of nodes.
        """
        return min(nodes, key=lambda n: np.linalg.norm(n.point - self.point))

class PathPlanner:

    def __init__(self, floats, step_size):
        self.floats = floats
        self.step_size = step_size
        self.sphereId = self.floats.load_object("sphere", [0, 15, 2], [0, 0, 0], useFixedBase=False)

    # -------- Utilities --------
    def get_random_point(self, bounds):
        """
        Random point within the defined search space.
        :param bounds: List of (min, max) tuples for each dimension
        :return: np.array point
        """
        return np.array([random.uniform(bounds[i][0], bounds[i][1])
                        for i in range(len(bounds))])


    def interpolate(self, points, num=200, kind='linear'):
        """
        Interpolates a sequence of points to produce a smoother path.
        :param points: Sequence of points as a (N x D) array-like object
        :param num: Number of interpolated points to generate (default: 200)
        :param kind: Type of interpolation ('linear', 'quadratic', 'cubic')
        :return: Interpolated points as a (num x D) NumPy array
        """
        points = np.asarray(points)
        N, D = points.shape

        t_old = np.linspace(0, 1, N)
        t_new = np.linspace(0, 1, num)

        interpolated = np.vstack([
            interp1d(t_old, points[:, dim], kind=kind)(t_new)
            for dim in range(D)
        ]).T

        return interpolated

    def shortcut_path(self, path, max_iters=100):
        """
        Performs linear shortcut optimization on a given path.
        :param path: List of waypoints
        :param max_iters: Maximum number of attempts to find shortcuts
        :return: Optimized path as a list of points
        """
        if path is None or len(path) < 3:
            return path

        optimized = [np.array(p) for p in path]

        # Try up to max_iters times to shorten the path by randomly selecting two waypoints and replacing the intermediate segment with a direct connection if it is collision-free
        for _ in range(max_iters):
            n = len(optimized)
            if n < 3:
                break

            i = random.randint(0, n - 3)
            j = random.randint(i + 2, n - 1)

            p_i = optimized[i]
            p_j = optimized[j]

            if self.line_collision_check(p_i, p_j):
                optimized = optimized[:i+1] + optimized[j:]

        return [p.tolist() for p in optimized]

    def pop_until_length(self, path, target_length):
        """
        Removes points from the end of `path` until the sum of the removed 
        segment lengths is greater than or equal to `target_length`.
        Returns: (remaining_path, popped_points).
        """
        popped_points: List[List[float]] = []
        popped_length = 0.0

        while len(path) >= 2 and popped_length < target_length:
            last = path.pop()
            prev = path[-1]
            seg_len = math.hypot(last[0] - prev[0], last[1] - prev[1])
            popped_points.append(last)
            popped_length += seg_len

        return path


    def get_bounds(self, start, end):
        """
        Computes axis-aligned search space bounds that enclose the start and end points.
        :param start: Starting position as [x, y, z]
        :param end: Target position as [x, y, z]
        :return: List of [min, max] pairs for each axis [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        """
        space = 10
        bounds = [
            [-1000, 1000], [-1000, 1000], [-1000, 1000]
        ]
        for i in range(3):
            bounds[i][0] = min(start[i], end[i]) - space
            bounds[i][1] = max(start[i], end[i]) + space
        return bounds


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


    def plot_path(self, path, color=(1,0,0), width=3):
        """
        Draws a 3D path in the PyBullet debug view.
        :param path: List of 3D points (each as a list or numpy array)
        """
        if p is None or path is None:
            return
        last = None
        for pt in path:
            if last is not None:
                p.addUserDebugLine(last, pt, color, width, 0)
            last = pt

    def line_collision_check(self, p1, p2, step_size: float = 0.5) -> bool:
        """
        Checks whether the straight-line segment between two points is collision-free.
        If a `line_collision_check` callback was provided, forwards to it.
        Otherwise samples along the line and uses the point-wise `collision_check` callback.
        """

        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        direction = p2 - p1
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            return bool(self.collision_check(p1))

        direction /= length
        steps = int(length / step_size)
        for i in range(steps + 1):
            point = p1 + direction * i * step_size
            if not bool(self.collision_check(point)):
                return False
        return True

    def collision_check(self, position):
        """
        Checks whether a sphere placed at the given position collides with any objects in the simulation.
        :param position: Target position for the sphere
        :return: True if no collision is detected, False otherwise
        """
        p.resetBasePositionAndOrientation(self.sphereId, position, [0, 0, 0, 1])
        p.stepSimulation()
        contacts = p.getContactPoints(bodyA=self.sphereId)
        return not bool(len(contacts))


    def reset_sphere(self):
        p.resetBasePositionAndOrientation(self.sphereId, [50, 50, 50], [0, 0, 0, 1])

    # -------- A* --------
    def astar(self, start, goal, bounds, resolution=1.0):
        """
        A* on a 3D grid.
        :param start: [x, y, z]
        :param goal: [x, y, z]
        :param bounds: [(min, max), ...] for x, y, z
        :param resolution: Grid resolution
        :return: List of [x, y, z] waypoints or None
        """
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        z_min, z_max = bounds[2]
        nx = int(math.floor((x_max - x_min) / resolution))
        ny = int(math.floor((y_max - y_min) / resolution))
        nz = int(math.floor((z_max - z_min) / resolution))

        def to_idx(pt):
            return (
                int(round((pt[0] - x_min) / resolution)),
                int(round((pt[1] - y_min) / resolution)),
                int(round((pt[2] - z_min) / resolution)),
            )
        def to_point(idx):
            return np.array([
                x_min + idx[0] * resolution,
                y_min + idx[1] * resolution,
                z_min + idx[2] * resolution,
            ])

        start_idx = to_idx(start)
        goal_idx  = to_idx(goal)

        # check whether there is a collision with start or goal 
        if not self.collision_check(start, self.sphereId) or not self.collision_check(goal, self.sphereId):
            return None

        open_set = []
        # Initialize the A* open set with the start node
        start_node = Node(to_point(start_idx))
        # Set the cost from start (g) to 0
        start_node.g = 0.0
        # Estimate the total cost (f) using the Euclidean distance to the goal
        start_node.f = np.linalg.norm(start_node.point - np.array(goal))
        # Push the start node into the priority queue (min-heap) for exploration
        heapq.heappush(open_set, (start_node.f, start_idx, start_node))

        g_score = {start_idx: 0.0}
        closed = set()

        neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

        # Process nodes from the open set until empty: retrieve the lowest-cost node, skip if already visited, return the reconstructed path if the goal is reached.
        while open_set:
            _, current_idx, current_node = heapq.heappop(open_set)
            if current_idx in closed:
                continue
            if current_idx == goal_idx:
                path = []
                n = current_node
                while n:
                    path.append(n.point.tolist())
                    n = n.parent
                return path[::-1]

            closed.add(current_idx)

            # For each neighboring cell: skip if out of bounds or blocked by collision, compute tentative cost from the start, update and enqueue the neighbor if this path is better than any previously found.
            for d in neighbors:
                nbr_idx = (current_idx[0] + d[0],
                        current_idx[1] + d[1],
                        current_idx[2] + d[2])
                if not (0 <= nbr_idx[0] <= nx and
                        0 <= nbr_idx[1] <= ny and
                        0 <= nbr_idx[2] <= nz):
                    continue

                nbr_pt = to_point(nbr_idx)
                if not self.line_collision_check(current_node.point, nbr_pt):
                    continue

                tentative_g = g_score[current_idx] + resolution

                if tentative_g < g_score.get(nbr_idx, float('inf')):
                    g_score[nbr_idx] = tentative_g
                    h = np.linalg.norm(nbr_pt - np.array(goal))
                    nbr_node = Node(nbr_pt, current_node)
                    nbr_node.g = tentative_g
                    nbr_node.f = tentative_g + h
                    heapq.heappush(open_set, (nbr_node.f, nbr_idx, nbr_node))

        return None


    # -------- RRT helpers --------
    def steer(self, from_pt: np.ndarray, to_pt: np.ndarray):
        """
        Moves from a starting point toward a target point by at most a fixed step size.
        :param from_pt: Starting point 
        :param to_pt: Target point
        :return: Array representing the new point
        """
        vec = to_pt - from_pt
        dist = float(np.linalg.norm(vec))
        if dist <= self.step_size:
            return to_pt.copy()
        return from_pt + vec/dist * self.step_size

    def _nearest(self, nodes, pt):
        return min(nodes, key=lambda n: np.linalg.norm(n.point - pt))


    # -------- RRT --------
    def rrt(self, start, goal, bounds, max_iter=10000, goal_sample_rate=0.1):
        """
        Rapidly-exploring Random Tree (RRT) path planning algorithm.
        """
        start_node = Node(np.array(start, dtype=float))
        goal_node  = Node(np.array(goal, dtype=float))
        tree = [start_node]
        for _ in range(max_iter):
            rnd = goal_node.point if random.random() < goal_sample_rate else self.get_random_point(bounds)
            nearest = self._nearest(tree, rnd)
            direction = rnd - nearest.point
            dist = float(np.linalg.norm(direction))
            if dist == 0:
                continue
            new_pt = nearest.point + direction/dist * min(self.step_size, dist)
            if not self.line_collision_check(nearest.point, new_pt):
                continue
            new_node = Node(new_pt, nearest)
            tree.append(new_node)
            if np.linalg.norm(new_node.point - goal_node.point) < self.step_size:
                if self.line_collision_check(new_node.point, goal_node.point):
                    goal_node.parent = new_node
                    path = []
                    n = goal_node
                    while n.parent is not None:
                        path.append(n.point.tolist())
                        n = n.parent
                    path.append(start_node.point.tolist())
                    return path[::-1]
        return None

    # -------- RRT* --------
    def rrt_star(self, start, goal, bounds, max_iter=10000, goal_sample_rate=0.1, neighbor_radius=2.0):
        """
        RRT* with rewiring for 3D planning.
        """
        start_node = Node(np.array(start, dtype=float)); start_node.cost = 0.0
        goal_node  = Node(np.array(goal, dtype=float));  goal_node.cost  = float('inf')
        tree = [start_node]
        for _ in range(max_iter):
            rnd = goal_node.point if random.random() < goal_sample_rate else self.get_random_point(bounds)
            nearest = self._nearest(tree, rnd)
            direction = rnd - nearest.point
            dist = float(np.linalg.norm(direction))
            if dist == 0: 
                continue
            new_pt = nearest.point + direction/dist * min(self.step_size, dist)
            if not self.line_collision_check(nearest.point, new_pt):
                continue
            neighbors = [n for n in tree if np.linalg.norm(n.point - new_pt) <= neighbor_radius]
            best_parent = nearest
            best_cost   = nearest.cost + float(np.linalg.norm(nearest.point - new_pt))
            for n in neighbors:
                if self.line_collision_check(n.point, new_pt):
                    c = n.cost + float(np.linalg.norm(n.point - new_pt))
                    if c < best_cost:
                        best_cost   = c
                        best_parent = n
            new_node = Node(new_pt, best_parent); new_node.cost = best_cost
            tree.append(new_node)
            if np.linalg.norm(new_node.point - goal_node.point) < self.step_size and self.line_collision_check(new_node.point, goal_node.point):
                goal_node.parent = new_node
                path = []
                n = goal_node
                while n:
                    path.append(n.point.tolist()); n = n.parent
                return path[::-1]
            for n in neighbors:
                c_through_new = new_node.cost + float(np.linalg.norm(n.point - new_node.point))
                if c_through_new < n.cost and self.line_collision_check(new_node.point, n.point):
                    n.parent = new_node; n.cost = c_through_new
        candidates = []
        for n in tree:
            if np.linalg.norm(n.point - goal_node.point) < self.step_size and self.line_collision_check(n.point, goal_node.point):
                total_cost = n.cost + float(np.linalg.norm(n.point - goal_node.point))
                candidates.append((total_cost, n))
        if not candidates:
            return None
        _, best = min(candidates, key=lambda x: x[0])
        goal_node.parent = best
        path = []
        n = goal_node
        while n:
            path.append(n.point.tolist()); n = n.parent
        return path[::-1]

    # -------- RRT-Connect --------
    def rrt_connect(self, start, goal, bounds, max_iter=10000, goal_sample_rate=0.1):
        """
        Bidirectional RRT-Connect for 3D planning.
        """
        start_node = Node(np.array(start, dtype=float))
        goal_node  = Node(np.array(goal, dtype=float))
        tree_start = [start_node]
        tree_goal  = [goal_node]
        tree_grow, tree_connect = tree_start, tree_goal

        for _ in range(max_iter):
            rnd = goal_node.point if random.random() < goal_sample_rate else self.get_random_point(bounds)
            nearest = self._nearest(tree_grow, rnd)
            new_pt  = self.steer(nearest.point, rnd)
            if not self.line_collision_check(nearest.point, new_pt):
                continue
            new_node = Node(new_pt, nearest)
            tree_grow.append(new_node)

            while True:
                nearest_c = self._nearest(tree_connect, new_node.point)
                new_pt_c  = self.steer(nearest_c.point, new_node.point)
                if not self.line_collision_check(nearest_c.point, new_pt_c):
                    break
                new_node_c = Node(new_pt_c, nearest_c)
                tree_connect.append(new_node_c)
                if np.allclose(new_node_c.point, new_node.point):
                    if tree_grow is tree_start:
                        conn_s, conn_g = new_node, new_node_c
                    else:
                        conn_s, conn_g = new_node_c, new_node
                    path_s = []; n = conn_s
                    while n: path_s.append(n.point.tolist()); n = n.parent
                    path_s.reverse()
                    path_g = []; n = conn_g
                    while n: path_g.append(n.point.tolist()); n = n.parent
                    return path_s + path_g[1:]
            tree_grow, tree_connect = tree_connect, tree_grow
        return None


    def compute_path(self, waypoints):

        global_path = []
        timings = []
        print("waypoints", waypoints)

        for i in range(len(waypoints) - 1):
            start = self.floats.modules[waypoints[i]]
            goal  = self.floats.modules[waypoints[i + 1]]
            print("start", start); print("goal", goal)
            bounds = self.get_bounds(start, goal)
            print("bounds", bounds)
            t0 = time.perf_counter()
            points = self.rrt_connect(start, goal, bounds, max_iter=10000, goal_sample_rate=0.1)
            t1 = time.perf_counter()
            print(f"Planning took {t1 - t0:.4f} seconds")
            timings.append(t1 - t0)
            global_path.extend(points)

        print("Planning took in total:", sum(timings))
        total_length = self.compute_path_length(global_path, M=np.identity(3))
        print(f"The path is {total_length} long.")

        robot_parameters = self.floats.get_robot_parameters()

        target = copy.deepcopy(self.floats.modules[waypoints[-1]])

    
        global_path = self.pop_until_length(global_path, 2.0)
        global_path = self.shortcut_path(global_path, max_iters=200)
        global_path = self.interpolate(global_path, num=200).tolist()

        self.plot_path(global_path)

        self.reset_sphere()

        p.stepSimulation()

        return global_path

