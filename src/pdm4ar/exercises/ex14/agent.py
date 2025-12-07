from decimal import Decimal
import random
from dataclasses import dataclass
from re import A
from turtle import position
from typing import Mapping, Sequence, List, Dict
import math
from math import inf
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point as ShPoint
import heapq
import dg_commons
import numpy as np
from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations, InitSimObservations, SharedGoalObservation, SimObservations
from dg_commons.sim.agents import Agent, GlobalPlanner
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle
from numpydantic import NDArray
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment


class GlobalPlanMessage(BaseModel):
    trajectories: Dict[str, NDArray]  # for each robot assign a trajectory (as an array)


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: DiffDriveGeometry
    sp: DiffDriveParameters
    trajectory: NDArray  # define the trajectory as a class parameter so we can set it in on_receive_global_plan and see it in get_commands
    point: int  # point indicates which point of the trajectory we are chasing

    # previous_state: #need to understand the type of this!!!!!!
    def __init__(self, res: float = 0.1, robot_radius: float = 0.6):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.res = res
        self.robot_radius = robot_radius

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        # called at the beginning of the simulation
        # in init_sim_obs there are name, seed, boundaries, static obstacles, robot geometry, model parameters
        # estrarre i parametri e assegnarli alle variabili di classe
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.name = init_sim_obs.my_name
        self.prev_ang_err = 0.0

        pass

    def on_receive_global_plan(
        self,
        serialized_msg: str,
    ):
        # TO DO: process here the received global plan
        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)
        # This method receives the dictionary of strings returned by the global planner’s send_plan(...) method.
        # You can deserialize it here and store the information for use during execution.
        # here i have to define global parameters to access than during the whole simulation
        # example

        # save trajectory
        self.trajectory = global_plan.trajectories[str(self.name)]

        # set point counters
        self.point = 0

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: DiffDriveState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        if not hasattr(self, "trajectory") or self.trajectory is None:
            # initial check if a trajectory exist
            return DiffDriveCommands(omega_l=0, omega_r=0)
        dt = 0.1  # input data
        kp_linear = 2.0
        kp_angular = 4.0
        kd_angular = 0.1
        dist_tolerance = 0.05
        ang_tolerance = 0.05
        R = self.sg.wheelradius  # radius of wheels
        L = self.sg.wheelbase  # distance between wheels
        # constant terms for the PD control
        w_min, w_max = self.sp.omega_limits
        t = sim_obs.time
        current_state = sim_obs.players[self.name].state
        # extract the state, it should work even if it doesnt seem
        x = current_state.x
        y = current_state.y
        psi = current_state.psi
        if t < 1e-8:
            # initialize the previus state as current state at the initial timestep, we could initialize to zero but i don't know how to do it
            self.previous_state = current_state
            self.point = 0  # initial point is the first on the list
            return DiffDriveCommands(omega_l=0, omega_r=0)
        else:
            self.previous_time = t

        # now i have to make the robot follow the trajectory
        if self.point == len(self.trajectory):
            return DiffDriveCommands(omega_l=0, omega_r=0)  # stop the robot if we arrived to the last point

        target = self.trajectory[self.point]  # (x_target, y_target, psi_target)
        x_goal = target[0]
        y_goal = target[1]
        psi_goal = target[2]

        dx = x_goal - x
        dy = y_goal - y
        distance = math.sqrt(dx**2 + dy**2)  # distance from the target
        heading_to_point = math.atan2(dy, dx)  # check if i am heading to the point
        normalize = lambda angle: math.atan2(math.sin(angle), math.cos(angle))  # clamp angles between [-pi, pi]
        alpha = normalize(
            heading_to_point - psi
        )  # normailzed error to check if i am heading to the goal --> this could be not relevant, it is if we d
        beta = normalize(psi_goal - psi)  # normailzed error to check if i am heading to the target angle
        # implement a state machine to control
        # case 1 : heading error > 0 --> means have to head in the right direction
        if distance > dist_tolerance:  # if i am not in the point
            if abs(alpha) > ang_tolerance:  # it might be that i am not aligned to it
                v_cmd = 0.0
                relevant_angular_error = alpha  # and so I have to first align to it
            else:
                v_cmd = kp_linear * distance  # or I might want to move on the straight line to get to the point
                relevant_angular_error = alpha
        else:  # i only need to get the right angular position
            if abs(beta) > ang_tolerance:  # in this case i need to rotate to reach the right angular position
                v_cmd = 0.0
                relevant_angular_error = beta
            else:
                # Waypoint Reached! Move to next point
                self.point += 1
                return DiffDriveCommands(omega_l=0, omega_r=0)
                # If we have more points, this logic will pick up next step
                # For this step, just stop to be safe

        # !!!! ERROR HERE WE HAVE TO UNDERSTAND HOW TO ACCESS TO A STATE !!!
        error_derivative = (relevant_angular_error - self.prev_ang_err) / dt if dt > 0 else 0.0  # (x, y, psi)
        self.prev_ang_err = relevant_angular_error
        w_cmd = (kp_angular * relevant_angular_error) + (kd_angular * error_derivative)

        omega_r = (v_cmd + (w_cmd * L / 2)) / R
        omega_l = (v_cmd - (w_cmd * L / 2)) / R
        # Clamp the values in order to don't avoid the constarints
        if omega_r < w_min:
            omega_r = w_min

        if omega_l < w_min:
            omega_l = w_min

        if omega_r > w_max:
            omega_r = w_max

        if omega_l > w_max:
            omega_l = w_max

        return DiffDriveCommands(omega_l=omega_l, omega_r=omega_r)


class Pdm4arGlobalPlanner(GlobalPlanner):
    """
    This is the Global Planner for PDM4AR
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    def __init__(self, res: float = 0.1, robot_radius: float = 0.6):
        self.res = res
        self.robot_radius = robot_radius
        self.grid = None
        self.x_min = None
        self.y_min = None

    # function that builds the occupancy grid
    def get_occupancy_grid(self, init_sim_obs: InitSimGlobalObservations):
        # return: occupancy grid, and origin of the grid in world coords
        dg = init_sim_obs.dg_scenario
        r = self.robot_radius
        res = self.res

        boundary_geom: BaseGeometry | None = None
        shapely_geoms: List[BaseGeometry] = []

        # extract boundaries
        for obs in dg.static_obstacles:
            if isinstance(obs, StaticObstacle):
                geom = obs.shape
            elif isinstance(obs, BaseGeometry):
                geom = obs
            elif hasattr(obs, "polygon"):
                geom = obs.polygon
            else:
                continue

            if geom.geom_type == "LinearRing":
                # boundaries found
                boundary_geom = geom
                continue

            # buffer obstacles
            shapely_geoms.append(geom.buffer(r + 0.1))

        minx, miny, maxx, maxy = boundary_geom.bounds

        # shrink the map for robot radius
        x_min = minx + r
        y_min = miny + r
        x_max = maxx - r
        y_max = maxy - r

        # number of cells of the grid
        nx = int(np.ceil((x_max - x_min) / res))
        ny = int(np.ceil((y_max - y_min) / res))

        grid = np.zeros((ny, nx), dtype=bool)

        # set to false occupied cells
        for geom in shapely_geoms:
            gminx, gminy, gmaxx, gmaxy = geom.bounds

            cell_min_x = max(0, int((gminx - x_min) / res))
            cell_max_x = min(nx - 1, int((gmaxx - x_min) / res))
            cell_min_y = max(0, int((gminy - y_min) / res))
            cell_max_y = min(ny - 1, int((gmaxy - y_min) / res))

            if cell_min_x > cell_max_x or cell_min_y > cell_max_y:
                continue

            xs = x_min + (np.arange(cell_min_x, cell_max_x + 1) + 0.5) * res
            ys = y_min + (np.arange(cell_min_y, cell_max_y + 1) + 0.5) * res
            X, Y = np.meshgrid(xs, ys)

            pts = [ShPoint(x, y) for x, y in zip(X.ravel(), Y.ravel())]
            mask = np.array([geom.covers(p) for p in pts], dtype=bool).reshape(len(ys), len(xs))

            grid[cell_min_y : cell_max_y + 1, cell_min_x : cell_max_x + 1] |= mask

        self.grid = grid
        self.x_min = x_min
        self.y_min = y_min

    # helper function for trajectory smoothing, description inside
    def bresenham_cells(self, i0, j0, i1, j1):
        # function that, given two cells in the grid, returns all the cells encountered ...
        # if connecting the two cells with a straight line
        cells = []
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i1 > i0 else -1
        sj = 1 if j1 > j0 else -1
        i, j = i0, j0
        if dj <= di:
            err = 2 * dj - di
            for _ in range(di + 1):
                cells.append((i, j))
                if err > 0:
                    j += sj
                    err -= 2 * di
                i += si
                err += 2 * dj
        else:
            err = 2 * di - dj
            for _ in range(dj + 1):
                cells.append((i, j))
                if err > 0:
                    i += si
                    err -= 2 * dj
                j += sj
                err += 2 * di
        return cells

    # helper function for trajectory smoothing, description inside
    def line_of_sight(self, a, b):
        # function that returns TRUE if all cells crossed if connecting two cells with ...
        # a straight line (bresenham) are free
        i0, j0 = a
        i1, j1 = b
        grid = self.grid
        for i, j in self.bresenham_cells(i0, j0, i1, j1):
            if grid[i, j]:
                return False
        return True

    # function that actually performs trajectory smoothing
    # (already called inside astar, no need to actively use it)
    def smooth_path(self, path):
        # function that actually perfomrs line smoothing
        if path is None:
            return None
        if len(path) <= 2:
            return path
        n = len(path)
        new = [path[0]]
        i = 0
        while i < n - 1:
            j = n - 1
            # find farthest point reachable
            while j > i + 1 and not self.line_of_sight(path[i], path[j]):
                j -= 1
            new.append(path[j])
            i = j
        return new

    # function that finds the best path from start to goal,
    # and then smooths it to improve it
    def astar(self, start, goal):
        # return the optimal path from start to goal. grid must be an occupancy grid
        # Also performs trajecotry smoothing inside
        grid = self.grid
        ny, nx = grid.shape
        si, sj = start
        gi, gj = goal

        if not (0 <= si < ny and 0 <= sj < nx):
            return None
        if not (0 <= gi < ny and 0 <= gj < nx):
            return None
        if grid[si, sj] or grid[gi, gj]:
            return None

        def heuristic(i, j):
            return ((i - gi) ** 2 + (j - gj) ** 2) ** 0.5

        # neighbor directions
        dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        dirs_diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        def neighbors(i, j):
            for di, dj in dirs4:
                ni = i + di
                nj = j + dj
                if 0 <= ni < ny and 0 <= nj < nx and not grid[ni, nj]:
                    yield ni, nj, 1.0
            for di, dj in dirs_diag:
                ni = i + di
                nj = j + dj
                if not (0 <= ni < ny and 0 <= nj < nx):
                    continue
                if grid[ni, nj]:
                    continue
                # prevent squeezing through corners
                if grid[i + di, j] or grid[i, j + dj]:
                    continue
                yield ni, nj, 2**0.5

        g_cost = np.full((ny, nx), np.inf, dtype=float)
        g_cost[si, sj] = 0.0

        parent = {}
        parent[(si, sj)] = None

        open_heap = []
        heapq.heappush(open_heap, (heuristic(si, sj), (si, sj)))

        in_open = np.zeros((ny, nx), dtype=bool)
        in_open[si, sj] = True

        while open_heap:
            f, (i, j) = heapq.heappop(open_heap)
            in_open[i, j] = False

            if (i, j) == (gi, gj):
                path = []
                cur = (i, j)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                # smooth trajectory
                return self.smooth_path(path)

            for ni, nj, c_step in neighbors(i, j):
                tentative_g = g_cost[i, j] + c_step
                if tentative_g < g_cost[ni, nj]:
                    g_cost[ni, nj] = tentative_g
                    parent[(ni, nj)] = (i, j)
                    f_new = tentative_g + heuristic(ni, nj)
                    if not in_open[ni, nj]:
                        heapq.heappush(open_heap, (f_new, (ni, nj)))
                        in_open[ni, nj] = True

        return None

    # helper to transforms world points into grid points
    def world_to_grid(self, x, y):
        grid = self.grid
        res = self.res
        x_min = self.x_min
        y_min = self.y_min
        ny, nx = grid.shape
        j = int((x - x_min) / res)
        i = int((y - y_min) / res)
        if 0 <= i < ny and 0 <= j < nx:
            return i, j
        return None

    # helper to transforms grid points into world points
    def grid_to_world(self, i, j):
        res = self.res
        x_min = self.x_min
        y_min = self.y_min
        x = x_min + (j + 0.5) * res
        y = y_min + (i + 0.5) * res
        return x, y

    # transforms the path, as a list of gird cells,
    # into an array of [x,y,psi] trajectory waypoints
    def cells_to_waypoints(self, path_cells):
        res = self.res
        x_min = self.x_min
        y_min = self.y_min
        if path_cells is None or len(path_cells) == 0:
            return np.zeros((0, 3), dtype=float)
        n = len(path_cells)
        xy = np.zeros((n, 2), dtype=float)
        for k, (i, j) in enumerate(path_cells):
            x, y = self.grid_to_world(i, j)
            xy[k, 0] = x
            xy[k, 1] = y

        psi = np.zeros(n, dtype=float)
        if n == 1:
            psi[0] = 0.0
        else:
            for k in range(n - 1):
                dx = xy[k + 1, 0] - xy[k, 0]
                dy = xy[k + 1, 1] - xy[k, 1]
                psi[k] = math.atan2(dy, dx)
            psi[-1] = psi[-2]

        traj = np.zeros((n, 3), dtype=float)
        traj[:, 0] = xy[:, 0]
        traj[:, 1] = xy[:, 1]
        traj[:, 2] = psi
        return traj

    def path_cost(self, path):
        """
        Compute geometric cost of a path in grid cells as the
        true Euclidean length in world coordinates.
        """
        if path is None or len(path) < 2:
            return 0.0

        total = 0.0
        res = self.res

        for (i1, j1), (i2, j2) in zip(path[:-1], path[1:]):
            di = i2 - i1
            dj = j2 - j1
            total += math.hypot(di * res, dj * res)

        return total

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        # TO DO: implement here your global planning stack.
        # create a grid representing the world.
        self.get_occupancy_grid(init_sim_obs)

        # extract robots, goals and dropoff points
        robots = init_sim_obs.players_obs
        robots_states = init_sim_obs.initial_states
        goals = init_sim_obs.shared_goals
        drops = init_sim_obs.collection_points

        # helper function to get centre of a shapely polygon
        # we need this because the goals and dropoff points are defined as shapely polygons
        def centre_of_poly(poly):
            c = poly.centroid
            return float(c.x), float(c.y)

        # convert robots to the grid
        # so map each robots name to its grid cell location
        robot_grid = {}
        for name in sorted(robots.keys()):
            state = robots_states[name]
            xr = float(state.x)
            yr = float(state.y)
            g = self.world_to_grid(xr, yr)
            if g is None:
                # robots is out of bounds
                continue
            robot_grid[name] = g

        # convert goals to grid
        # so map each goals ID to its grid cell location
        goal_grid = {}
        if goals is not None:
            for gid, gobj in sorted(goals.items()):
                xg, yg = centre_of_poly(gobj.polygon)
                g = self.world_to_grid(xg, yg)
                if g is None:
                    # goals is out of bounds
                    continue
                goal_grid[gid] = g

        # convert dropoff to grid
        # so map each dropoff ID to its grid cell location
        drop_grid = {}
        if drops is not None:
            for cp_id, cp in sorted(drops.items()):
                xd, yd = centre_of_poly(cp.polygon)
                g = self.world_to_grid(xd, yd)
                if g is not None:
                    drop_grid[cp_id] = g

        # precompute goal to nearest drop off !!!!!!!!!only for clustering!!!!!
        goal_drop_cost = {}
        goal_drop_path = {}
        goal_clusters = {d: [] for d in drop_grid.keys()}
        goal_drop_id = {}

        # precompute goal to all drop offs !!!!!!for better goal-dropoff allocation!!!!
        goal_drop_paths_all = {}
        goal_drop_costs_all = {}

        for gid, gpos in goal_grid.items():
            best_cost = float("inf")
            best_path = None
            best_drop_id = None

            for drop_id, dpos in drop_grid.items():
                p = self.astar(gpos, dpos)
                if p is None:
                    continue
                c = self.path_cost(p)

                goal_drop_paths_all[(gid, drop_id)] = p
                goal_drop_costs_all[(gid, drop_id)] = c
                if c < best_cost:
                    best_cost = c
                    best_path = p
                    best_drop_id = drop_id

            if best_drop_id is not None and best_path is not None:
                goal_drop_cost[gid] = best_cost
                goal_drop_path[gid] = best_path
                goal_drop_id[gid] = best_drop_id
                goal_clusters[best_drop_id].append(gid)
            else:
                # unreachable goal
                goal_drop_cost[gid] = float("inf")
                goal_drop_path[gid] = None
                goal_drop_id[gid] = None

        robots_sorted = sorted(robot_grid.keys())
        drops_sorted = sorted(drop_grid.keys())
        goals_sorted = sorted(goal_grid.keys())

        # apply hungarian to assign to each drop-off a single robot, ie the one closest to the closest goal in the dropoff's cluster
        num_r = len(robots_sorted)
        num_d = len(drops_sorted)
        cost_matrix = np.full((num_r, num_d), np.inf)

        for i, r in enumerate(robots_sorted):
            start = robot_grid[r]
            for j, d in enumerate(drops_sorted):
                goals_in_cluster = goal_clusters[d]

                best_cost = float("inf")

                if goals_in_cluster:
                    for gid in goals_in_cluster:
                        gpos = goal_grid[gid]
                        path_rg = self.astar(start, gpos)
                        if path_rg is None:
                            continue
                        c = self.path_cost(path_rg)
                        if c < best_cost:
                            best_cost = c
                else:
                    # if the dropoff's cluster is empty, we assign a robot to it based on tghe distance to the dropoff itslef
                    dpos = drop_grid[d]
                    path_rd = self.astar(start, dpos)
                    if path_rd is not None:
                        best_cost = self.path_cost(path_rd)

                if best_cost < float("inf"):
                    cost_matrix[i, j] = best_cost

        finite_rows = np.any(np.isfinite(cost_matrix), axis=1)
        finite_cols = np.any(np.isfinite(cost_matrix), axis=0)

        row_ind_base: list[int] = []
        col_ind_base: list[int] = []

        if finite_rows.any() and finite_cols.any():
            row_ids = [i for i, ok in enumerate(finite_rows) if ok]
            col_ids = [j for j, ok in enumerate(finite_cols) if ok]
            cost_sub = cost_matrix[finite_rows][:, finite_cols]
            sub_row_ind, sub_col_ind = linear_sum_assignment(cost_sub)
            row_ind_base = [row_ids[i] for i in sub_row_ind]
            col_ind_base = [col_ids[j] for j in sub_col_ind]

        # actual one-to-one assignment: to each drop-off locations, we assign a single robot
        drop_to_robot = {}
        for i, j in zip(row_ind_base, col_ind_base):
            if not np.isfinite(cost_matrix[i, j]):
                continue
            r = robots_sorted[i]
            d = drops_sorted[j]
            drop_to_robot[d] = r

        # but actually we want all drop-offs to be exploited. Thus, here we assign the
        # drop-offs left to the closest robot
        for j, d in enumerate(drops_sorted):
            # if d in drop_to_robot, that drop-off has already been assigned to a robot
            if d in drop_to_robot:
                continue
            best_i = None
            best_cost = float("inf")
            for i, r in enumerate(robots_sorted):
                c = cost_matrix[i, j]
                if np.isfinite(c) and c < best_cost:
                    best_cost = c
                    best_i = i
            if best_i is not None:
                r = robots_sorted[best_i]
                drop_to_robot[d] = r

        # now we keep account of the fact that to a robot multiple drop-offs can be assigned
        # ans thus we link each robot to all the drop-offs it has been assigned to
        robot_to_drops = {r: [] for r in robots_sorted}
        for d, r in drop_to_robot.items():
            robot_to_drops[r].append(d)

        # now we have to build the plan for each robot, but we want to abandon the drop-off-focused
        # logic: if a robot has multiple dropoffs assigned, it must not care of what drop off it is
        # assigned to, but just optimize its trajectory over all goals that are indirectly assigned to it
        robot_goals = {r: [] for r in robots_sorted}
        for r in robots_sorted:
            for d in robot_to_drops[r]:
                robot_goals[r].extend(goal_clusters[d])

        # here we implement a logic to balance the workload of each robot. Each time we compare two robots: if the number
        # of goals assigned to one is greater than the other +1, the goal of of the overloaded robot which is closest to the other
        # one passes to the latter.
        # Moreover, we also check that the distance of the candidate goal to be moved, from its new drop-off, is not too
        # greater than the of the distance to its past dropoff
        for r1 in robots_sorted:
            goals_for_r1 = robot_goals[r1]
            if not goals_for_r1:
                continue
            for r2 in robots_sorted:
                if r1 == r2:
                    continue
                goals_for_r2 = robot_goals[r2]
                if not goals_for_r2:
                    continue
                # r1 has more goals than r2
                if len(goals_for_r1) > len(goals_for_r2) + 1:
                    closest_goal = None
                    best_cost = float("inf")
                    for goal1 in goals_for_r1:
                        # first of all, compute the distance of the goal to its potential new drop off
                        best_new_drop = float("inf")
                        for d in robot_to_drops[r2]:
                            key = (goal1, d)
                            if key in goal_drop_costs_all:
                                dist_g_d_new = goal_drop_costs_all[key]
                                if dist_g_d_new < best_new_drop:
                                    best_new_drop = dist_g_d_new
                        if best_new_drop == float("inf"):
                            continue
                        # base_dist is the distance between this goal and its closest drop-off, in general
                        base_dist = goal_drop_cost[goal1]
                        # with the following condition, we consider to move only goals whose new dropoff is not too worse from the previous one
                        if best_new_drop > 1.5 * base_dist:
                            continue
                        # here, for each goal of r1, we compute the distance to r2, and the closest one is moved from r1 to r2
                        path = self.astar(robot_grid[r2], goal_grid[goal1])
                        if path is None:
                            continue
                        cost = self.path_cost(path)
                        if cost < best_cost:
                            best_cost = cost
                            closest_goal = goal1
                    if closest_goal is not None:
                        robot_goals[r1].remove(closest_goal)
                        robot_goals[r2].append(closest_goal)
                # r2 has more goals than r1
                elif len(goals_for_r2) > len(goals_for_r1) + 1:
                    closest_goal = None
                    best_cost = float("inf")
                    for goal2 in goals_for_r2:
                        # first of all, compute the distance of the goal to its potential new drop off
                        best_new_drop = float("inf")
                        for d in robot_to_drops[r1]:
                            key = (goal2, d)
                            if key in goal_drop_costs_all:
                                dist_g_d_new = goal_drop_costs_all[key]
                                if dist_g_d_new < best_new_drop:
                                    best_new_drop = dist_g_d_new
                        if best_new_drop == float("inf"):
                            continue
                        # base_dist is the distance between this goal and its closest drop-off, in general
                        base_dist = goal_drop_cost[goal2]
                        # with the following condition, we consider to move only goals whose new dropoff is not too worse from the previous one
                        if best_new_drop > 1.5 * base_dist:
                            continue
                        # here, for each goal of r2, we compute the distance to r1, and the closest one is moved from r2 to r1
                        path = self.astar(robot_grid[r1], goal_grid[goal2])
                        if path is None:
                            continue
                        cost = self.path_cost(path)
                        if cost < best_cost:
                            best_cost = cost
                            closest_goal = goal2
                    if closest_goal is not None:
                        robot_goals[r2].remove(closest_goal)
                        robot_goals[r1].append(closest_goal)

        robot_paths = {r: [] for r in robots_sorted}
        robot_current_pos = {}
        for r in robots_sorted:
            robot_current_pos[r] = robot_grid[r]
            goals_for_r = robot_goals[r]
            # no goals assigned to this robot
            if not goals_for_r:
                continue

            while goals_for_r:
                best_gid = None
                best_drop_for_gid = None
                best_path_rg = None
                best_cost = float("inf")

                remaining_goals_set = set(goals_for_r)
                for gid in list(goals_for_r):
                    gpos = goal_grid[gid]
                    path_rg = self.astar(robot_current_pos[r], gpos)
                    if path_rg is None:
                        continue
                    cost_rg = self.path_cost(path_rg)
                    # from now on we want to select the most convenient drop-off, not the closest one
                    # To this aim, we only consider the drop-offs in the jurisdiction of the robot
                    drops_for_r = robot_to_drops[r]
                    for d in drops_for_r:
                        key = (gid, d)
                        # drop-off not reachable form this goal
                        if key not in goal_drop_paths_all:
                            continue
                        cost_gd = goal_drop_costs_all[key]

                        # here we use this heuristic to evaluate how much bringing teh goal to a dropoff
                        # is convenient, in terms of how far is the closest candidate next goal form it
                        h_future = 0.0
                        other_goals = [g for g in remaining_goals_set if g != gid]
                        # only consider this case if there are multiple dropoffs in the robot jurisdiction
                        if other_goals and len(drops_for_r) > 1:
                            dpos = drop_grid[d]
                            best_future = float("inf")
                            for g_next in other_goals:
                                g_next_pos = goal_grid[g_next]
                                path_dg = self.astar(dpos, g_next_pos)
                                if path_dg is None:
                                    continue
                                cost_dg = self.path_cost(path_dg)
                                if cost_dg < best_future:
                                    best_future = cost_dg
                            h_future = best_future

                        total_cost = cost_rg + cost_gd + h_future

                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_gid = gid
                            best_drop_for_gid = d
                            best_path_rg = path_rg
                if best_gid is None:
                    # no reachable remaining goal for this robot
                    break

                if not robot_paths[r]:
                    # if it's the first path, take teh full trajectory
                    robot_paths[r].extend(best_path_rg)
                else:
                    # if it's not the first path, don't take the first cell (drop-off) to avoid duplicate cells in the traj
                    robot_paths[r].extend(best_path_rg[1:])
                g2d = goal_drop_paths_all[(best_gid, best_drop_for_gid)]
                robot_paths[r].extend(g2d[1:])
                robot_current_pos[r] = g2d[-1]
                goals_for_r.remove(best_gid)

        # convert grid paths to world trajectories
        trajectories = {}
        for r in robots_sorted:
            cells = robot_paths[r]
            trajectories[r] = self.cells_to_waypoints(cells)

        # build global plan message
        global_plan_message = GlobalPlanMessage(trajectories=trajectories)

        # note for control pupose you should return a trajectory of size (N_points, 3), for each waypoint in the trajectory we should have (x, y, psi)
        # plis stick to the convention (x, y, psi) in a numpy array where the first element is x, the second y, and the 3rd is psi

        return global_plan_message.model_dump_json(round_trip=True)
