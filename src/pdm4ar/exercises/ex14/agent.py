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
from scipy.interpolate import splprep, splev


class GlobalPlanMessage(BaseModel):
    trajectories: Dict[str, NDArray]  # for each robot assign a trajectory (as an array)
    goals: Dict[str, NDArray]


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
        def adaptive_linear_oversampling(raw_traj, base=5, max_points=20):
            raw = np.array(raw_traj)
            out = []

            for i in range(1, len(raw) - 1):
                # take more points where i have more difficult angles
                p_prev = raw[i - 1][:2]
                p = raw[i][:2]
                p_next = raw[i + 1][:2]

                v1 = p - p_prev
                v2 = p_next - p

                cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cosang = np.clip(cosang, -1, 1)
                ang = np.arccos(cosang)

                curvature = (np.pi - ang) / np.pi

                n_points = int(base + curvature * (max_points - base))

                seg = np.linspace(raw[i - 1], raw[i], n_points, endpoint=False)
                for j in range(1, len(seg)):
                    out.append(seg[j])

            out.append(raw[-1])
            return np.array(out)

        raw_trajectory = global_plan.trajectories[str(self.name)]
        if len(raw_trajectory) >= 2:
            self.trajectory = adaptive_linear_oversampling(raw_trajectory)
        else:
            self.trajectory = []
        self.current_traj_idx = 0  # stores the pure pursuit point
        self.current_goal_idx = 0
        self.goals = global_plan.goals[str(self.name)]
        self.target_goal = False
        self.inplace_rotation = False
        self.last_alpha = 0  # for calculationg the d term
        self.backwards = False

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:

        if self.trajectory is None or len(self.trajectory) == 0:
            return DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        LOOKAHEAD_CRUISE = 0.15
        KP_ROT = 4.0
        KD_ROT = 0.3
        last_point = self.trajectory[-1]

        R = self.sg.wheelradius
        L = self.sg.wheelbase
        w_min, w_max = self.sp.omega_limits

        state = sim_obs.players[self.name].state
        x = state.x
        y = state.y
        psi = state.psi

        dist_to_last = math.hypot(last_point[0] - x, last_point[1] - y)

        if dist_to_last < 0.1 and self.current_traj_idx > len(self.trajectory) - 3:
            return DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        # pick lookahead target
        target_point = None
        for i in range(self.current_traj_idx, len(self.trajectory)):
            px, py = self.trajectory[i][0], self.trajectory[i][1]
            if math.hypot(px - x, py - y) >= LOOKAHEAD_CRUISE:
                target_point = (px, py)
                self.current_traj_idx = i
                break
        if target_point is None:
            target_point = self.trajectory[-1]

        tx, ty = target_point[0], target_point[1]
        Ld = math.hypot(tx - x, ty - y)

        # compute heading error
        alpha_fwd = math.atan2(ty - y, tx - x) - psi
        alpha_fwd = (alpha_fwd + math.pi) % (2 * math.pi) - math.pi

        # decide forward/backward
        if abs(alpha_fwd) > math.pi / 2:
            self.backwards = True
        else:
            self.backwards = False

        # recompute alpha in backward mode (use psi + pi)
        if self.backwards:
            alpha = math.atan2(ty - y, tx - x) - (psi + math.pi)
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
            v_cmd = -(w_max * R)
            # PD rotation in backward mode

            w_cmd = -(2 * v_cmd * math.sin(alpha)) / Ld

        else:
            alpha = alpha_fwd
            v_cmd = w_max * R
            w_cmd = (2 * v_cmd * math.sin(alpha)) / Ld

        # compute wheel speeds
        omega_r = (v_cmd + (w_cmd * L / 2.0)) / R
        omega_l = (v_cmd - (w_cmd * L / 2.0)) / R

        # saturate
        lim = max(abs(w_min), abs(w_max))
        scale = max(abs(omega_r), abs(omega_l))
        if scale > lim:
            k = lim / scale
            omega_r *= k
            omega_l *= k

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

        # precompute goal to nearest drop off
        goal_drop_cost = {}
        goal_drop_path = {}
        goal_clusters = {d: [] for d in drop_grid.keys()}
        goal_drop_id = {}

        for gid, gpos in goal_grid.items():
            best_cost = float("inf")
            best_path = None
            best_drop_id = None

            for drop_id, dpos in drop_grid.items():
                p = self.astar(gpos, dpos)
                if p is None:
                    continue
                c = self.path_cost(p)
                if c < best_cost:
                    best_cost = c
                    best_path = p
                    best_drop_id = drop_id

            goal_drop_cost[gid] = best_cost
            goal_drop_path[gid] = best_path
            goal_drop_id[gid] = best_drop_id
            if best_drop_id is not None and best_path is not None:
                goal_clusters[best_drop_id].append(gid)
            else:
                # unreachable goal
                goal_drop_cost[gid] = float("inf")
                goal_drop_path[gid] = None
                goal_drop_id[gid] = None

        robots_sorted = sorted(robot_grid.keys())
        drops_sorted = sorted(drop_grid.keys())
        goals_sorted = sorted(goal_grid.keys())

        # apply hungarian to assign to each drop-off a single robot, ie the one closest the dropoff
        num_r = len(robots_sorted)
        num_d = len(drops_sorted)
        cost_matrix = np.full((num_r, num_d), np.inf)
        paths_rd: dict[tuple[str, int], list[tuple[int, int]]] = {}

        for i, r in enumerate(robots_sorted):
            for j, d in enumerate(drops_sorted):
                start = robot_grid[r]
                dpos = drop_grid[d]
                path_rd = self.astar(start, dpos)
                if path_rd is None:
                    continue
                c = self.path_cost(path_rd)
                cost_matrix[i, j] = c
                paths_rd[(r, d)] = path_rd

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
        drop_to_robot: dict[int, str] = {}
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
        robot_goals: dict[str, list[int]] = {r: [] for r in robots_sorted}
        for r in robots_sorted:
            for d in robot_to_drops[r]:
                robot_goals[r].extend(goal_clusters[d])

        robot_target_goals = {r: [] for r in robots_sorted}
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
                best_path_rg = None
                best_cost = float("inf")

                for gid in list(goals_for_r):
                    drop_id = goal_drop_id.get(gid)
                    if drop_id is None:
                        continue
                    if goal_drop_path[gid] is None:
                        continue
                    gpos = goal_grid[gid]
                    path_rg = self.astar(robot_current_pos[r], gpos)
                    if path_rg is None:
                        continue

                    cost_rg = self.path_cost(path_rg)
                    if cost_rg < best_cost:
                        best_cost = cost_rg
                        best_gid = gid
                        best_path_rg = path_rg
                if best_gid is None:
                    # no reachable remaining goal for this robot
                    break

                if not robot_paths[r]:
                    # if it's the first path, take teh full trajectory
                    robot_paths[r].extend(best_path_rg)
                else:
                    # if it's not the first path, don't take the first cell (drop-off) to avoid duplicates
                    robot_paths[r].extend(best_path_rg[1:])
                g2d = goal_drop_path[best_gid]
                robot_paths[r].extend(g2d[1:])
                robot_current_pos[r] = g2d[-1]
                gx, gy = centre_of_poly(goals[best_gid].polygon)
                robot_target_goals[r].append([gx, gy])
                goals_for_r.remove(best_gid)

        # old clustering
        """ robot_clusters = {}
        for d in drops_sorted:
            best_r = None
            best_cost = float("inf")
            for r in robots_sorted:
                path_rd = self.astar(robot_grid[r], drop_grid[d])
                if path_rd is None:
                    continue
                cost_rd = self.path_cost(path_rd)
                if cost_rd < best_cost:
                    best_cost = cost_rd
                    best_r = r
            if best_r is not None:
                robot_clusters[d] = (
                    best_r  ######## must enforce that a single robot can't be assigned to multiple drop offs
                )
        #### but then, the end, if there

        # for each cluster, compute the optimal path for the robot
        robot_paths = {r: [] for r in robots_sorted}
        robot_current_pos = {}
        for d in drops_sorted:
            r = robot_clusters[d]
            robot_current_pos[r] = robot_grid[r]

            cluster_goals = goal_clusters[d]
            while cluster_goals:
                best_path = None
                best_cost = float("inf")
                best_goal = None

                for gid in cluster_goals:
                    gpos = goal_grid[gid]
                    path_rg = self.astar(robot_current_pos[r], gpos)
                    if path_rg is None or goal_drop_path[gid] is None:
                        continue
                    cost_rg = self.path_cost(path_rg)
                    if cost_rg < best_cost:
                        best_cost = cost_rg
                        best_path = path_rg
                        best_goal = gid
                if best_goal is None:
                    break

                robot_paths[r].extend(best_path)
                robot_paths[r].extend(goal_drop_path[best_goal][1:])
                robot_current_pos[r] = goal_drop_path[best_goal][-1]
                # remove goal just assigned to the cluster
                cluster_goals.remove(best_goal)"""

        # no clustering
        """# build cost matrix for robots to goals
        robots_sorted = sorted(robot_grid.keys())
        goals_sorted = sorted(goal_grid.keys())
        num_r = len(robots_sorted)
        num_g = len(goals_sorted)

        cost_matrix = np.full((num_r, num_g), np.inf)
        paths_rg = {}

        for i, r in enumerate(robots_sorted):
            for j, g in enumerate(goals_sorted):
                # robot to goal
                path_rg = self.astar(robot_grid[r], goal_grid[g])
                if path_rg is None:
                    continue
                # total cost robot to goal + goal to dropoff
                if goal_drop_path[g] is None:
                    continue

                cost_matrix[i, j] = self.path_cost(path_rg)  # + goal_drop_cost[g]
                paths_rg[(r, g)] = path_rg

        # use optimizer for first assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # initialize assignment structures
        assignments = {r: [] for r in robots_sorted}
        robot_paths = {r: [] for r in robots_sorted}
        robot_current_pos = {}
        robot_finish_time = {}
        assigned_goals = set()

        for i, j in zip(row_ind, col_ind):
            r = robots_sorted[i]
            g = goals_sorted[j]

            if not np.isfinite(cost_matrix[i, j]):
                continue

            assigned_goals.add(g)

            # robot to goal
            path_rg = paths_rg[(r, g)]
            robot_paths[r].extend(path_rg)

            # goal to dropoff
            g2d = goal_drop_path[g]
            robot_paths[r].extend(g2d[1:])

            # update robot
            robot_current_pos[r] = g2d[-1]
            robot_finish_time[r] = cost_matrix[i, j]
            assignments[r].append(g)

        # remaining goals
        remaining_goals = [g for g in goals_sorted if g not in assigned_goals]

        while remaining_goals:
            # pick the robot that becomes available first
            r = min(robot_finish_time, key=lambda x: robot_finish_time[x])

            # find best goal for this robot
            best_g = None
            best_cost = float("inf")
            best_path_rg = None

            for g in remaining_goals:
                # skip unreachable goals
                if goal_drop_path[g] is None:
                    continue

                path_rg = self.astar(robot_current_pos[r], goal_grid[g])
                if path_rg is None:
                    continue

                total_cost = self.path_cost(path_rg) + goal_drop_cost[g]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_g = g
                    best_path_rg = path_rg

            if best_g is None:
                # if robot cannot reach any remaining goal
                break

            # assign this goal
            assignments[r].append(best_g)
            remaining_goals.remove(best_g)

            # append robot to goal path
            robot_paths[r].extend(best_path_rg[1:])

            # append goal to dropoff path
            g2d = goal_drop_path[best_g]
            robot_paths[r].extend(g2d[1:])

            # update robot current position
            robot_current_pos[r] = g2d[-1]

            # update robot finish time
            robot_finish_time[r] += best_cost

        # we have to avoid robots stopping at the drop off locations.
        # Once they finish their path, we must send them somewhere else.
        parking_radius = 3.0
        max_trials = 100
        radius_cells = max(1, int(parking_radius / self.res))
        ny, nx = self.grid.shape
        robot_rad = self.robot_radius
        drop_radius = 1.0
        min_drop_dist = drop_radius + robot_rad
        # sort robot in order of finish time
        robots_by_finish = sorted(robot_finish_time.items(), key=lambda kv: kv[1])
        for r, _ in robots_by_finish:
            if r not in robot_current_pos:
                continue
            si, sj = robot_current_pos[r]
            # final dropoff center for this robot
            x_d, y_d = self.grid_to_world(si, sj)

            best_segment = None
            for _ in range(max_trials):
                di = random.randint(-radius_cells, radius_cells)
                dj = random.randint(-radius_cells, radius_cells)
                ni = si + di
                nj = sj + dj
                if not (0 <= ni < ny and 0 <= nj < nx):
                    continue
                if self.grid[ni, nj]:
                    continue
                if (ni, nj) == (si, sj):
                    continue

                x_c, y_c = self.grid_to_world(ni, nj)
                if math.hypot(x_c - x_d, y_c - y_d) < min_drop_dist:
                    continue

                segment = self.astar((si, sj), (ni, nj))
                if segment is None:
                    continue
                best_segment = segment
                break

            if best_segment is None:
                continue

            robot_paths[r].extend(best_segment[1:])
            fi, fj = best_segment[-1]
            robot_current_pos[r] = (fi, fj)

            # mark all cells within robot radius around parking cell as occupied
            x_p, y_p = self.grid_to_world(fi, fj)
            disc = ShPoint(x_p, y_p).buffer(robot_rad)  # circular footprint in world coords

            gminx, gminy, gmaxx, gmaxy = disc.bounds
            x_min = self.x_min
            y_min = self.y_min
            res = self.res
            ny, nx = self.grid.shape

            cell_min_x = max(0, int((gminx - x_min) / res))
            cell_max_x = min(nx - 1, int((gmaxx - x_min) / res))
            cell_min_y = max(0, int((gminy - y_min) / res))
            cell_max_y = min(ny - 1, int((gmaxy - y_min) / res))

            if cell_min_x <= cell_max_x and cell_min_y <= cell_max_y:
                xs = x_min + (np.arange(cell_min_x, cell_max_x + 1) + 0.5) * res
                ys = y_min + (np.arange(cell_min_y, cell_max_y + 1) + 0.5) * res
                X, Y = np.meshgrid(xs, ys)

                pts = [ShPoint(x, y) for x, y in zip(X.ravel(), Y.ravel())]
                mask = np.array([disc.covers(p) for p in pts], dtype=bool).reshape(len(ys), len(xs))

                self.grid[cell_min_y : cell_max_y + 1, cell_min_x : cell_max_x + 1] |= mask"""

        # convert grid paths to world trajectories
        trajectories = {}
        final_goals = {}
        final_drops = {}
        for r in robots_sorted:
            cells = robot_paths[r]
            trajectories[r] = self.cells_to_waypoints(cells)
            if len(robot_target_goals[r]) > 0:
                final_goals[r] = np.array(robot_target_goals[r], dtype=float)
            else:
                final_goals[r] = np.zeros((0, 2), dtype=float)

        # build global plan message
        global_plan_message = GlobalPlanMessage(trajectories=trajectories, goals=final_goals)

        DEBUG = False
        if DEBUG:
            import matplotlib.pyplot as plt

            print("\nDEBUG: Generating debug_map.png\n")

            plt.figure(figsize=(10, 10))
            ax = plt.gca()

            dg = init_sim_obs.dg_scenario

            # 1) Obstacles + boundaries
            for obs in dg.static_obstacles:
                geom = obs.shape if hasattr(obs, "shape") else obs.polygon

                if geom.geom_type == "Polygon":
                    x, y = geom.exterior.xy
                else:
                    # LinearRing or other types
                    coords = list(geom.coords)
                    x, y = zip(*coords)

                plt.fill(x, y, color="lightgray", alpha=0.7, edgecolor="black")

            # 2) Goals (yellow)
            if goals is not None:
                for gid, gobj in goals.items():
                    poly = gobj.polygon

                    if poly.geom_type == "Polygon":
                        x, y = poly.exterior.xy
                    else:
                        coords = list(poly.coords)
                        x, y = zip(*coords)

                    plt.fill(x, y, color="gold", alpha=0.9, edgecolor="black")
                    cx, cy = poly.centroid.x, poly.centroid.y
                    plt.text(cx, cy, f"G{gid}", color="black", ha="center", va="center", fontsize=10)

            # 3) Dropoff points (green)
            if drops is not None:
                for cp_id, cp in drops.items():
                    poly = cp.polygon

                    if poly.geom_type == "Polygon":
                        x, y = poly.exterior.xy
                    else:
                        coords = list(poly.coords)
                        x, y = zip(*coords)

                    plt.fill(x, y, color="lightgreen", alpha=0.9, edgecolor="black")
                    cx, cy = poly.centroid.x, poly.centroid.y
                    plt.text(cx, cy, f"D{cp_id}", color="black", ha="center", va="center", fontsize=10)

            # 4) Robot starting positions (blue)
            for name, state in robots_states.items():
                plt.plot(state.x, state.y, "bo", markersize=8)
                plt.text(state.x, state.y, name, color="blue", ha="left", va="bottom", fontsize=10)

            # 5) Trajectories (red)
            for name, traj in trajectories.items():
                if len(traj) > 0:
                    xs = traj[:, 0]
                    ys = traj[:, 1]
                    plt.plot(xs, ys, "r-", linewidth=2)
                    plt.plot(xs[0], ys[0], "ro", markersize=5)  # start point

            # 6) Final figure settings
            plt.title("DEBUG MAP — Obstacles, Robots, Goals, Dropoffs, Trajectories")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.grid(True)

            plt.savefig("debug_map.png", dpi=200)
            plt.close()

            print("Saved debug map to debug_map.png\n")

        # but keep in mind that this could be a bottle neck for high number of robots/goals
        # frist sample points, create grid
        #
        # assigning each goal to drop off
        #
        # assign all goals to robots
        #
        # create trajectory, concatenate the path and handle multi goal case
        #
        # include the paths for each message
        #
        # pass the environmental information
        #
        # note for control pupose you should return a trajectory of size (N_points, 3), for each waypoint in the trajectory we should have (x, y, psi)
        # plis stick to the convention (x, y, psi) in a numpy array where the first element is x, the second y, and the 3rd is psi

        return global_plan_message.model_dump_json(round_trip=True)
