from decimal import Decimal
import random
from dataclasses import dataclass
from re import A
from turtle import position
from typing import Mapping, Optional, Sequence, List, Dict
import math
from math import inf
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point as ShPoint
import heapq
import dg_commons
import numpy as np
from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations, InitSimObservations, SharedGoalObservation, SimObservations, PlayerObservations
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
        # basic parameters
        self.params = Pdm4arAgentParams()
        self.res = res
        self.robot_radius = robot_radius

        # trajectory following
        self.trajectory = None
        self.point = 0

        # PD controller memory
        self.prev_ang_err = 0.0

        # environment and agent identifiers
        self.priority: Optional[int] = None
        self.static_obstacles: Sequence[StaticObstacle] = []

        # if true the robot is done with his assignment, the robot is in idle and can still move out of the way if necessary
        self.done = False

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        # called at the beginning of the simulation
        # in init_sim_obs there are name, seed, boundaries, static obstacles, robot geometry, model parameters

        # robot geometry and dynmaic parameters
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.name = init_sim_obs.my_name

        # static obstacles
        if init_sim_obs.dg_scenario is not None:
            self.static_obstacles = init_sim_obs.dg_scenario.static_obstacles
        
        # extract number from robot name to assign to priority, the priority is used to decide which robot should move first if they are about to collide
        name_str = str(self.name)
        parts = name_str.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            self.priority = int(parts[-1])
        else:
            self.priority = 999 # assign very low priority if name fails

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

    def is_pose_free(self, x: float, y: float, other_players: Mapping[PlayerName, PlayerObservations], min_robot_dist: float = 0.5) -> bool:
        """
        Check if a robot placed would collide with an obstacle or be too close to another robot.
        """
        # check static obstacles
        disc = ShPoint(x, y).buffer(self.robot_radius)
        for obs in self.static_obstacles:
            geom = getattr(obs, "shape", None)
            if geom is None:
                continue
            if disc.intersects(geom):
                return False

        # check distance to other robots
        for other_name, other_obs in other_players.items():
            if other_name == self.name:
                continue
            ox = other_obs.state.x
            oy = other_obs.state.y
            if math.hypot(ox - x, oy - y) < min_robot_dist:
                return False

        return True

    def choose_avoidance_velocity(self, x: float, y: float, psi: float, sim_obs: SimObservations) -> tuple[float, float]:
        """
        Avoidance behavior for the robot with lower priority or for robot in emergency case (if they are too close).
        Returns v_cmd, w_cmd in robot frame.
        """
        players = sim_obs.players

        # step for probing candidate poses
        step_dist = 0.5

        # speed for the movement
        v_forward = 0.6
        v_backward = -0.8
        w_turn = 1.0

        # point behind the robot to move out of the way (backward pose)
        bx = x - step_dist * math.cos(psi)
        by = y - step_dist * math.sin(psi)

        # try to back up
        if self.is_pose_free(bx, by, players):
            return v_backward, 0.0 

        # left/right sidestep poses
        lx = x - step_dist * math.sin(psi)
        ly = y + step_dist * math.cos(psi)
        rx = x + step_dist * math.sin(psi)
        ry = y - step_dist * math.cos(psi) 

        # try sidestep left
        if self.is_pose_free(lx, ly, players): 
            return v_forward, +w_turn
        
        # try sidestep right
        if self.is_pose_free(rx, ry, players):
            return v_forward, -w_turn

        # if the robot is fully blocked it will rotate in place to search for another way out
        return 0.0, w_turn

    def idle_avoidance_velocity(self, x: float, y: float, psi: float, sim_obs: SimObservations) -> tuple[float, float]:
        """
        Idle behavior for the robot, meaning the robot has finished all his tasks and is waiting.
        If another robot is coming it should move away from it in the opposite direction.
        Returns v_cmd, w_cmd in robot frame.
        """
        players = sim_obs.players

        # find closest robot
        closest = None
        closest_d = float("inf")
        for other_name, other_obs in players.items():
            if other_name == self.name:
                continue
            ox = other_obs.state.x
            oy = other_obs.state.y
            d = math.hypot(ox - x, oy - y)
            if d < closest_d:
                closest_d = d
                closest = other_obs.state

        if closest is None:
            # if nobody is nearby then dont move
            return 0.0, 0.0

        # vector away from the other robot
        dx = x - closest.x
        dy = y - closest.y

        escape_angle = math.atan2(dy, dx)

        # candidate directions:
        primary_dir = escape_angle
        left_dir = escape_angle + math.pi/2
        right_dir = escape_angle - math.pi/2

        # helper: compute (v,w) to move toward a desired global heading
        def control_to_heading(target_angle):
            angle_error = math.atan2(math.sin(target_angle - psi), math.cos(target_angle - psi))
            kp_ang = 2.0
            w = kp_ang * angle_error
            # move only when roughly aligned
            v = 1.0 if abs(angle_error) < 0.01 else 0.0
            return v, w

        # step size to test occupancy
        step = 0.5

        # primary escape route
        ex = x + step * math.cos(primary_dir)
        ey = y + step * math.sin(primary_dir)

        if self.is_pose_free(ex, ey, players):
            return control_to_heading(primary_dir)

        # left escape route
        lx = x + step * math.cos(left_dir)
        ly = y + step * math.sin(left_dir)

        if self.is_pose_free(lx, ly, players):
            return control_to_heading(left_dir)

        # right escape route
        rx = x + step * math.cos(right_dir)
        ry = y + step * math.sin(right_dir)

        if self.is_pose_free(rx, ry, players):
            return control_to_heading(right_dir)

        # if everything is blocked, spin to find a way out
        return 0.0, 1.0

    def vw_to_wheels(self, v, w):
        R = self.sg.wheelradius
        L = self.sg.wheelbase
        w_min, w_max = self.sp.omega_limits

        omega_r = (v + (w * L / 2)) / R
        omega_l = (v - (w * L / 2)) / R

        omega_r = max(min(omega_r, w_max), w_min)
        omega_l = max(min(omega_l, w_max), w_min)

        return DiffDriveCommands(omega_l=omega_l, omega_r=omega_r)

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: DiffDriveState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        # AVOIDANCE: detect robots that are close
        # extract the observation of other robots
        players = sim_obs.players
        # extract the state
        current_state = players[self.name].state
        x = current_state.x
        y = current_state.y
        psi = current_state.psi
        # distance where one robot starts to get out of the way
        danger_dist = 2
        # distance where robot with higher priority should stop to avoid collision
        emergency_dist = 1.3
        speed_factor = 1

        # check if a robot is nearby
        danger_robot_name = None
        min_dist = float("inf")

        for other_name, other_obs in players.items():
            if other_name == self.name:
                continue

            ox = other_obs.state.x
            oy = other_obs.state.y
            d = math.hypot(ox - x, oy - y)

            if d < min_dist:
                min_dist = d
                danger_robot_name = other_name
        
        # if robot is found, decide priority
        if danger_robot_name is not None:
            # determine the priority
            name_str = str(danger_robot_name)
            parts = name_str.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                other_prio = int(parts[-1])
            else:
                other_prio = 9999
            low_prio_robot = (self.priority > other_prio)
            dist = min_dist

            # check if robot is in idle
            if self.done:
                # case 1: both robots are in idle so both have the priority 9999 and they should do nothing
                if self.priority == other_prio:
                    return DiffDriveCommands(omega_l=0, omega_r=0)
                # case 2: robot is in idle but priorities are not equal, in that case move away from the other robot
                if dist < danger_dist:
                    v_avoid, w_avoid = self.idle_avoidance_velocity(x, y, psi, sim_obs)
                    return self.vw_to_wheels(v_avoid, w_avoid)

            # low priority robot tries to avoid the other robot
            if low_prio_robot and (dist < danger_dist):
                v_avoid, w_avoid = self.choose_avoidance_velocity(x, y, psi, sim_obs)
                return self.vw_to_wheels(v_avoid, w_avoid)
            
            # high priority robot only stops if he gets closer than emergency distance
            else:
                if dist < emergency_dist:
                    return self.vw_to_wheels(0.0, 0.0)

                # if not in emergency distance high priority robot slows a bit but keeps going
                speed_factor = 0.5

        if not hasattr(self, "trajectory") or self.trajectory is None:
            # initial check if a trajectory exist
            return DiffDriveCommands(omega_l=0, omega_r=0)
        dt = 0.1  # input data
        kp_linear = 2.0
        kp_angular = 4.0
        kd_angular = 0.1
        dist_tolerance = 0.05
        ang_tolerance = 0.05
        t = sim_obs.time
        
        if t < 1e-8:
            # initialize the previus state as current state at the initial timestep, we could initialize to zero but i don't know how to do it
            self.previous_state = current_state
            self.point = 0  # initial point is the first on the list
            return DiffDriveCommands(omega_l=0, omega_r=0)
        else:
            self.previous_time = t

        # now i have to make the robot follow the trajectory
        if self.point == len(self.trajectory):
            # robot is in idle so it will move out of the way if needed
            self.done = True
            self.priority = 9999
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

        # adjust the speed if necessary
        v_cmd *= speed_factor

        commands = self.vw_to_wheels(v_cmd, w_cmd)
        return commands

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
        ### STEP 1: create a grid representing the world.
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
        drop_grid = []
        if drops is not None:
            for cp_id, cp in sorted(drops.items()):
                xd, yd = centre_of_poly(cp.polygon)
                g = self.world_to_grid(xd, yd)
                if g is not None:
                    drop_grid.append(g)

        ### STEP 2: precompute goal to nearest drop off
        goal_drop_cost = {}
        goal_drop_path = {}

        for gid, gpos in goal_grid.items():
            best_cost = float("inf")
            best_path = None

            for dpos in drop_grid:
                p = self.astar(gpos, dpos)
                if p is None:
                    continue
                c = self.path_cost(p)
                if c < best_cost:
                    best_cost = c
                    best_path = p
            goal_drop_cost[gid] = best_cost
            goal_drop_path[gid] = best_path
            if best_path is None:
                goal_drop_cost[gid] = float("inf")
                goal_drop_path[gid] = None

        # build cost matrix for robots to goals
        robots_sorted = sorted(robot_grid.keys())
        goals_sorted = sorted(goal_grid.keys())
        num_r = len(robots_sorted)
        num_g = len(goals_sorted)

        cost_matrix = np.full((num_r, num_g), np.inf)
        # paths from robot to goal
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

        ### STEP 4: assign the remaining goals to robots
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

                self.grid[cell_min_y : cell_max_y + 1, cell_min_x : cell_max_x + 1] |= mask

        # convert grid paths to world trajectories
        trajectories = {}
        for r in robots_sorted:
            cells = robot_paths[r]
            trajectories[r] = self.cells_to_waypoints(cells)

        # build global plan message
        global_plan_message = GlobalPlanMessage(trajectories=trajectories)

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
