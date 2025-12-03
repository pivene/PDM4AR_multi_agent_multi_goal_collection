from math import isclose
import random
from dataclasses import dataclass
from turtle import position
from typing import Mapping, Sequence

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


class GlobalPlanMessage(BaseModel):
    # TODO: modify/add here the fields you need to send your global plan
    fake_id: int
    fake_name: str
    trajectory: NDArray  # If you need to send numpy arrays, annotate them with NDArray


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
    point: int #point indicates which point of the trajectory we are chasing
    previous_state: #need to understand the type of this!!!!!!
    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()

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
        # TODO: process here the received global plan
        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)
        # This method receives the string returned by the global planner’s send_plan(...) method.
        # You can deserialize it here and store the information for use during execution.
        # here i have to define global parameters to access than during the whole simulation
        # example

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: DiffDriveState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        kp_linear = 2.0       
        kp_angular = 4.0      
        kd_angular = 0.1
        dist_tolerance = 0.05 
        ang_tolerance = 0.05 
        R = self.sg.wheelradius #radius of wheels 
        L = self.sg.wheelbase #distance between wheels
        #constant terms for the PD control
        w_min, w_max = self.sp.omega_limits
        t = sim_obs.time
        current_state = sim_obs.players[self.name].state
        # extract the state, it should work even if it doesnt seem
        x = current_state.x
        y = current_state.y
        psi = current_state.psi
        if t < 1e-8:
            #initialize the previus state as current state at the initial timestep, we could initialize to zero but i don't know how to do it
            self.previous_state = current_state
            self.point = 0 #initial point is the first on the list
            return DiffDriveCommands(omega_l=0, omega_r=0)
        else:
            dt = t - self.previous_time
            self.previous_time = t

        #now i have to make the robot follow the trajectory 
        if (self.point == len(self.trajectory)):
            return DiffDriveCommands(omega_l=0, omega_r=0) #stop the robot if we arrived to the last point
        
        target = self.trajectory[self.point] #(x_target, y_target, psi_target)
        x_goal = target[0]
        y_goal = target[1]
        psi_goal = target[2]

        dx = x_goal - x
        dy = y_goal - y
        distance = math.sqrt(dx**2 + dy**2) #distance from the target
        heading_to_point = math.atan2(dy, dx) #check if i am heading to the point
        normalize = lambda angle: math.atan2(math.sin(angle), math.cos(angle)) #clamp angles between [-pi, pi]
        alpha = normalize(heading_to_point - psi) #normailzed error to check if i am heading to the goal --> this could be not relevant, it is if we d
        beta = normalize(psi_goal - psi) #normailzed error to check if i am heading to the target angle
        #implement a state machine to control 
        # case 1 : heading error > 0 --> means have to head in the right direction
        if distance > dist_tolerance: #if i am not in the point
            if abs(alpha) > ang_tolerance: #it might be that i am not aligned to it
                v_cmd = 0.0
                relevant_angular_error = alpha # and so I have to first align to it
            else:
                v_cmd = kp_linear * distance #or I might want to move on the straight line to get to the point
                relevant_angular_error = alpha 
        else: #i only need to get the right angular position
            if abs(beta) > ang_tolerance: #in this case i need to rotate to reach the right angular position
                v_cmd = 0.0
                relevant_angular_error = beta
            else:
                # Waypoint Reached! Move to next point
                self.point += 1
                # If we have more points, this logic will pick up next step
                # For this step, just stop to be safe
                
                
        error_derivative = (relevant_angular_error - self.previous_state[2]) / dt if dt > 0 else 0. #(x, y, psi)
        w_cmd = (kp_angular * relevant_angular_error) + (kd_angular * error_derivative)
        self.previous_state[2] = relevant_angular_error

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

    def __init__(self):
        pass

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        # TODO: implement here your global planning stack.
        # create a grid representing the world.
        global_plan_message = GlobalPlanMessage(
            fake_id=1,
            fake_name="agent_1",
            trajectory=np.array([[1, 2, 3], [4, 5, 6]]),
        )
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
        #note for control pupose you should return a trajectory of size (N_points, 3), for each waypoint in the trajectory we should have (x, y, psi)
        #plis stick to the convention (x, y, psi) in a numpy array where the first element is x, the second y, and the 3rd is psi

        return global_plan_message.model_dump_json(round_trip=True)
