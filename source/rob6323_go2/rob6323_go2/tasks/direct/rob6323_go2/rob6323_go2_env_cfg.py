# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4
    state_space = 0
    debug_vis = True

    # Tutorial Part 3: Termination criteria
    base_height_min = 0.20

    # Tutorial Part 2: PD Control Gains
    Kp = 20.0
    Kd = 0.5
    torque_limits = 23.5

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Tutorial Part 2: Disable implicit PD
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # CRITICAL: Disable implicit P
        damping=0.0,  # CRITICAL: Disable implicit D
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # CRITICAL FIX: Restrict contact sensor to ONLY the feet.
    # This ensures indices 0,1,2,3 in the sensor data correspond exactly to the 4 feet.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
    )

    # Set the scale of the visualization markers
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # --------------------------------------------------------
    # REWARD SCALES (Aligned with Professor's Exact Defaults)
    # --------------------------------------------------------
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    # Tutorial Part 1
    action_rate_reward_scale = -0.01

    # Tutorial Part 4
    raibert_heuristic_reward_scale = -10.0

    # Tutorial Part 5 (Refining) -- UPDATE THESE THREE LINES
    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.02  # Changed from -2.0 to -0.02
    dof_vel_reward_scale = -1e-4
    ang_vel_xy_reward_scale = -0.001  # Changed from -0.05 to -0.001

    # Rubric Requirement
    action_l2_reward_scale = -0.0001

    # Tutorial Part 6 -- UPDATE THIS LINE
    feet_clearance_reward_scale = -30.0  # Changed from -10.0 to -30.0
    tracking_contacts_shaped_force_reward_scale = 4.0
