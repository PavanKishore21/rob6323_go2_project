# rob6323_go2_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
rob6323_go2_env_cfg.py

Configuration for the ROB6323 Go2 DirectRLEnv environment.

This module defines:
- Environment timing (dt/decimation) and episode length.
- Action/observation/state space sizes and action scaling.
- Termination thresholds (e.g., minimum base height).
- PD controller constants used by the environment's manual controller (Kp/Kd, torque limits).
- Simulation and terrain physics materials (friction/restitution).
- Robot articulation configuration, including disabling implicit PD on leg actuators.
- Scene layout (number of environments, spacing, physics replication).
- Contact sensor configuration restricted to feet for consistent indexing (0..3).
- Debug visualization marker configurations for goal vs. current velocity arrows.
- Reward scale constants used by rob6323_go2_env.py.
"""

from __future__ import annotations

from typing import cast

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp  # Imported for compatibility with typical IsaacLab task templates.
import isaaclab.sim as sim_utils

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    """
    Configuration class for the ROB6323 Go2 locomotion environment.

    Attributes:
        decimation (int): Number of simulation steps per policy step.
        episode_length_s (float): Episode length in seconds.
        action_scale (float): Scale applied to policy actions before mapping to joint targets.
        action_space (int): Size of the action vector.
        observation_space (int): Size of the observation vector.
        state_space (int): Size of the privileged state vector (0 if unused).
        debug_vis (bool): Enable debug visualization markers.

        base_height_min (float): Minimum base height before termination.

        Kp (float): Proportional gain used by the manual PD controller.
        Kd (float): Derivative gain used by the manual PD controller.
        torque_limits (float): Maximum magnitude of torque allowed per joint.

        sim (SimulationCfg): Physics simulation configuration.
        terrain (TerrainImporterCfg): Terrain configuration.
        robot_cfg (ArticulationCfg): Robot articulation configuration.
        scene (InteractiveSceneCfg): Scene (vectorized env) configuration.
        contact_sensor (ContactSensorCfg): Contact sensor configuration.

        goal_vel_visualizer_cfg (VisualizationMarkersCfg): Marker config for commanded velocity.
        current_vel_visualizer_cfg (VisualizationMarkersCfg): Marker config for current velocity.

        lin_vel_reward_scale (float): Reward scale for linear XY velocity tracking.
        yaw_rate_reward_scale (float): Reward scale for yaw-rate tracking.

        action_rate_reward_scale (float): Reward scale for action-rate penalty.
        raibert_heuristic_reward_scale (float): Reward scale for Raibert heuristic term.

        orient_reward_scale (float): Reward scale for base orientation penalty.
        lin_vel_z_reward_scale (float): Reward scale for vertical velocity penalty.
        dof_vel_reward_scale (float): Reward scale for joint velocity penalty.
        ang_vel_xy_reward_scale (float): Reward scale for angular velocity XY penalty.

        action_l2_reward_scale (float): Reward scale for torque L2 penalty.
        feet_clearance_reward_scale (float): Reward scale for feet clearance penalty.
        tracking_contacts_shaped_force_reward_scale (float): Reward scale for shaped force tracking.
    """

    decimation: int = 4
    episode_length_s: float = 20.0

    action_scale: float = 0.25
    action_space: int = 12
    observation_space: int = 48 + 4
    state_space: int = 0
    debug_vis: bool = True

    base_height_min: float = 0.20

    Kp: float = 20.0
    Kd: float = 0.5
    torque_limits: float = 23.5

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

    terrain: TerrainImporterCfg = TerrainImporterCfg(
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

    robot_cfg: ArticulationCfg = cast(
        ArticulationCfg,
        UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
    )

    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # Disable implicit P
        damping=0.0,  # Disable implicit D
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = cast(
        VisualizationMarkersCfg,
        GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal"),
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = cast(
        VisualizationMarkersCfg,
        BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current"),
    )

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    lin_vel_reward_scale: float = 1.0
    yaw_rate_reward_scale: float = 0.5

    action_rate_reward_scale: float = -0.01
    raibert_heuristic_reward_scale: float = -10.0

    orient_reward_scale: float = -5.0
    lin_vel_z_reward_scale: float = -0.02
    dof_vel_reward_scale: float = -1e-4
    ang_vel_xy_reward_scale: float = -0.001

    action_l2_reward_scale: float = -0.0001

    feet_clearance_reward_scale: float = -30.0
    tracking_contacts_shaped_force_reward_scale: float = 4.0