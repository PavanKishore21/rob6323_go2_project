# rob6323_go2_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):

    # Tutorial Part 1 & 5: Observations and State

    decimation: int = 4
    episode_length_s: float = 20.0

    action_scale: float = 0.25
    action_space: int = 12

    # [Tutorial Part 4] Observation Expansion
    # We added 4 dimensions for the clock inputs (sin/cos of gait phase).
    observation_space: int = 48 + 4
    state_space: int = 0

    debug_vis: bool = True

    # Tutorial Part 2: Custom Controller & Part 3: Termination
    # [Tutorial Part 3] Termination Criteria
    # Early stop if robot falls (base height < 0.20m).
    base_height_min: float = 0.20

    # [Tutorial Part 2] Custom Low-Level PD Controller
    # We disable the implicit controller and use these manual gains in _apply_action.
    Kp: float = 20.0  # Proportional gain (Stiffness)
    Kd: float = 0.5  # Derivative gain (Damping)
    torque_limits: float = (
        25.0  # Max torque (Nm) - We increased from tutorial default for stability
    )

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

    # Robot & Scene Setup
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

    # [Tutorial Part 2] Disable Implicit Controller
    # We have set the stiffness/damping to 0.0 allows our manual PD loop in the Env class
    # to take full control of the torques.
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=25.0,
        velocity_limit=30.0,
        stiffness=0.0,  # Zero out implicit P
        damping=0.0,  # Zero out implicit D
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # [Tutorial Part 6] Advanced Foot Interaction
    # We need accurate contact data for the feet_clearance and shaped_force rewards.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # Visualization
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

    # Tutorial Parts 1, 4, 5, 6: Rewards
    # -- Task Rewards --
    lin_vel_reward_scale: float = 1.0
    yaw_rate_reward_scale: float = 0.5

    # [Tutorial Part 1] Action Rate Penalties
    # We penalize jerky motion by checking history of actions (1st/2nd derivative).
    action_rate_reward_scale: float = -0.01

    # [Tutorial Part 4] Raibert Heuristic
    # This guides foot placement based on velocity to ensure stability.
    raibert_heuristic_reward_scale: float = -10.0

    # [Tutorial Part 5] Refining Reward Function
    # Stabilization terms to keep robot upright and efficient.
    orient_reward_scale: float = -5.0  # Keep body flat
    lin_vel_z_reward_scale: float = -0.02  # Prevent bouncing
    dof_vel_reward_scale: float = -1e-4  # Minimize energy
    ang_vel_xy_reward_scale: float = -0.001  # Prevent rolling/pitching
    action_l2_reward_scale: float = -0.0001  # Torque penalty

    # [Tutorial Part 6] Advanced Foot Interaction
    # This encourages lifting feet during swing and applying force during stance.
    feet_clearance_reward_scale: float = -30.0
    tracking_contacts_shaped_force_reward_scale: float = 4.0
