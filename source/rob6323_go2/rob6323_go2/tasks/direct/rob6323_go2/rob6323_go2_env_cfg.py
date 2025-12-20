# ============================================================================
# BONUS TASK 2: UNEVEN TERRAIN LOCOMOTION FOR GO2 ROBOT
# Complete Configuration File: rob6323_go2_env_cfg.py
# ============================================================================

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg  # <--- FIXED TYPO
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import GO2_CFG  # Your Go2 robot config
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # Rough terrain config


# ============================================================================
# EVENT CONFIGURATION (Domain Randomization)
# ============================================================================
@configclass
class EventCfg:
    """Configuration for domain randomization events."""

    # Randomize physics material properties at startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.2),  # Vary ground friction
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.1),  # Small bounce
            "num_buckets": 64,
        },
    )

    # Randomize robot base mass (payload variation)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2.0, 3.0),  # +/- kg variation
            "operation": "add",
        },
    )


# ============================================================================
# FLAT TERRAIN CONFIGURATION (Your Original Config)
# ============================================================================
@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    """Configuration for Go2 quadruped on FLAT terrain."""

    # ========================================================================
    # ENVIRONMENT SETTINGS
    # ========================================================================
    episode_length_s = 20.0
    decimation = 4  # Control frequency = sim_freq / decimation
    action_scale = 0.25
    action_space = 12  # 12 joint positions
    observation_space = 52  # Updated below
    state_space = 0

    # ========================================================================
    # SIMULATION SETTINGS
    # ========================================================================
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,  # 200 Hz simulation
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Flat terrain (plane)
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

    # ========================================================================
    # SCENE CONFIGURATION
    # ========================================================================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # ========================================================================
    # DOMAIN RANDOMIZATION
    # ========================================================================
    events: EventCfg = EventCfg()

    # ========================================================================
    # ROBOT CONFIGURATION
    # ========================================================================
    robot: ArticulationCfg = GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Contact sensor on all robot bodies (feet + base for termination)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # ========================================================================
    # PD CONTROLLER PARAMETERS
    # ========================================================================
    Kp: float = 20.0  # Proportional gain
    Kd: float = 0.5  # Derivative gain
    torque_limits: tuple = (23.7, 23.7, 45.43) * 4  # (hip, thigh, calf) x 4 legs

    # ========================================================================
    # REWARD SCALES (Tuned for Flat Terrain)
    # ========================================================================
    lin_vel_reward_scale = 1.5
    yaw_rate_reward_scale = 0.75
    action_rate_reward_scale = -0.005
    raibert_heuristic_reward_scale = -10.0
    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -1.0
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.05
    action_l2_reward_scale = -0.0001
    feet_clearance_reward_scale = -10.0
    tracking_contacts_shaped_force_reward_scale = 1.5

    # ========================================================================
    # TERMINATION CONDITIONS
    # ========================================================================
    base_height_min = 0.25  # Terminate if robot drops too low

    # ========================================================================
    # VISUALIZATION (Debug)
    # ========================================================================
    debug_vis: bool = False

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = (
        GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Velocity")
    )


# ============================================================================
# ROUGH TERRAIN CONFIGURATION (BONUS TASK 2)
# ============================================================================
@configclass
class Rob6323Go2RoughEnvCfg(Rob6323Go2EnvCfg):
    """Configuration for Go2 quadruped on ROUGH TERRAIN."""

    # 52 (base) + 187 (height scan) = 239
    observation_space = 239

    # ========================================================================
    # ROUGH TERRAIN CONFIGURATION
    # ========================================================================
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,  # Use predefined rough terrain
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # ========================================================================
    # HEIGHT SCANNER (Proprioceptive Perception)
    # ========================================================================
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # Start 20m above
        attach_yaw_only=True,  # Only follow robot yaw (not pitch/roll)
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,  # 10cm spacing
            size=[1.6, 1.0],  # 1.6m forward x 1m wide
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # Grid pattern creates: ceil(1.6/0.1) * ceil(1.0/0.1) = 17 x 11 = 187 rays

    # ========================================================================
    # ADJUSTED REWARD SCALES FOR ROUGH TERRAIN
    # ========================================================================
    orient_reward_scale = -1.0  # Reduced from -5.0 (allow more body tilt)
    lin_vel_reward_scale = 2.0  # Increased from 1.5 (prioritize forward motion)
    yaw_rate_reward_scale = 1.0  # Increased from 0.75
    feet_clearance_reward_scale = -15.0  # Increased from -10.0
    tracking_contacts_shaped_force_reward_scale = 2.0  # Increased from 1.5

    # ========================================================================
    # TERMINATION CONDITIONS (More Lenient)
    # ========================================================================
    base_height_min = 0.20  # Allow slightly lower (rough terrain dips)
