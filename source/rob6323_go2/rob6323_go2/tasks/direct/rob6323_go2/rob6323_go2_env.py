# ============================================================================
# UPDATED ENVIRONMENT FILE: rob6323_go2_env.py
# Modifications to support rough terrain with height scanner
# ============================================================================

from __future__ import annotations

import torch
import gymnasium as gym
import numpy as np
from typing import Any, Sequence, cast

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor, RayCaster

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(
        self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs: Any
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # -- Action buffers --
        self._actions: torch.Tensor = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions: torch.Tensor = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )

        # -- Commands: [vx, vy, yaw_rate] --
        self._commands: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)

        # -- Episode logging accumulators --
        self._episode_sums: dict[str, torch.Tensor] = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",
                "raibert_heuristic",
                "orient",
                "lin_vel_z",
                "dof_vel",
                "ang_vel_xy",
                "action_l2",
                "feet_clearance",
                "tracking_contacts_shaped_force",
            ]
        }

        # -- Action history (num_envs, act_dim, history_len=3) --
        self.last_actions: torch.Tensor = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # -- Manual PD controller parameters --
        self.Kp: torch.Tensor = torch.full(
            (self.num_envs, 12), self.cfg.Kp, device=self.device
        )
        self.Kd: torch.Tensor = torch.full(
            (self.num_envs, 12), self.cfg.Kd, device=self.device
        )
        self.torque_limits: torch.Tensor = cast(torch.Tensor, self.cfg.torque_limits)

        self.desired_joint_pos: torch.Tensor = torch.zeros(
            self.num_envs, 12, device=self.device
        )
        self._applied_torques: torch.Tensor = torch.zeros(
            self.num_envs, 12, device=self.device
        )

        # --- BONUS 1: FRICTION PARAMETERS ---
        self.friction_coeffs_viscous: torch.Tensor = torch.zeros(
            self.num_envs, 12, device=self.device
        )
        self.friction_coeffs_stiction: torch.Tensor = torch.zeros(
            self.num_envs, 12, device=self.device
        )

        # -- Foot/body indexing --
        self._feet_ids: list[int] = []
        self._feet_ids_sensor: list[int] = []
        self._foot_names: list[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._base_id: int | None = None

        # -- Gait state buffers --
        self.gait_indices: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.clock_inputs: torch.Tensor = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.desired_contact_states: torch.Tensor = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.foot_indices: torch.Tensor = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        # -- Pre-computed tensors for Raibert Heuristic --
        self.desired_width = 0.25
        self.desired_length = 0.45
        self.ys_nom = torch.tensor(
            [
                self.desired_width / 2,
                -self.desired_width / 2,
                self.desired_width / 2,
                -self.desired_width / 2,
            ],
            device=self.device,
        ).unsqueeze(0)
        self.xs_nom = torch.tensor(
            [
                self.desired_length / 2,
                self.desired_length / 2,
                -self.desired_length / 2,
                -self.desired_length / 2,
            ],
            device=self.device,
        ).unsqueeze(0)

        # -- Validation flags --
        self._indices_initialized: bool = False
        self._obs_validated: bool = False

        # --- BONUS 2: HEIGHT SCANNER FLAG ---
        self._has_height_scanner: bool = hasattr(self.cfg, "height_scanner")

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self) -> None:
        """
        Create robot, sensors, terrain, clone environments, and add lighting.
        """
        self.robot: Articulation = Articulation(self.cfg.robot_cfg)
        self._contact_sensor: ContactSensor = ContactSensor(self.cfg.contact_sensor)

        # --- BONUS 2: SETUP HEIGHT SCANNER (if configured) ---
        if self._has_height_scanner:
            self._height_scanner: RayCaster = RayCaster(self.cfg.height_scanner)
            print("✓ Height scanner enabled for rough terrain")

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # --- BONUS 2: ADD HEIGHT SCANNER TO SCENE ---
        if self._has_height_scanner:
            self.scene.sensors["height_scanner"] = self._height_scanner

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _initialize_body_indices(self) -> None:
        """
        Initialize robot body indices and contact sensor indices with validation.
        """
        if self._indices_initialized:
            return

        try:
            # Find base body (optional)
            try:
                base_ids, _ = self._contact_sensor.find_bodies("base")
                if len(base_ids) > 0:
                    self._base_id = int(base_ids[0])
            except Exception:
                pass

            # Find foot bodies
            for name in self._foot_names:
                robot_ids, robot_names = self.robot.find_bodies(name)
                robot_idx = int(robot_ids[0])
                self._feet_ids.append(robot_idx)

                sensor_ids, sensor_names = self._contact_sensor.find_bodies(name)
                sensor_idx = int(sensor_ids[0])
                self._feet_ids_sensor.append(sensor_idx)

            self._indices_initialized = True

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize body indices: {e}. "
                f"This likely indicates a configuration problem."
            ) from e

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Return foot positions in world frame as (num_envs, 4, 3)."""
        if not self._feet_ids:
            self._initialize_body_indices()
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Cache current actions and compute desired joint positions."""
        if not self._indices_initialized:
            self._initialize_body_indices()

        self._actions = actions.clone()
        self.desired_joint_pos = (
            self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        """Apply manual PD torques with friction model."""
        # 1. Base PD torque
        pd_torque = (
            self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
            - self.Kd * self.robot.data.joint_vel
        )

        # 2. Friction model
        q_dot = self.robot.data.joint_vel
        tau_stiction = self.friction_coeffs_stiction * torch.tanh(q_dot / 0.1)
        tau_viscous = self.friction_coeffs_viscous * q_dot
        tau_friction = tau_stiction + tau_viscous

        # 3. Subtract friction
        torque_with_friction = pd_torque - tau_friction

        # 4. Clip to limits
        self._applied_torques = torch.clip(
            torque_with_friction, -self.torque_limits, self.torque_limits
        )

        # 5. Send to simulator
        self.robot.set_joint_effort_target(self._applied_torques)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        Assemble and return the policy observation vector.
        Flat terrain: 52 dims
        Rough terrain: 52 + 187 = 239 dims (with height scan)
        """
        self._previous_actions = self._actions.clone()

        # Base observations (52 dims)
        base_obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,  # 3
                    self.robot.data.root_ang_vel_b,  # 3
                    self.robot.data.projected_gravity_b,  # 3
                    self._commands,  # 3
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,  # 12
                    self.robot.data.joint_vel,  # 12
                    self._actions,  # 12
                    self.clock_inputs,  # 4
                )
                if tensor is not None
            ],
            dim=-1,
        )

        # --- BONUS 2: ADD HEIGHT SCAN FOR ROUGH TERRAIN ---
        if self._has_height_scanner:
            # GridPattern(1.6, 1.0, 0.1) -> 17x11 = 187 rays

            # --- FIXED LOGIC START ---
            # ray_hits_w is the world Z position of the terrain
            # root_pos_w is the world Z position of the robot base
            # We want terrain height RELATIVE to the robot base

            ground_z = self._height_scanner.data.ray_hits_w[:, :, 2]
            robot_z = self.robot.data.root_pos_w[:, 2].unsqueeze(1)

            height_data = ground_z - robot_z

            # Clip to reasonable range (-1.0 to 1.0)
            height_data = torch.clip(height_data, -1.0, 1.0)
            # --- FIXED LOGIC END ---

            # Concatenate with base observations
            obs = torch.cat([base_obs, height_data], dim=-1)
        else:
            obs = base_obs

        # One-time validation
        if not self._obs_validated:
            expected_dim = self.cfg.observation_space
            actual_dim = obs.shape[1]
            if actual_dim != expected_dim:
                raise RuntimeError(
                    f"Observation dimension mismatch! "
                    f"Expected {expected_dim}, got {actual_dim}. "
                    f"Check observation_space in config."
                )
            print(f"✓ Observation validation passed: {actual_dim} dims")
            self._obs_validated = True

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the scalar reward for each environment."""
        self._step_contact_targets()

        # Base tracking (exponential shaping)
        lin_vel_error: torch.Tensor = torch.sum(
            torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        lin_vel_error_mapped: torch.Tensor = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error: torch.Tensor = torch.square(
            self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped: torch.Tensor = torch.exp(-yaw_rate_error / 0.25)

        # Action-rate penalties
        rew_action_rate: torch.Tensor = torch.sum(
            torch.square(self._actions - self.last_actions[:, :, 0]), dim=1
        ) * (self.cfg.action_scale**2)

        rew_action_rate += torch.sum(
            torch.square(
                self._actions
                - 2 * self.last_actions[:, :, 0]
                + self.last_actions[:, :, 1]
            ),
            dim=1,
        ) * (self.cfg.action_scale**2)

        # Raibert heuristic
        rew_raibert: torch.Tensor = self._reward_raibert_heuristic()

        # Stabilization terms
        rew_orient: torch.Tensor = torch.sum(
            torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1
        )
        rew_lin_vel_z: torch.Tensor = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        rew_dof_vel: torch.Tensor = torch.sum(
            torch.square(self.robot.data.joint_vel), dim=1
        )
        rew_ang_vel_xy: torch.Tensor = torch.sum(
            torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1
        )

        # Torque penalty
        rew_action_l2: torch.Tensor = torch.sum(
            torch.square(self._applied_torques), dim=1
        )

        # Foot interaction terms
        rew_feet_clearance: torch.Tensor = self._reward_feet_clearance()
        rew_tracking_contacts_shaped_force: torch.Tensor = (
            self._reward_tracking_contacts_shaped_force()
        )

        rewards: dict[str, torch.Tensor] = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped
            * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped
            * self.cfg.yaw_rate_reward_scale,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "raibert_heuristic": rew_raibert * self.cfg.raibert_heuristic_reward_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            "action_l2": rew_action_l2 * self.cfg.action_l2_reward_scale,
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts_shaped_force": rew_tracking_contacts_shaped_force
            * self.cfg.tracking_contacts_shaped_force_reward_scale,
        }

        # Update action history
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions

        reward: torch.Tensor = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # NaN/Inf safety check
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            print("\n⚠️  WARNING: NaN/Inf detected in rewards!")
            for key, value in rewards.items():
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"  Problem in reward: {key}")
            raise RuntimeError("Invalid reward values detected!")

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return termination flags."""
        time_out: torch.Tensor = self.episode_length_buf >= self.max_episode_length - 1

        cstr_upsidedown: torch.Tensor = self.robot.data.projected_gravity_b[:, 2] > 0
        base_height: torch.Tensor = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min: torch.Tensor = base_height < self.cfg.base_height_min

        died: torch.Tensor = cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the specified environments and populate episode logs."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = cast(Sequence[int], self.robot._ALL_INDICES)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.gait_indices[env_ids] = 0.0

        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(
            -1.0, 1.0
        )

        # --- BONUS 1: RANDOMIZE FRICTION ---
        num_reset = len(env_ids)
        viscous_sample = torch.rand(num_reset, 1, device=self.device) * 0.3
        self.friction_coeffs_viscous[env_ids] = viscous_sample.expand(-1, 12)

        stiction_sample = torch.rand(num_reset, 1, device=self.device) * 2.5
        self.friction_coeffs_stiction[env_ids] = stiction_sample.expand(-1, 12)

        joint_pos: torch.Tensor = self.robot.data.default_joint_pos[env_ids]
        joint_vel: torch.Tensor = self.robot.data.default_joint_vel[env_ids]
        default_root_state: torch.Tensor = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Episode logging
        extras: dict[str, float] = {}
        for key in self._episode_sums.keys():
            if key in self._episode_sums:
                episodic_sum_avg: torch.Tensor = torch.mean(
                    self._episode_sums[key][env_ids]
                )
                extras["Episode_Reward/" + key] = float(
                    episodic_sum_avg / self.max_episode_length_s
                )
                self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = {}
        self.extras["log"].update(extras)

    def _step_contact_targets(self) -> None:
        """
        Calculates the desired gait (contact schedule) based on a trot pattern.
        """
        frequencies: float = 3.0
        phases: float = 0.5
        durations: torch.Tensor = 0.5 * torch.ones((self.num_envs,), device=self.device)
        self.gait_indices = torch.remainder(
            self.gait_indices + self.step_dt * frequencies, 1.0
        )
        foot_indices_list = [
            self.gait_indices + phases,
            self.gait_indices,
            self.gait_indices,
            self.gait_indices + phases,
        ]
        self.foot_indices = torch.remainder(torch.stack(foot_indices_list, dim=1), 1.0)
        warped_indices = self.foot_indices.clone()
        for i in range(4):
            fi = warped_indices[:, i]
            stance = fi < durations
            swing = ~stance
            fi[stance] = (fi[stance] / durations[stance]) * 0.5
            fi[swing] = (
                0.5 + ((fi[swing] - durations[swing]) / (1 - durations[swing])) * 0.5
            )
            warped_indices[:, i] = fi
        self.clock_inputs = torch.sin(2 * np.pi * warped_indices)
        kappa = 0.07
        cdf = torch.distributions.normal.Normal(0, kappa).cdf

        def smooth(fi):
            return cdf(torch.remainder(fi, 1.0)) * (
                1 - cdf(torch.remainder(fi, 1.0) - 0.5)
            ) + cdf(torch.remainder(fi, 1.0) - 1.0) * (
                1 - cdf(torch.remainder(fi, 1.0) - 1.5)
            )

        for i in range(4):
            self.desired_contact_states[:, i] = smooth(warped_indices[:, i])

    def _reward_raibert_heuristic(self) -> torch.Tensor:
        foot_rel = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        foot_body = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            foot_body[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w),
                foot_rel[:, i, :],
            )
        phases = torch.abs(1.0 - self.foot_indices * 2.0) - 0.5
        freq = 3.0
        x_vel = self._commands[:, 0:1]
        yaw_vel = self._commands[:, 2:3]
        y_vel = yaw_vel * self.desired_length / 2
        ys_offset = phases * y_vel * (0.5 / freq)
        ys_offset[:, 2:4] *= -1
        xs_offset = phases * x_vel * (0.5 / freq)
        ys_des = self.ys_nom + ys_offset
        xs_des = self.xs_nom + xs_offset
        desired = torch.cat((xs_des.unsqueeze(2), ys_des.unsqueeze(2)), dim=2)
        err = torch.abs(desired - foot_body[:, :, 0:2])
        return torch.sum(torch.square(err), dim=(1, 2))

    def _reward_feet_clearance(self) -> torch.Tensor:
        target_height = 0.1
        is_swing = self.desired_contact_states < 0.5
        foot_z = self.foot_positions_w[:, :, 2]
        delta = target_height - foot_z
        penalty = torch.square(torch.clip(delta, min=0.0)) * is_swing
        return torch.sum(penalty, dim=1)

    def _reward_tracking_contacts_shaped_force(self) -> torch.Tensor:
        all_forces = self._contact_sensor.data.net_forces_w
        foot_forces = all_forces[:, self._feet_ids_sensor, :]
        foot_forces_norm = torch.norm(foot_forces, dim=-1)
        robot_weight_approx = 12.0 * 9.81
        nominal_force = robot_weight_approx / 2.0
        desired_forces = self.desired_contact_states * nominal_force
        error = foot_forces_norm - desired_forces
        return torch.exp(-torch.sum(torch.square(error), dim=1) / nominal_force)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event: Any) -> None:
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self._commands[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )
        self.goal_vel_visualizer.visualize(
            base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale
        )
        self.current_vel_visualizer.visualize(
            base_pos_w, vel_arrow_quat, vel_arrow_scale
        )

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(
            default_scale, device=self.device, dtype=torch.float32
        ).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat
