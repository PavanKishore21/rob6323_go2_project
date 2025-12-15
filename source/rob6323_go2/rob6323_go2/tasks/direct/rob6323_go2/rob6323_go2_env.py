# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )

        # Commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # --------------------------------------------------------
        # Logging Keys
        # --------------------------------------------------------
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",  # Part 1
                "raibert_heuristic",  # Part 4
                "orient",  # Part 5
                "lin_vel_z",  # Part 5
                "dof_vel",  # Part 5
                "ang_vel_xy",  # Part 5
                "action_l2",  # Rubric (Torque)
                "feet_clearance",  # Part 6
                "tracking_contacts_shaped_force",  # Part 6
            ]
        }

        # Tutorial Part 1: Action History
        self.last_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Tutorial Part 2: PD Controller Params
        self.Kp = torch.full((self.num_envs, 12), self.cfg.Kp, device=self.device)
        self.Kd = torch.full((self.num_envs, 12), self.cfg.Kd, device=self.device)
        self.torque_limits = self.cfg.torque_limits
        self.desired_joint_pos = torch.zeros(self.num_envs, 12, device=self.device)
        self._applied_torques = torch.zeros(self.num_envs, 12, device=self.device)

        # --------------------------------------------------------
        # Indices for Body and Sensors (Part 4 & 6)
        # --------------------------------------------------------
        self._feet_ids = []  # For kinematics (Part 4)
        self._feet_ids_sensor = [0, 1, 2, 3]  # Fixed indices for the restricted sensor
        self._foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._base_id = None

        try:
            base_ids, _ = self._contact_sensor.find_bodies("base")
            if len(base_ids) > 0:
                self._base_id = base_ids[0]

            for name in self._foot_names:
                # Part 4: Robot indices (for position lookups on the Articulation)
                ids, _ = self.robot.find_bodies(name)
                self._feet_ids.append(ids[0])
        except Exception:
            pass

        # Gait variables
        self.gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.clock_inputs = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.desired_contact_states = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.foot_indices = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @property
    def foot_positions_w(self) -> torch.Tensor:
        if not self._feet_ids:
            for name in self._foot_names:
                ids, _ = self.robot.find_bodies(name)
                self._feet_ids.append(ids[0])
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        # Tutorial Part 2: Manual PD Prep
        self.desired_joint_pos = (
            self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        # Tutorial Part 2: Manual PD Application
        self._applied_torques = torch.clip(
            self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
            - self.Kd * self.robot.data.joint_vel,
            -self.torque_limits,
            self.torque_limits,
        )
        self.robot.set_joint_effort_target(self._applied_torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                    self.robot.data.joint_vel,
                    self._actions,
                    self.clock_inputs,  # Tutorial Part 4
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Tutorial Part 4: Update Gait
        self._step_contact_targets()

        # --- Base Tracking ---
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        yaw_rate_error = torch.square(
            self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # --- Tutorial Part 1: Action Rate ---
        rew_action_rate = torch.sum(
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

        # --- Tutorial Part 4: Raibert ---
        rew_raibert = self._reward_raibert_heuristic()

        # --- Tutorial Part 5: Refined Rewards ---
        rew_orient = torch.sum(
            torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1
        )
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        rew_ang_vel_xy = torch.sum(
            torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1
        )

        # --- Rubric Requirement: Action L2 (Torque) ---
        rew_action_l2 = torch.sum(torch.square(self._applied_torques), dim=1)

        # --- Tutorial Part 6: Advanced Foot Interaction ---
        rew_feet_clearance = self._reward_feet_clearance()
        rew_tracking_contacts_shaped_force = (
            self._reward_tracking_contacts_shaped_force()
        )

        rewards = {
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

        # Tutorial Part 1: Update History
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Robust logic: since we restricted the contact sensor to feet only,
        # we can't detect "base" contacts with it. We rely on height/orient checks.

        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min

        died = cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Tutorial Reset logic
        self.last_actions[env_ids] = 0.0
        self.gait_indices[env_ids] = 0.0

        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(
            -1.0, 1.0
        )

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"].update(extras)

    # ----------------------------------------------------------------------
    # Tutorial Part 4: Gait Logic
    # ----------------------------------------------------------------------
    def _step_contact_targets(self):
        frequencies = 3.0
        phases = 0.5
        durations = 0.5 * torch.ones((self.num_envs,), device=self.device)

        self.gait_indices = torch.remainder(
            self.gait_indices + self.step_dt * frequencies, 1.0
        )

        foot_indices_list = [
            self.gait_indices + phases,
            self.gait_indices,
            self.gait_indices,
            self.gait_indices + phases,
        ]

        # Unwarped for Raibert
        self.foot_indices = torch.remainder(torch.stack(foot_indices_list, dim=1), 1.0)

        # Warped for Contact States
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

        # Von Mises Smoothing
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

    def _reward_raibert_heuristic(self):
        foot_rel = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        foot_body = torch.zeros(self.num_envs, 4, 3, device=self.device)

        for i in range(4):
            foot_body[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w),
                foot_rel[:, i, :],
            )

        desired_width = 0.25
        desired_length = 0.45

        ys_nom = torch.tensor(
            [
                desired_width / 2,
                -desired_width / 2,
                desired_width / 2,
                -desired_width / 2,
            ],
            device=self.device,
        ).unsqueeze(0)
        xs_nom = torch.tensor(
            [
                desired_length / 2,
                desired_length / 2,
                -desired_length / 2,
                -desired_length / 2,
            ],
            device=self.device,
        ).unsqueeze(0)

        phases = torch.abs(1.0 - self.foot_indices * 2.0) - 0.5
        freq = 3.0

        x_vel = self._commands[:, 0:1]
        yaw_vel = self._commands[:, 2:3]
        y_vel = yaw_vel * desired_length / 2

        ys_offset = phases * y_vel * (0.5 / freq)
        ys_offset[:, 2:4] *= -1  # Flip offset for rear legs
        xs_offset = phases * x_vel * (0.5 / freq)

        ys_des = ys_nom + ys_offset
        xs_des = xs_nom + xs_offset

        desired = torch.cat((xs_des.unsqueeze(2), ys_des.unsqueeze(2)), dim=2)
        err = torch.abs(desired - foot_body[:, :, 0:2])
        return torch.sum(torch.square(err), dim=(1, 2))

    # ----------------------------------------------------------------------
    # Tutorial Part 6: Advanced Foot Interaction
    # ----------------------------------------------------------------------
    def _reward_feet_clearance(self):
        target_height = 0.1
        is_swing = self.desired_contact_states < 0.5
        foot_z = self.foot_positions_w[:, :, 2]
        delta = target_height - foot_z
        penalty = torch.square(torch.clip(delta, min=0.0)) * is_swing
        return torch.sum(penalty, dim=1)

    def _reward_tracking_contacts_shaped_force(self):
        # With the restricted contact sensor, self._feet_ids_sensor is just [0,1,2,3]
        # net_forces_w is (num_envs, 4, 3)
        foot_forces = self._contact_sensor.data.net_forces_w
        foot_forces_norm = torch.norm(foot_forces, dim=-1)  # (num_envs, 4)

        robot_weight_approx = 12.0 * 9.81
        nominal_force = robot_weight_approx / 2.0

        desired_forces = self.desired_contact_states * nominal_force
        error = foot_forces_norm - desired_forces
        return torch.exp(-torch.sum(torch.square(error), dim=1) / nominal_force)

    # Debug Visualization Helpers
    def _set_debug_vis_impl(self, debug_vis: bool):
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

    def _debug_vis_callback(self, event):
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
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat
