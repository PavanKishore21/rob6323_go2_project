# ROB6323 Go2 Locomotion — PPO in Isaac Lab (Friction + Reward Shaping)

This repository contains a DirectRL environment for **Unitree Go2** locomotion in **Isaac Lab**, trained with **PPO**. Starting from a velocity-tracking baseline, we introduce:
1) **Reward shaping / regularization** for stability + smoothness + gait structure  
2) **Actuator friction (stiction + viscous)** applied to PD torques, with **per-episode randomization**

The goal is to preserve commanded velocity tracking while improving robustness (fewer falls, more timeouts, smoother actions).

---

## 1) Project Structure


- `rob6323_go2_env_cfg.py`  
  Environment configuration: sim dt/decimation, PD gains, torque limits, contact sensor config, reward scales.

- `rob6323_go2_env.py`  
  DirectRLEnv implementation:
  - Action → joint target mapping
  - Manual PD torque control
  - **Friction torque model** (bonus)
  - Reward terms + termination logic
  - Feet-only contact sensing + shaped-force reward

---

## 2) Documentation of Changes (Major Additions + Rationale)

### A) Manual PD Control Pipeline (baseline foundation)
**What changed / exists:**
- Policy outputs `12D` actions.
- Actions mapped to desired joint positions:
  - `qd = q_default + action_scale * a`
- PD torque computed each step:
  - `τ_PD = Kp(qd − q) − Kd * q_dot`

**Why:**
- Gives an interpretable, stable low-level controller so PPO can learn coordination rather than raw stabilization.

**Where:**
- `Rob6323Go2Env._pre_physics_step()` and `Rob6323Go2Env._apply_action()`

---

### B) Reward Shaping + Regularization (stability + smoothness + gait)
**What changed / added:**
- Tracking rewards (exp-shaped):
  - `track_lin_vel_xy_exp`, `track_ang_vel_z_exp`
- Smoothness penalties:
  - action-rate (1st + 2nd differences) → `rew_action_rate`
- Stability penalties:
  - tilt/orientation → `orient`
  - vertical velocity → `lin_vel_z`
  - roll/pitch angular velocity → `ang_vel_xy`
  - joint velocity → `dof_vel`
- Effort penalty:
  - applied torque L2 → `action_l2`
- Gait/foot structure:
  - Raibert heuristic foot placement → `raibert_heuristic`
  - swing-phase clearance penalty → `feet_clearance`
- Contact-based shaping:
  - desired contact schedule + force matching → `tracking_contacts_shaped_force`

**Why:**
- Tracking alone often leads to brittle/high-frequency control and unstable exploration.
- Regularizers encourage smoother actions, reduced bouncing/tilt, and cleaner gait structure.

**Where:**
- `Rob6323Go2Env._get_rewards()`
- `Rob6323Go2Env._step_contact_targets()`
- `Rob6323Go2Env._reward_raibert_heuristic()`, `_reward_feet_clearance()`,
  `_reward_tracking_contacts_shaped_force()`

---

### C) BONUS: Actuator Friction Model (stiction + viscous) + Randomization
**What changed / added:**
- Added per-joint friction parameters (buffers):
  - `friction_coeffs_stiction` (Fs)
  - `friction_coeffs_viscous` (μv)
- Friction torque model:
  - `τ_friction = Fs * tanh(q_dot / 0.1) + μv * q_dot`
- Applied by subtracting from PD torque:
  - `τ = τ_PD − τ_friction`
- Per-episode randomization at reset:
  - `Fs ~ Uniform(0, 2.5)`
  - `μv ~ Uniform(0, 0.3)`

**Why:**
- Penalizes overly sharp motion and discourages exploiting “perfect actuators” in sim.
- Randomization improves robustness across actuator conditions and reduces overfitting.

**Where:**
- Buffers: `Rob6323Go2Env.__init__()`
- Torque application: `Rob6323Go2Env._apply_action()`
- Randomization: `Rob6323Go2Env._reset_idx()`

---

### D) Observation Space Update (Clock Inputs)
**What changed / added:**
- Added gait phase features (`clock_inputs`, 4 dims) to the policy observation.
- Updated `observation_space` accordingly.

**Why:**
- Phase information helps learn periodic coordination and structured locomotion.

**Where:**
- `Rob6323Go2Env._step_contact_targets()`
- `Rob6323Go2Env._get_observations()`
- `Rob6323Go2EnvCfg.observation_space`

---

### E) Feet-Only Contact Sensing (consistent indexing)
**What changed / added:**
- `ContactSensorCfg` targets only feet prims.
- Dynamic indexing maps the 4 feet for position + force usage.

**Why:**
- Makes reward computation consistent and avoids ambiguity across bodies.

**Where:**
- `Rob6323Go2EnvCfg.contact_sensor`
- `Rob6323Go2Env._initialize_body_indices()`
- `Rob6323Go2Env._reward_tracking_contacts_shaped_force()`

---

## 3) Key Hyperparameters (from cfg defaults)

- Simulation:
  - `dt = 1/200`
  - `decimation = 4`  → policy at 50 Hz
  - `episode_length_s = 20`
- Controller:
  - `Kp = 20.0`, `Kd = 0.5`
  - `torque_limits = 23.5`
- Actions:
  - `action_dim = 12`
  - `action_scale = 0.25`
- Observations:
  - Includes root velocities, projected gravity, commands, joint pos/vel, actions, + `clock_inputs (4)`
- Rewards (scales in `Rob6323Go2EnvCfg`):
  - tracking: `lin_vel_reward_scale=1.0`, `yaw_rate_reward_scale=0.5`
  - action smoothness: `action_rate_reward_scale=-0.01`
  - gait/feet: `raibert_heuristic_reward_scale=-10.0`, `feet_clearance_reward_scale=-30.0`
  - stability: `orient=-5.0`, `lin_vel_z=-0.02`, `dof_vel=-1e-4`, `ang_vel_xy=-0.001`
  - effort: `action_l2=-1e-4`
  - contact force shaping: `tracking_contacts_shaped_force_reward_scale=4.0`

---

## 4) Reproducibility (Exact Commands + Seeds)

> .To run training and generate logs, please follow the instructions provided in
the official ROB6323 Go2 project repository:

https://github.com/machines-in-motion/rob6323_go2_project?tab=readme-ov-file

All commands, logging behavior, and seed handling used in this project are
consistent with the procedures described in the above repository.


### A) Create Environment
The environment setup follows the official ROB6323 Go2 project instructions.

Please refer to the following tutorial for detailed installation and setup steps:
https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md

All experiments and results reported in this project assume the environment has been set up exactly as described in the above tutorial.
