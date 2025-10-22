from typing import Dict, Tuple

from os import path

import numpy as np

import quaternion

from scipy.spatial.transform import Rotation as R

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import mujoco

from robot_descriptions import booster_t1_mj_description


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


class T1Env(utils.EzPickle, MujocoEnv):
    def __init__(
        self,
        ref: Dict[str, np.array],
        episode_length: int,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            ref,
            episode_length,
            **kwargs,
        )

        MujocoEnv.__init__(
            self,
            model_path=path.join(
                booster_t1_mj_description.PACKAGE_PATH, "scene.xml"
            ),
            frame_skip=4,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.model.opt.timestep = 0.005

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.episode_length = episode_length

        self._nbody = self.model.nbody

        self._root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Trunk"
        )

        self._end_effector_ids = np.array([
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link"
            ),
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link"
            ),
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_link"
            ),
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link"
            ),
        ])

        root_z_pitch_roll_size = 3
        jnt_qpos_size = self.model.jnt_range[1:].shape[0]
        qvel_size = self.model.nv
        xipos_rel_root_size = np.delete(
            (self.data.xipos - self.data.xipos[self._root_id]),
            [0, self._root_id], axis=0,
        ).size,
        xmat_rel_root_size = np.delete(
            self.data.xmat.reshape(-1, 3, 3),
            [0, self._root_id], axis=0,
        )[..., :-1].size,

        self.observation_space = Box(
            low=np.concatenate([
                np.array([0.0, -np.pi/2, -np.pi]),
                self.model.jnt_range[1:, 0],
                np.full(qvel_size, -np.inf),
                np.full(xipos_rel_root_size, -np.inf),
                np.full(xmat_rel_root_size, -1.0)
            ]),
            high=np.concatenate([
                np.array([np.inf, np.pi/2, np.pi]),
                self.model.jnt_range[1:, 1],
                np.full(qvel_size, np.inf),
                np.full(xipos_rel_root_size, np.inf),
                np.full(xmat_rel_root_size, 1.0)
            ]),
            dtype=np.float64,
        )

        self.observation_structure = {
            "root_z_pitch_roll": root_z_pitch_roll_size,
            "jnt_qpos": jnt_qpos_size,
            "qvel": qvel_size,
            "xipos_rel_root": xipos_rel_root_size,
            "xmat_rel_root": xmat_rel_root_size,
        }

        self.ref_qposs = ref["qposs"]
        self.ref_qvels = ref["qvels"]
        self.ref_xiposs = ref["xiposs"]
        self.ref_xmats = ref["xmats"]
        self.ref_coms = ref["coms"]

        assert self.episode_length <= self.ref_qposs.shape[0]

        assert self.ref_qposs.shape[0] == self.ref_qvels.shape[0]
        assert self.ref_qposs.shape[0] == self.ref_xiposs.shape[0]
        assert self.ref_qposs.shape[0] == self.ref_xmats.shape[0]
        assert self.ref_qposs.shape[0] == self.ref_coms.shape[0]

        self.ref_xiposs_rel_root = (
            (self.ref_xiposs - self.ref_xiposs[:, self._root_id])
            @ self.ref_xmats[:, self._root_id].reshape(
                self.ref_xmats.shape[0], 3, 3
            )[:, None, :, :]
        )
        self.ref_xmats_rel_root = (
            np.transpose(self.ref_xmats[:, self._root_id].reshape(
                self.ref_xmats.shape[0], 3, 3
            ), axes=(0, 2, 1))[:, None, :, :]
            @ self.ref_xmats.reshape(self.ref_xmats.shape[0], -1, 3, 3)
        )

    def _get_observation(self) -> Tuple[np.array, Dict[str, np.array]]:
        root_z = self.data.qpos[2]
        _, root_pitch, root_roll = R.from_quat(
            np.roll(self.data.qpos[3:7], -1)
        ).as_euler("zyx")
        root_z_pitch_roll = np.array([root_z, root_pitch, root_roll])

        jnt_qpos = self.data.qpos[7:]

        qvel = self.data.qvel

        xipos_rel_root = (
            (self.data.xipos - self.data.xipos[self._root_id])
            @ self.data.xmat[self._root_id].reshape(
                3, 3
            )[None, :, :]
        )

        xmat_rel_root = (
            np.transpose(self.data.xmat[self._root_id].reshape(
                3, 3
            ), axes=(1, 0))[None, :, :]
            @ self.data.xmat.reshape(-1, 3, 3)
        )

        return np.concatenate([
            root_z_pitch_roll.flatten(),
            jnt_qpos.flatten(),
            qvel.flatten(),
            np.delete(
                xipos_rel_root, [0, self._root_id], axis=0
            ).flatten(),
            np.delete(
                xmat_rel_root, [0, self._root_id], axis=0
            )[..., :-1].flatten(),
        ]), {
            "root_z_pitch_roll": root_z_pitch_roll,
            "jnt_qpos": jnt_qpos,
            "qvel": qvel,
            "xipos_rel_root": xipos_rel_root,
            "xmat_rel_root": xmat_rel_root,
        }

    def _get_reward(
        self,
        observation_dict: Dict[str, np.array],
    ) -> np.float64:
        ref_idx = self.init_ref_idx + self.step

        reward_pose = self._get_reward_pose(
            ref_idx, observation_dict,
        )
        reward_velocity = self._get_reward_velocity(
            ref_idx, observation_dict,
        )
        reward_end_effector = self._get_reward_end_effector(
            ref_idx, observation_dict,
        )
        reward_root = self._get_reward_root(
            ref_idx, observation_dict,
        )
        reward_center_of_mass = self._get_reward_center_of_mass(
            ref_idx, observation_dict,
        )

        reward = np.float64(
            0.50 * reward_pose +
            0.05 * reward_velocity +
            0.15 * reward_end_effector +
            0.20 * reward_root +
            0.10 * reward_center_of_mass
        )
        reward_info = {
            "reward_pose": reward_pose,
            "reward_velocity": reward_velocity,
            "reward_end_effector": reward_end_effector,
            "reward_root": reward_root,
            "reward_center_of_mass": reward_center_of_mass,
        }

        return reward, reward_info

    def _get_reward_pose(
        self,
        ref_idx: int,
        observation_dict: Dict[str, np.array],
    ) -> np.float64:
        xmat_rel_root = observation_dict["xmat_rel_root"]
        ref_xmat_rel_root = self.ref_xmats_rel_root[ref_idx]

        ref_xmat_rel_xmat = np.transpose(
            xmat_rel_root, axes=(0, 2, 1)
        ) @ ref_xmat_rel_root

        squared_errs = np.arccos(np.clip(
            (np.trace(ref_xmat_rel_xmat, axis1=1, axis2=2) - 1)/2,
            -1.0, 1.0,
        )) ** 2

        return np.exp(-30.0 / self._nbody * np.sum(squared_errs))

    def _get_reward_velocity(
        self,
        ref_idx: int,
        observation_dict: Dict[str, np.array],
    ) -> np.float64:
        qvel = observation_dict["qvel"]
        ref_qvel = self.ref_qvels[ref_idx]

        squared_errs = (
            ref_qvel[6:] - qvel[6:]
        ) ** 2

        return np.exp(-1.5 / self._nbody * np.sum(squared_errs))

    def _get_reward_end_effector(
        self,
        ref_idx: int,
        observation_dict: Dict[str, np.array],
    ) -> np.float64:
        end_effector_xipos_rel_root = (
            observation_dict["xipos_rel_root"][self._end_effector_ids]
        )
        end_effector_ref_xipos_rel_root = (
            self.ref_xiposs_rel_root[ref_idx][self._end_effector_ids]
        )

        squared_errs = np.sum((
            end_effector_ref_xipos_rel_root - end_effector_xipos_rel_root
        ) ** 2, axis=1)

        return np.exp(-10.0 * np.sum(squared_errs))

    def _get_reward_root(
        self,
        ref_idx: int,
        observation_dict: Dict[str, np.array],
    ) -> np.float64:
        lin_pos = self.data.qpos[:3]
        ang_pos = quaternion.quaternion(self.data.qpos[3:7])
        lin_vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]

        ref_lin_pos = self.ref_qposs[ref_idx][:3]
        ref_ang_pos = quaternion.quaternion(self.ref_qposs[ref_idx][3:7])
        ref_lin_vel = self.ref_qvels[ref_idx][:3]
        ref_ang_vel = self.ref_qvels[ref_idx][3:6]

        lin_pos_squared_err = np.sum((ref_lin_pos - lin_pos) ** 2)
        ang_pos_squared_err = (ref_ang_pos * ang_pos.conj()).angle() ** 2
        lin_vel_squared_err = np.sum((ref_lin_vel - lin_vel) ** 2)
        ang_vel_squared_err = np.sum((ref_ang_vel - ang_vel) ** 2)

        squared_err = (
            1.000 * lin_pos_squared_err +
            0.100 * ang_pos_squared_err +
            0.010 * lin_vel_squared_err +
            0.001 * ang_vel_squared_err
        )

        return np.exp(-5.0 * squared_err)

    def _get_reward_center_of_mass(
        self,
        ref_idx: int,
        observation_dict: Dict[str, np.array],
    ) -> np.float64:
        com = self.data.subtree_coms[0]
        ref_com = self.ref_coms[ref_idx]

        squared_err = np.sum((ref_com - com) ** 2)

        return np.exp(-10.0 * squared_err)

    def _get_terminated(
        self,
        observation_dict: Dict[str, np.array],
    ) -> bool:
        terminated = observation_dict["root_z_pitch_roll"][0] < 0.5

        return terminated

    def _get_truncated(
        self,
        observation_dict: Dict[str, np.array],
    ) -> bool:
        truncated = self.step >= self.episode_length

        return truncated

    def step(self, action):
        self.step = self.step + 1

        self.do_simulation(action, self.frame_skip)

        observation, observation_dict = self._get_observation()
        reward, reward_info = self._get_reward(observation_dict)
        terminated = self._get_terminated(observation_dict)
        truncated = self._get_truncated(observation_dict)
        info = {
            **reward_info
        }

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        self.init_ref_idx = self.np_random.integers(
            0, self.ref_qposs.shape[0] - self.episode_length,
        )

        self.step = 0

        qpos = self.ref_qposs[self.init_ref_idx] + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nq
        )
        qvel = self.ref_qvels[self.init_ref_idx] + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation, _ = self._get_observation()

        return observation
