from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from gymnasium import spaces
import gymnasium as gym
import mujoco.viewer 
import numpy as np
import mujoco

class CustomHopperEnv(gym.Env):
    def __init__(self, XML_FILE, render_mode=None):
        super(CustomHopperEnv, self).__init__()

        self.model = mujoco.MjModel.from_xml_path(XML_FILE)
        self.data = mujoco.MjData(self.model)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.render_mode = render_mode
        self.viewer = None

    def _get_obs(self):
        obs = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        return obs

    def reset(self,  seed=None, options=None):
        '''
        qpos indices (positions):
            x-position of the root
            z-position of the root
            y-axis rotation of the root
            thigh joint angle
            leg joint angle
            foot joint angle
        qvel indices (velocities):
            x-velocity of the root
            z-velocity of the root
            y-axis angular velocity of the root
            thigh joint angular velocity
            leg joint angular velocity
            foot joint angular velocity
        '''
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.asarray([0, 2, 0, 0, 0, 0])
        self.data.qvel = np.asarray([0, 0 ,0, 0, 0, 0])
        observation = self._get_obs()

        return observation.astype(np.float32), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        observation = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._check_termination()
        info = {}

        return observation.astype(np.float32), reward, terminated, False, info

    def _calculate_reward(self):
        forward_velocity = self.data.qvel[0]
        height = self.data.qpos[1]
        y_rot = self.data.qpos[2]
        reward = height + forward_velocity
        if y_rot < 0.8 and y_rot > -0.7:
            reward = abs(y_rot)
        return reward

    def _check_termination(self):
        if self.data.qpos[1] < 0.3:
            return True
        return False

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
