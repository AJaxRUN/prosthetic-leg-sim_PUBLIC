from gymnasium import spaces
import gymnasium as gym
import mujoco.viewer 
import numpy as np
import mujoco
import math

class ProstheticHopperEnv(gym.Env):
    def __init__(self, XML_FILE, render_mode=None):
        super(ProstheticHopperEnv, self).__init__()

        self.model = mujoco.MjModel.from_xml_path(XML_FILE)
        self.data = mujoco.MjData(self.model)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.render_mode = render_mode
        self.viewer = None

    def _get_obs(self):
        obs = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        return obs

    def reset(self,  seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.asarray([0, 3, 0, math.radians(55), math.radians(45)])
        self.data.qvel = np.asarray([0, 0 ,0, 0, 0])
        observation = self._get_obs()

        return observation.astype(np.float32), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        observation = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._check_termination()
        if terminated:
            reward = -100
        info = {}

        return observation.astype(np.float32), reward, terminated, False, info

    def _calculate_reward(self):
        '''
        qpos indices (positions):
            x-position of the root
            z-position of the root
            y-axis rotation of the root
            knee joint angle
            foot joint angle
        qvel indices (velocities):
            x-velocity of the root
            z-velocity of the root
            y-axis angular velocity of the root
            knee joint angular velocity
            foot joint angular velocity
        Sensors:
            sensor_x_position: 0
            sensor_y_position: 1
            sensor_z_position: 2
            load_sensor [x, y, z]: [3, 4, 5]
            foot_gyro_sensor [x, y, z]: [6, 7, 8]
            foot_accel_sensor [x, y, z]: [9, 10, 11]
            knee_gyro_sensor [x, y, z]: [12, 13, 14]
            knee_accel_sensor [x, y, z]: [15, 16, 17]
            sensor_x_force [x, y, z]: [18, 19, 20]
        '''
        reward = 10
        forward_velocity = self.data.qvel[0]
        upward_velocity = self.data.qvel[1]
        load_in_kg = self.data.sensordata[5] / 9.81
        x_dist = self.data.qpos[0]
        height = self.data.qpos[1]
        reward -= abs(x_dist)
        reward -= abs(height - 2)
        # y_rot = self.data.qpos[2]
        # if y_rot > 0.8 or y_rot < -0.8:
        #     reward -= abs(reward) * 10
        # reward += upward_velocity
        # if load_in_kg < 0:
        #     reward -= height
        # reward += load_in_kg + x_dist
        
        return reward

    def _check_termination(self):
        if abs(self.data.qpos[1] - 2) > 5:
            return True
        if self.data.qpos[1] < 1:
            return True
        if abs(self.data.qpos[0]) > 5:
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
