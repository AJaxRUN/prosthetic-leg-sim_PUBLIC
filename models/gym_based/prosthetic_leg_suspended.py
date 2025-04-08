from gymnasium import spaces
import gymnasium as gym
import mujoco.viewer 
import numpy as np
import mujoco
from utils.prosthetic_hopper_utils import get_stride, load_mat_data, compute_squared_error

SELECTED_STRIDE = 16

############ Mujoco Data #######################
'''
    qpos indices (positions):
        y-axis rotation of the root
        knee joint angle
        foot joint angle
    qvel indices (velocities):
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

class ProstheticEnv(gym.Env):
    def __init__(self, XML_FILE, render_mode=None):
        super(ProstheticEnv, self).__init__()

        self.model = mujoco.MjModel.from_xml_path(XML_FILE)
        self.data = mujoco.MjData(self.model)
        self.one_stride_data = get_stride(mat_data=load_mat_data(), stride_start_num = SELECTED_STRIDE)
        self.simulation_current_step = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.render_mode = render_mode
        self.viewer = None

    def _get_obs(self):
        obs = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        return obs

    def reset(self,  seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.simulation_current_step = 0
        self.data.qpos[:] = [self.one_stride_data["thighIMU_Theta_X"][0], self.one_stride_data["kneeAngles"][0], self.one_stride_data["ankleAngles"][0]]
        self.data.qvel = np.asarray([0, 0, 0])
        observation = self._get_obs()

        return observation.astype(np.float32), {}

    def step(self, action):
        self.data.ctrl[:] = action[0]
        self.data.qpos[1] = self.one_stride_data["kneeAngles"][self.simulation_current_step]
        self.data.qpos[2] = self.one_stride_data["ankleAngles"][self.simulation_current_step]
        self.simulation_current_step += 1
        mujoco.mj_step(self.model, self.data)
        observation = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._check_termination()
        info = {}
        return observation.astype(np.float32), reward, terminated, False, info

    def _calculate_reward(self):
        actual_theta = self.one_stride_data["thighIMU_Theta_X"][self.simulation_current_step]
        measured_theta = self.data.qpos[0]
        reward = compute_squared_error(self.one_stride_data["thighIMU_Theta_X"][0], measured_theta)
        return reward

    def _check_termination(self):
        if self.simulation_current_step < self.one_stride_data["dataLength"] - 1:
            return False
        if self._calculate_reward() < -225:
            return True
        return True

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
