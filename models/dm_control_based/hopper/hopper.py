from dm_control import suite
import gymnasium as gym
import numpy as np


class CustomHopperEnv(gym.Env):
    def __init__(self):
        super(CustomHopperEnv, self).__init__()
        self.env = suite.load(
            domain_name="hopper", task_name="hop", visualize_reward=True
        )
        self.physics = self.env.physics
        self.model = self.physics.model
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def reward_function(self, qpos, qvel):
        forward_velocity = self.physics.data.qvel[0]
        velocity_reward = forward_velocity

        torso_height = self.physics.data.qpos[2]
        upright_reward = np.clip(torso_height, 0.7, 1.5)  # Reward for staying upright

        torso_angle = self.physics.data.qpos[3]  # Angle of the torso
        balance_penalty = -abs(torso_angle)  # Penalize deviation from upright

        control_effort = np.sum(
            np.square(self.physics.data.ctrl)
        )  # Control effort penalty
        control_penalty = -0.001 * control_effort

        # Penalize for falling
        fall_penalty = -100 if torso_height < 0.7 else 0

        total_reward = (
            velocity_reward
            + upright_reward
            + balance_penalty
            + control_penalty
            + fall_penalty
        )
        return total_reward

    def step(self, action):
        action = action.astype(np.float32)
        time_step = self.env.step(action)
        qpos = self.env.physics.data.qpos
        qvel = self.env.physics.data.qvel
        obs = qpos[:].copy().astype(np.float32)
        reward = self.reward_function(qpos, qvel)
        terminated = time_step.last()
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env.reset()
        # self.env.physics.data.qpos[:] = [
        #     np.float32(np.random.uniform(-1, 1)),
        #     np.float32(np.random.uniform(-1, 1)),
        #     np.float32(np.random.uniform(-3.14, 3.14)),
        #     np.float32(np.random.uniform(-0.524, 0.524)),
        #     np.float32(np.random.uniform(-2.97, 0.174)),
        #     np.float32(np.random.uniform(0.0873, 2.62)),
        #     np.float32(np.random.uniform(-0.785, 0.785))
        # ]

        self.env.physics.data.qpos[:] = [0, 0, 1, 0, 0, 0, 0]
        self.env.physics.data.qvel[:] = [0] * 7
        obs = self.env.physics.data.qpos[:].copy().astype(np.float32)
        return obs, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass
