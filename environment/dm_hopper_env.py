from functools import partial
from dm_control import viewer
from stable_baselines3 import PPO
from models.dm_control_based.hopper import CustomHopperEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

def dm_control_hopper(mode, agent_model):
    env = CustomHopperEnv()
    check_env(env, warn=True)

    # Vectorized environment for stable-baselines3
    vec_env = DummyVecEnv([lambda: env])

    model = PPO('MlpPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=20000)

    policy = model.policy
    # save_model_and_env(policy, env, 'policy_weights_new.pth', 'physics_state.bin')
    # load_model_and_env(policy, env, 'policy_weights.pth', 'physics_state.bin')

    # Add model as a param for the get_policy function
    # viewer.launch(env.env, policy=lambda time_step: get_policy(time_step=time_step, model=model), title="Hopper")
    
