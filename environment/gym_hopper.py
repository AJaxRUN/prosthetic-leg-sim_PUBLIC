from stable_baselines3 import PPO
from utils.common_utils import create_log_and_agent_dirs, visualize
from models.gym_based.hopper import CustomHopperEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import torch

# Modify this to change the training meta info
TOTAL_TIMESTEPS = 100_000
TIMESTEPS_PER_BATCH = 10_000
XML_FILE = "assets/hopper_gym.xml"
AGENTS_DIR = "agent"
LOGS_DIR = "logs"


def gym_hopper(mode, agent_model):
    if mode == "test":
        test_hopper(agent_model)
    elif mode == "train":
        train_hopper()


def test_hopper(agent_model):
    if not str(agent_model).__len__:
        print(
            "Please provide a valid path to the trained RL agent to visualise the policy!"
        )
        return 0

    env = CustomHopperEnv(XML_FILE=XML_FILE, render_mode="human")
    # Change the RL algorithm here if neccessary
    model = PPO("MlpPolicy", env, verbose=1)
    try:
        model.policy.load_state_dict(torch.load(agent_model))
    except:
        print(
            "Unable to load the model! Please provide a valid path to the trained RL agent to visualise the policy!"
        )
        return 0

    visualize(env, model)


def train_hopper():
    create_log_and_agent_dirs(LOGS_DIR, AGENTS_DIR)
    env = CustomHopperEnv(XML_FILE=XML_FILE, render_mode="human")
    check_env(env, warn=True)
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=1)
    num_of_iterations = int(TOTAL_TIMESTEPS / TIMESTEPS_PER_BATCH)
    for i in range(num_of_iterations):
        model.learn(
            total_timesteps=TIMESTEPS_PER_BATCH,
            reset_num_timesteps=False,
            tb_log_name=f"gym_hopper_PPO",
        )
        # Use time.time() or any unique ID if you don't want to the learned models to be replaced
        model.save(f"{AGENTS_DIR}/PPO_{TIMESTEPS_PER_BATCH*i}")
    # Visualisation
    visualize(env, model)
