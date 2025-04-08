from stable_baselines3 import PPO
from utils.common_utils import create_log_and_agent_dirs, load_config, visualize
from models.gym_based.prosthetic_leg import ProstheticHopperEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Modify this to change the training meta info
TOTAL_TIMESTEPS = 600_000
TIMESTEPS_PER_BATCH = 100_000
XML_FILE = "assets/prosthetic_gym.xml"
AGENTS_DIR = "agent"
LOGS_DIR = "logs"

def gym_prosthetic(mode, agent_model):
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

    env = ProstheticHopperEnv(XML_FILE=XML_FILE, render_mode="human")
    # Change the RL algorithm here if neccessary
    model = PPO("MlpPolicy", env, verbose=1)
    try:
        model.load(agent_model, env)
    except:
        print(
            "Unable to load the model! Please provide a valid path to the trained RL agent to visualise the policy!"
        )
        return 0

    visualize(env, model)


def train_hopper():
    create_log_and_agent_dirs(LOGS_DIR, AGENTS_DIR)
    env = ProstheticHopperEnv(XML_FILE=XML_FILE, render_mode="human")
    check_env(env, warn=True)
    config = load_config()
    seed = config.get("seed")
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOGS_DIR, seed=seed)
    num_of_iterations = int(TOTAL_TIMESTEPS / TIMESTEPS_PER_BATCH)
    for i in range(num_of_iterations):
        model.learn(
            total_timesteps=TIMESTEPS_PER_BATCH,
            reset_num_timesteps=False,
            tb_log_name=f"gym_prosthetic_PPO",
        )
        # Use time.time() or any unique ID if you don't want to the learned models to be replaced
        model.save(f"{AGENTS_DIR}/PPO_{TIMESTEPS_PER_BATCH*i}")
    # Visualisation
    visualize(env, model)