from stable_baselines3 import PPO
from utils.common_utils import create_log_and_agent_dirs, load_config, visualize, dump_csv_data
from models.gym_based.prosthetic_leg_suspended import ProstheticEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env

# Modify this to change the training meta info
TOTAL_TIMESTEPS = int(1e10)
REWARD_THRESHOLD = -5
XML_FILE = "assets/prosthetic_gym_suspended.xml"
AGENTS_DIR = "agent"
LOGS_DIR = "logs"

# TODO:
# Create a function to load the matlab experiment data similar to the function load_data_dict_from_file in duong_env.py
# First step: just load one stride of data base on robustFC signal in the function get_stride_info_from_data_dict in duong_env.py
# This stride data inlcude the time data (timeOut)
# Convert this time data to begin at 0
# others will be functions with this time data

# Second step: Save this stride data when setting up the environment (__init__)
# Implement the reset function to load this stride data from the beginning (inital state)
# Implement the step function to take actions based on the time data
# Get mujoco_t data
# Use interpolate to get input data (knee and ankle angles) from the experiment data
# Use mujoco_t to get the state of the robot
# Calculate the reward
## First test: The leg is not in contact with the ground: thigh position is fixed at the initial position
# Control
# At time mujoco_t: reward = -(mujoco_thigh_IMU_X - experiment_thigh_IMU_X)**2 +
# Check the termination
# At time mujoco_t: terminated = mujoco_t >= timeOut
# Return the observation, reward, terminated, truncated, info


def gym_prosthetic_suspended(mode, agent_model):
    if mode == "test":
        print("Running Testing mode with agent: ", agent_model)
        test_hopper(agent_model)
    elif mode == "train":
        print("Running Training")

        train_hopper()


def test_hopper(agent_model):
    if not str(agent_model).__len__:
        print(
            "Please provide a valid path to the trained RL agent to visualise the policy!"
        )
        return 0

    env = ProstheticEnv(XML_FILE=XML_FILE, render_mode="human")
    try:
        # Change the RL algorithm here if neccessary
        model = PPO.load(agent_model, env)
    except:
        print(
            "Unable to load the model! Please provide a valid path to the trained RL agent to visualise the policy!"
        )
        return 0

    visualize(env=env, model=model)


def train_hopper():
    create_log_and_agent_dirs(LOGS_DIR, AGENTS_DIR)
    env = ProstheticEnv(XML_FILE=XML_FILE, render_mode="human")
    check_env(env, warn=True)
    config = load_config()
    seed = config.get("seed")
    resume_training = config.get("resume_training")
    vec_env = DummyVecEnv([lambda: env])

    if resume_training:
        agent_model = config.get("agent_model")
        print("Resuming Training For agent: ", agent_model)
        # Change the RL algorithm here if neccessary
        model = PPO.load(agent_model, env= vec_env, verbose=1, tensorboard_log=LOGS_DIR, seed=seed)
    else: 
        # Change the RL algorithm here if neccessary
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOGS_DIR, seed=seed)
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    eval_env = DummyVecEnv([lambda: env])
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, tb_log_name="gym_prosthetic_PPO")
        model.save(f"{AGENTS_DIR}/PPO_suspended")
    
    except KeyboardInterrupt:
        model.save(f"{AGENTS_DIR}/PPO_suspended_interrupt")

    finally:
        # Data dump for plots
        dump_csv_data(env, model)
        # Visualisation
        visualize(env=env, model=model)
    
