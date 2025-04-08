from environment.dm_hopper_env import dm_control_hopper
from environment.gym_prosthetic import gym_prosthetic
from environment.gym_hopper import gym_hopper
from environment.gym_prosthetic_suspended import gym_prosthetic_suspended
from utils.common_utils import remove_all_files_in_dir, load_config

def main(env_type="gym_hopper", mode="train", agent_model="", clear_logs=False):
    if clear_logs:
        remove_all_files_in_dir("logs/")
    if env_type == "dm_control_hopper":
        dm_control_hopper(mode, agent_model)
    elif env_type == "gym_hopper":
        gym_hopper(mode, agent_model)
    elif env_type == "gym_prosthetic":
        gym_prosthetic(mode, agent_model)
    elif env_type == "gym_prosthetic_suspended":
        gym_prosthetic_suspended(mode, agent_model)
    else:
        print(f"Unknown environment type: {env_type}")

if __name__ == "__main__":
    config = load_config()
    env = config.get("env")
    mode = config.get("mode")
    agent_model = config.get("agent_model")
    clear_logs = config.get("clear_logs", False)
    main(env, mode, agent_model, clear_logs)
