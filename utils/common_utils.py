import os
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import mujoco
import shutil
import numpy as np
import yaml
import csv

# Value from XML definition
GEAR_RATIO = 50

def create_log_and_agent_dirs(LOGS_DIR, AGENTS_DIR):
    if not os.path.exists(AGENTS_DIR):
        os.makedirs(AGENTS_DIR)
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)


def visualize(env, model):
    vec_env = DummyVecEnv([lambda: env])
    obs = vec_env.reset()
    env.render()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        mujoco.mj_step(vec_env.envs[0].model, vec_env.envs[0].data)
        env.viewer.sync()
        time.sleep(0.006)

        if done:
            obs = vec_env.reset()

def dump_csv_data(env, model):
    vec_env = DummyVecEnv([lambda: env])
    obs = vec_env.reset()
    done = False
    
    oneStrideData = {key: value for key, value in vec_env.envs[0].one_stride_data.items() if key != "dataLength"}
    sensor_data_list = [[*vec_env.envs[0].data.sensordata]]
    obs_list = [[*obs[0]]]
    torque_input_list = [vec_env.envs[0].data.ctrl[0] * GEAR_RATIO]
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        mujoco.mj_step(vec_env.envs[0].model, vec_env.envs[0].data)
        obs_list.append([*obs[0]])
        sensor_data_list.append([*vec_env.envs[0].data.sensordata])
        torque_input_list.append(vec_env.envs[0].data.ctrl[0] * GEAR_RATIO)
        time.sleep(0.006)

        if done:
            obs = vec_env.reset()
    
    obs_dict = {f'obs_col{i}': col for i, col in enumerate(zip(*obs_list))}
    sensor_dict = {f'sensor_col{i}': col for i, col in enumerate(zip(*sensor_data_list))}
    oneStrideData['torque_input_list'] = torque_input_list
    combined_dict = {**oneStrideData, **obs_dict, **sensor_dict}
    with open("data/plot_data.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(combined_dict.keys())
        writer.writerows(zip(*combined_dict.values()))

def remove_all_files_in_dir(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def random_float(min=-1, max=1, precision=2):
    return round(np.random.uniform(0.8,3), precision)

def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)
