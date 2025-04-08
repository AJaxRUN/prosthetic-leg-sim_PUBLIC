import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from scipy.interpolate import interp1d
from scipy.io import loadmat

from src.utils.utils import (
    get_interpolate_AB_data,
    get_interpolate_AB_data_full_gait_cycle,
    load_AB_averages_data,
    load_action_range_from_csv,
    normalize_and_rescale,
    rl_normalize_data,
    rl_scale_data,
)

EPS = 1e-3

data_dict_max = {}
data_dict_min = {}
data_dict_max["stance_phase"] = 1
data_dict_min["stance_phase"] = 0
data_dict_max["swing_phase"] = 1
data_dict_min["swing_phase"] = 0
data_dict_max["knee_angle"] = 90
data_dict_min["knee_angle"] = 0
data_dict_max["ankle_angle"] = 45
data_dict_min["ankle_angle"] = -45
data_dict_max["knee_vel"] = 1000
data_dict_min["knee_vel"] = -1000
data_dict_max["ankle_vel"] = 1000
data_dict_min["ankle_vel"] = -1000
data_dict_max["thigh_angle"] = 180
data_dict_min["thigh_angle"] = -180
data_dict_max["thigh_vel"] = 180
data_dict_min["thigh_vel"] = -180
data_dict_max["knee_des"] = 90
data_dict_min["knee_des"] = 0
data_dict_max["ankle_des"] = 45
data_dict_min["ankle_des"] = -45
data_dict_max["knee_angle_AB"] = 90
data_dict_min["knee_angle_AB"] = 0
data_dict_max["ankle_angle_AB"] = 45
data_dict_min["ankle_angle_AB"] = -45
data_dict_max["knee_vel_des"] = 1000
data_dict_min["knee_vel_des"] = -1000
data_dict_max["ankle_vel_des"] = 1000
data_dict_min["ankle_vel_des"] = -1000
data_dict_max["knee_stiffness"] = 1.0
data_dict_min["knee_stiffness"] = 0.0
data_dict_max["knee_eqAngle"] = 1.0
data_dict_min["knee_eqAngle"] = 0.0
data_dict_max["knee_damping"] = 1.0
data_dict_min["knee_damping"] = 0.0
data_dict_max["ankle_stiffness"] = 1.0
data_dict_min["ankle_stiffness"] = 0.0
data_dict_max["ankle_eqAngle"] = 1.0
data_dict_min["ankle_eqAngle"] = 0.0
data_dict_max["ankle_damping"] = 1.0
data_dict_min["ankle_damping"] = 0.0
data_dict_max["knee_Kp"] = 1.0
data_dict_min["knee_Kp"] = 0.0
data_dict_max["knee_Kd"] = 1.0
data_dict_min["knee_Kd"] = 0.0
data_dict_max["ankle_Kp"] = 1.0
data_dict_min["ankle_Kp"] = 0.0
data_dict_max["ankle_Kd"] = 1.0
data_dict_min["ankle_Kd"] = 0.0
data_dict_max["knee_torque"] = 100
data_dict_min["knee_torque"] = -100
data_dict_max["ankle_torque"] = 100
data_dict_min["ankle_torque"] = -100
data_dict_max["knee_torque_AB"] = 100
data_dict_min["knee_torque_AB"] = -100
data_dict_max["ankle_torque_AB"] = 100
data_dict_min["ankle_torque_AB"] = -100
data_dict_max["CoP"] = 0.1
data_dict_min["CoP"] = -0.1
data_dict_max["knee_torque_diff"] = 100
data_dict_min["knee_torque_diff"] = -100
data_dict_max["ankle_torque_diff"] = 100
data_dict_min["ankle_torque_diff"] = -100
data_dict_max["robustFC"] = 1
data_dict_min["robustFC"] = 0


def get_reward_from_dict(data_dict, start_index, window_size):
    # If data_dict don't have the 'reward' key, breakpoint
    if "reward" not in data_dict.keys():
        breakpoint()
    if window_size == 1:
        this_reward = data_dict["reward"][start_index]
    else:
        this_reward = np.mean(data_dict["reward"][start_index : start_index + window_size])

    return this_reward


def get_sample_obs_data_from_dict(data_dict, start_index, window_size):
    # keys = ['stance_phase', 'swing_phase', 'knee_angle', 'ankle_angle',
    #         'knee_des', 'ankle_des', 'knee_vel', 'ankle_vel',
    #         'knee_vel_des', 'ankle_vel_des', 'knee_torque', 'ankle_torque',
    #         'CoP', 'thigh_angle', 'thigh_vel']
    # keys = ['stance_phase', 'swing_phase', 'knee_angle', 'ankle_angle',
    #         'knee_des', 'ankle_des', 'knee_vel', 'ankle_vel']
    keys = [
        "stance_phase",
        "swing_phase",
        "knee_angle",
        "ankle_angle",
        "knee_des",
        "ankle_des",
        "knee_vel",
        "ankle_vel",
        "knee_vel_des",
        "ankle_vel_des",
        "knee_torque",
        "ankle_torque",
        "knee_angle_AB",
        "ankle_angle_AB",
        "knee_torque_AB",
        "ankle_torque_AB",
        "CoP",
        "thigh_angle",
        "thigh_vel",
        "robustFC",
    ]
    if start_index is None:
        this_obs = np.array([data_dict[key] for key in keys]).reshape((-1, window_size))
    else:
        this_obs = np.array([data_dict[key][start_index : start_index + window_size] for key in keys]).reshape(
            (-1, window_size)
        )
    return this_obs


def get_sample_actions_data_from_dict(data_dict, start_index, window_size):
    keys = [
        "knee_stiffness",
        "knee_eqAngle",
        "knee_damping",
        "ankle_stiffness",
        "ankle_eqAngle",
        "ankle_damping",
        "knee_Kp",
        "knee_Kd",
        "ankle_Kp",
        "ankle_Kd",
    ]
    if start_index is None:
        this_actions = np.array([data_dict[key] for key in keys]).reshape((-1, window_size))
    else:
        this_actions = np.array([data_dict[key][start_index : start_index + window_size] for key in keys]).reshape(
            (-1, window_size)
        )
    return this_actions


def get_sample_rl_data_from_dict(data_dict, start_index, window_size, step_size):
    this_obs = get_sample_obs_data_from_dict(data_dict, start_index, window_size)
    this_actions = get_sample_actions_data_from_dict(data_dict, start_index, window_size)
    this_reward = get_reward_from_dict(data_dict, start_index, window_size)
    this_done = data_dict["dones"][start_index + window_size - 1]
    if start_index + step_size + window_size <= len(data_dict["stance_phase"]):
        this_next_obs = get_sample_obs_data_from_dict(data_dict, start_index + step_size, window_size)
    else:
        this_next_obs = this_obs
    return this_obs, this_actions, this_next_obs, this_reward, this_done


def load_data_dict_from_file(filename, desample_factor=1):
    """
    Processes the data from the given filename.

    Parameters:
    - filename: path to the data file.

    Returns:
    Dictionary containing the processed data.
    """
    # Load the dataset
    data = loadmat(filename)

    # Extracting data
    stance_phase = data["Walking"]["stancePhaseEstimate"][0, 0].ravel()
    phase_variable = data["Walking"]["phaseEstimate"][0, 0].ravel()
    robustFC = data["Common"]["robustFC"][0, 0].ravel()

    # Assuming data['Walking'] is a structured array
    if "swingPhaseEstimate" in data["Walking"].dtype.fields:
        swing_phase = data["Walking"]["swingPhaseEstimate"][0, 0].ravel()
    else:
        swing_phase = data["Walking"]["inStance"][0, 0].ravel()

    # State
    knee_vel = data["Common"]["kneeVel"][0, 0].ravel()
    ankle_vel = data["Common"]["ankleVel"][0, 0].ravel()
    knee_angle = data["Knee_Encoder"]["Angle"][0, 0].ravel()
    ankle_angle = data["Ankle_Encoder"]["Angle"][0, 0].ravel()
    knee_des = data["Walking"]["knee_des"][0, 0].ravel()
    ankle_des = data["Walking"]["ankle_des"][0, 0].ravel()
    # knee_vel_des = ddt(knee_des, data['Common']['dt'][0, 0].ravel())
    # ankle_vel_des = ddt(ankle_des, data['Common']['dt'][0, 0].ravel())
    knee_vel_des = data["Walking"]["kneeVel_des"][0, 0].ravel()
    ankle_vel_des = data["Walking"]["ankleVel_des"][0, 0].ravel()

    # Control Inputs
    CoP = data["Common"]["CoP"][0, 0].ravel()
    knee_torque = data["Walking"]["kneeTorqueCommandStance"][0, 0].ravel()
    ankle_torque = data["Walking"]["ankleTorqueCommandStance"][0, 0].ravel()
    thigh_angle = data["Thigh_IMU"]["Theta_X"][0, 0].ravel()
    thigh_vel = data["Thigh_IMU"]["Gyro_X"][0, 0].ravel()

    # Torque diff
    knee_torque_diff = np.diff(knee_torque, prepend=knee_torque[0])
    ankle_torque_diff = np.diff(ankle_torque, prepend=ankle_torque[0])

    # Impedance Parameters
    knee_stiffness = data["Walking"]["kneeStiffness"][0, 0].ravel()
    knee_damping = data["Walking"]["kneeDamping"][0, 0].ravel()
    knee_eqAngle = data["Walking"]["kneeEqAngle"][0, 0].ravel()
    ankle_stiffness = data["Walking"]["ankleStiffness"][0, 0].ravel()
    ankle_damping = data["Walking"]["ankleDamping"][0, 0].ravel()
    ankle_eqAngle = data["Walking"]["ankleEqAngle"][0, 0].ravel()

    # PD gains
    knee_Kp = data["Walking"]["kp_knee_swing"][0, 0].ravel()
    knee_Kd = data["Walking"]["kd_knee_swing"][0, 0].ravel()
    ankle_Kp = data["Walking"]["kp_ankle_swing"][0, 0].ravel()
    ankle_Kd = data["Walking"]["kd_ankle_swing"][0, 0].ravel()

    first_index = np.where(stance_phase == 1)[0][0]
    last_index = np.where(stance_phase == 1)[0][-1]

    data_dict = {}
    data_dict["phase_variable"] = phase_variable[first_index:last_index]
    data_dict["stance_phase"] = stance_phase[first_index:last_index]
    data_dict["swing_phase"] = swing_phase[first_index:last_index]
    data_dict["knee_angle"] = knee_angle[first_index:last_index]
    data_dict["ankle_angle"] = ankle_angle[first_index:last_index]
    data_dict["knee_des"] = knee_des[first_index:last_index]
    data_dict["ankle_des"] = ankle_des[first_index:last_index]
    data_dict["knee_vel"] = knee_vel[first_index:last_index]
    data_dict["ankle_vel"] = ankle_vel[first_index:last_index]
    data_dict["knee_vel_des"] = knee_vel_des[first_index:last_index]
    data_dict["ankle_vel_des"] = ankle_vel_des[first_index:last_index]
    data_dict["CoP"] = CoP[first_index:last_index]
    data_dict["knee_torque"] = knee_torque[first_index:last_index]
    data_dict["knee_torque_diff"] = knee_torque_diff[first_index:last_index]
    data_dict["ankle_torque"] = ankle_torque[first_index:last_index]
    data_dict["ankle_torque_diff"] = ankle_torque_diff[first_index:last_index]
    data_dict["thigh_angle"] = thigh_angle[first_index:last_index]
    data_dict["thigh_vel"] = thigh_vel[first_index:last_index]

    data_dict["phase_stance_swing"] = data_dict["stance_phase"] + data_dict["swing_phase"]
    data_dict["robustFC"] = robustFC[first_index:last_index]

    ##
    new_action_ranges = load_action_range_from_csv("initial_model/model_gains_range.csv")
    old_action_ranges = load_action_range_from_csv("initial_model/model_gains_range_zeroToOne.csv")

    # PD gains collected with Shihao was not normalized. We need to normalize them to the range [0, 1] with old action ranges
    data_dict["knee_Kp"] = rl_normalize_data(
        knee_Kp[first_index:last_index],
        old_action_ranges[1, 6],
        old_action_ranges[0, 6],
        0,
    )
    data_dict["knee_Kd"] = rl_normalize_data(
        knee_Kd[first_index:last_index],
        old_action_ranges[1, 7],
        old_action_ranges[0, 7],
        0,
    )
    data_dict["ankle_Kp"] = rl_normalize_data(
        ankle_Kp[first_index:last_index],
        old_action_ranges[1, 8],
        old_action_ranges[0, 8],
        0,
    )
    data_dict["ankle_Kd"] = rl_normalize_data(
        ankle_Kd[first_index:last_index],
        old_action_ranges[1, 9],
        old_action_ranges[0, 9],
        0,
    )

    # Renormalize the data to the new action ranges [-1, 1]: Data collected with Shihao
    # First need to scale the data back to the original values with old action ranges
    # Then normalize the data to the new action ranges
    data_dict["knee_stiffness"] = rl_scale_data(
        knee_stiffness[first_index:last_index],
        old_action_ranges[1, 0],
        old_action_ranges[0, 0],
        0,
    )
    data_dict["knee_eqAngle"] = rl_scale_data(
        knee_eqAngle[first_index:last_index],
        old_action_ranges[1, 1],
        old_action_ranges[0, 1],
        0,
    )
    data_dict["knee_damping"] = rl_scale_data(
        knee_damping[first_index:last_index],
        old_action_ranges[1, 2],
        old_action_ranges[0, 2],
        0,
    )
    data_dict["ankle_stiffness"] = rl_scale_data(
        ankle_stiffness[first_index:last_index],
        old_action_ranges[1, 3],
        old_action_ranges[0, 3],
        0,
    )
    data_dict["ankle_eqAngle"] = rl_scale_data(
        ankle_eqAngle[first_index:last_index],
        old_action_ranges[1, 4],
        old_action_ranges[0, 4],
        0,
    )
    data_dict["ankle_damping"] = rl_scale_data(
        ankle_damping[first_index:last_index],
        old_action_ranges[1, 5],
        old_action_ranges[0, 5],
        0,
    )
    data_dict["knee_Kp"] = rl_scale_data(data_dict["knee_Kp"], old_action_ranges[1, 6], old_action_ranges[0, 6], 0)
    data_dict["knee_Kd"] = rl_scale_data(data_dict["knee_Kd"], old_action_ranges[1, 7], old_action_ranges[0, 7], 0)
    data_dict["ankle_Kp"] = rl_scale_data(data_dict["ankle_Kp"], old_action_ranges[1, 8], old_action_ranges[0, 8], 0)
    data_dict["ankle_Kd"] = rl_scale_data(data_dict["ankle_Kd"], old_action_ranges[1, 9], old_action_ranges[0, 9], 0)

    # Normalize the data to the new action ranges [0, 1]
    data_dict["knee_stiffness"] = rl_normalize_data(
        data_dict["knee_stiffness"], new_action_ranges[1, 0], new_action_ranges[0, 0], 0
    )
    data_dict["knee_eqAngle"] = rl_normalize_data(
        data_dict["knee_eqAngle"], new_action_ranges[1, 1], new_action_ranges[0, 1], 0
    )
    data_dict["knee_damping"] = rl_normalize_data(
        data_dict["knee_damping"], new_action_ranges[1, 2], new_action_ranges[0, 2], 0
    )
    data_dict["ankle_stiffness"] = rl_normalize_data(
        data_dict["ankle_stiffness"],
        new_action_ranges[1, 3],
        new_action_ranges[0, 3],
        0,
    )
    data_dict["ankle_eqAngle"] = rl_normalize_data(
        data_dict["ankle_eqAngle"], new_action_ranges[1, 4], new_action_ranges[0, 4], 0
    )
    data_dict["ankle_damping"] = rl_normalize_data(
        data_dict["ankle_damping"], new_action_ranges[1, 5], new_action_ranges[0, 5], 0
    )
    data_dict["knee_Kp"] = rl_normalize_data(data_dict["knee_Kp"], new_action_ranges[1, 6], new_action_ranges[0, 6], 0)
    data_dict["knee_Kd"] = rl_normalize_data(data_dict["knee_Kd"], new_action_ranges[1, 7], new_action_ranges[0, 7], 0)
    data_dict["ankle_Kp"] = rl_normalize_data(
        data_dict["ankle_Kp"], new_action_ranges[1, 8], new_action_ranges[0, 8], 0
    )
    data_dict["ankle_Kd"] = rl_normalize_data(
        data_dict["ankle_Kd"], new_action_ranges[1, 9], new_action_ranges[0, 9], 0
    )

    return data_dict


def desample_data_dict(data_dict, desample_factor=1):
    data_dict = {k: v[::desample_factor] for k, v in data_dict.items()}
    return data_dict


def get_stride_info_from_data_dict(data_dict):
    robustFC = data_dict["robustFC"]
    diff = np.diff(robustFC)
    reset_indices = np.where(diff == 1)[0]
    # Begin stance phase by removing the last reset_indices
    stride_start_indices = reset_indices[:-1] + 1
    stride_end_indices = reset_indices[1:] + 1
    stride_end_stance = np.where(diff == -1)[0]

    num_strides = len(stride_start_indices)

    stride_info = {
        "stride_start_indices": stride_start_indices,
        "stride_end_indices": stride_end_indices,
        "stride_end_stance": stride_end_stance,
        "num_strides": num_strides,
    }

    return stride_info


def get_reward_from_dict_full_stride(data_dict, stride_indx, user_mass=80):
    robustFC = data_dict["robustFC"][stride_indx, :]
    joint_types = ["knee", "ankle"]
    quantities = ["angle", "torque"]
    reward_weights = {
        "knee": {
            "angle": 1,  # 100,
            "torque": 1,  # 0.2, #0.5
        },
        "ankle": {
            "angle": 1,  # 100,
            "torque": 1,  # 0.2, #0.5
        },
    }
    # Initialize dictionaries for actual data
    reward_data = {}
    this_reward = np.zeros(data_dict["stance_phase"].shape[1])

    # Populate actual data dictionary and normalize
    for joint in joint_types:
        reward_data[joint] = {}

        for quantity in quantities:
            # Extract AB data directly using the constructed key
            AB_data = data_dict[f"{joint}_{quantity}_AB"][stride_indx, :]
            # Extract actual data and normalize it
            Leg_data = data_dict[f"{joint}_{quantity}"][stride_indx, :] / (user_mass if quantity == "torque" else 1)
            # Normalize the data to the range [0, 1]
            Leg_data = rl_normalize_data(Leg_data, np.min(AB_data), np.max(AB_data), 0)
            AB_data = rl_normalize_data(AB_data, np.min(AB_data), np.max(AB_data), 0)

            # Compute square errors with two options
            # reward_data[joint][quantity] = -(AB_data - Leg_data)
            reward_data[joint][quantity] = -((AB_data - Leg_data) ** 2)
            # reward_data[joint][quantity] = np.exp(-(AB_data - Leg_data)**2)
            # Note: The Leg_data can be negative, so AB_data**2 - (AB_data - Leg_data)**2 is can be negative
            # Apply robustFC condition to torque errors
            if quantity == "torque":
                reward_data[joint][quantity][robustFC == 0] = 0

            # Add reward weights
            reward_data[joint][quantity] = reward_weights[joint][quantity] * reward_data[joint][quantity]
            this_reward += reward_data[joint][quantity]

    return this_reward, reward_data


def normalize_data_dict_to_strides(data_dict, num_points, AB_data=None, user_mass=80):
    stride_info = get_stride_info_from_data_dict(data_dict)

    # Pre-allocate dictionary to store interpolated data
    empty_nan = np.full((stride_info["num_strides"], num_points), np.nan)
    data_dict_interp = {key: empty_nan.copy() for key in data_dict.keys()}

    data_dict_interp["reward"] = empty_nan.copy()
    for joint in ["knee", "ankle"]:
        for key in ["angle", "torque"]:
            data_dict_interp[f"reward_{joint}_{key}"] = empty_nan.copy()

    AB_keys = ["knee_angle_AB", "ankle_angle_AB", "knee_torque_AB", "ankle_torque_AB"]
    for key in AB_keys:
        data_dict_interp[key] = empty_nan.copy()

    normalized_time = np.linspace(0, 1, num_points)
    for i in range(stride_info["num_strides"]):
        start_index = stride_info["stride_start_indices"][i]
        end_index = stride_info["stride_end_indices"][i]
        original_time = np.linspace(0, 1, end_index - start_index)
        for key, value in data_dict.items():
            interpolator = interp1d(original_time, value[start_index:end_index], kind="linear")
            value_interp = interpolator(normalized_time)
            data_dict_interp[key][i, :] = value_interp.copy()

        # Add AB data to the data_dict_interp
        if AB_data is not None:
            for joint in ["knee", "ankle"]:
                for key in ["angle", "torque"]:
                    AB_key = f"{joint}_{key}_AB"
                    AB_data_interp = AB_data["interpolators"][AB_key](normalized_time)
                    data_dict_interp[AB_key][i, :] = AB_data_interp

        # Add the reward to the data_dict_interp
        # Plot comparison between AB and actual data
        reward, reward_components = get_reward_from_dict_full_stride(data_dict_interp, i, user_mass)
        data_dict_interp["reward"][i, :] = reward
        for joint in ["knee", "ankle"]:
            for key in ["angle", "torque"]:
                data_dict_interp[f"reward_{joint}_{key}"][i, :] = reward_components[joint][key]

    # Check if stance_phase and swing_phase in value_interp in range [0, 1]
    if (
        (data_dict_interp["stance_phase"] < 0).any()
        or (data_dict_interp["stance_phase"] > 1).any()
        or (data_dict_interp["swing_phase"] < 0).any()
        or (data_dict_interp["swing_phase"] > 1).any()
    ):
        breakpoint()

    for key in data_dict_interp:
        data_dict_interp[key] = data_dict_interp[key][~np.isnan(data_dict_interp[key]).any(axis=1)].flatten()

    return data_dict_interp


def plot_strides_from_data_dict(data_dict, stride_info, stride_index):
    stride_start_indices = stride_info["stride_start_indices"]
    stride_end_indices = stride_info["stride_end_indices"]
    num_strides = stride_info["num_strides"]

    stride_start_index = stride_start_indices[stride_index]
    stride_end_index = stride_end_indices[stride_index]
    phase_stance_swing = data_dict["phase_stance_swing"]
    phase_stance_swing_stride = phase_stance_swing[stride_start_index:stride_end_index]

    # Plot the data_dict
    plt.figure()
    plt.plot(data_dict["knee_angle"][stride_start_index:stride_end_index], label="Knee Angle")
    plt.plot(
        data_dict["ankle_angle"][stride_start_index:stride_end_index],
        label="Ankle Angle",
    )
    plt.plot(
        data_dict["knee_des"][stride_start_index:stride_end_index],
        label="Knee Desired Angle",
    )
    plt.plot(
        data_dict["ankle_des"][stride_start_index:stride_end_index],
        label="Ankle Desired Angle",
    )
    plt.xlabel("Phase [%]")
    plt.ylabel("Angle [deg]")
    plt.title("Knee and Ankle Angle")
    plt.legend()

    plt.figure()
    plt.plot(
        data_dict["knee_vel"][stride_start_index:stride_end_index],
        label="Knee Velocity",
    )
    plt.plot(
        data_dict["ankle_vel"][stride_start_index:stride_end_index],
        label="Ankle Velocity",
    )
    plt.plot(
        data_dict["knee_vel_des"][stride_start_index:stride_end_index],
        label="Knee Desired Velocity",
    )
    plt.plot(
        data_dict["ankle_vel_des"][stride_start_index:stride_end_index],
        label="Ankle Desired Velocity",
    )
    plt.xlabel("Phase [%]")
    plt.ylabel("Velocity [deg/s]")