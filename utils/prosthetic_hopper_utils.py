from scipy.io import loadmat
from utils.common_utils import load_config
import numpy as np
import math

def load_mat_data():
    config = load_config()
    mat_data_path = config.get("mat_data")
    mat_data = loadmat(mat_data_path)
    return mat_data


def get_stride(mat_data, stride_start_num = 10, stride_end_num = None):
    stride_info = get_stride_info_from_data_dict(mat_data)

    start_index = stride_info["stride_start_indices"][stride_start_num - 1]
    end_index = stride_info["stride_start_indices"][stride_start_num if stride_end_num == None else stride_end_num] + 1

    timeOut =  np.array(mat_data["Common"]["timeOut"][0,0].ravel(), np.float64)

    thighIMUData =  mat_data["Thigh_IMU"]
    thighIMUData_Theta_X = np.array(thighIMUData["Theta_X"][0,0].ravel(), np.float64)
    kneeEncoder = np.array(mat_data["Knee_Encoder"]["Angle"][0,0].ravel(), np.float64)
    ankleEncoder =  np.array(mat_data["Ankle_Encoder"]["Angle"][0,0].ravel(), np.float64)

    #convert values to radians to match mujoco
    thighIMUData_Theta_X = [math.radians(-i) for i in thighIMUData_Theta_X]
    kneeEncoder = [math.radians(-i) for i in kneeEncoder]
    ankleEncoder =  [math.radians(i) for i in ankleEncoder]

    stride_data = {
        "timeOut": timeOut[start_index : end_index],
        "kneeAngles": kneeEncoder[start_index: end_index],
        "ankleAngles": ankleEncoder[start_index: end_index],
        "thighIMU_Theta_X": thighIMUData_Theta_X[start_index: end_index],
        "dataLength": end_index - start_index
    }

    return stride_data

def get_stride_info_from_data_dict(mat_data):
    robustFC = np.array(mat_data["Common"]["robustFC"][0,0].ravel(), np.int32)

    diff = np.diff(robustFC)
    reset_indices = np.where(diff == 1)[0]
    
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

def compute_squared_error(actual_value, measured_value):
    error_sq = pow((math.degrees(actual_value) - math.degrees(measured_value)),2)
    # loss = error_sq / (1 + error_sq)
    return -error_sq
