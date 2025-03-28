import numpy as np
from numpy import ndarray
from arbeitAI.ram_usage import predict_ram_if_gpu, predict_ram_if_not_gpu
from arbeitAI.temp_cpu import predict_temp_cpu_if_gpu, predict_temp_cpu_if_not_gpu
from arbeitAI.usage_cpu import predict_usage_cpu_if_gpu, predict_usage_cpu_if_not_gpu

def predict_if_not_gpu(data: ndarray, ut: int):
    future_temp_cpu = np.clip(predict_temp_cpu_if_not_gpu(data), a_min=0, a_max=100)
    future_usage_cpu = np.clip(predict_usage_cpu_if_not_gpu(data), a_min=0, a_max=100)
    future_usage_ram = np.clip(predict_ram_if_not_gpu(data), a_min=0, a_max=100)
    return ut, float(future_temp_cpu[0]), 100, -1, float(future_usage_cpu[0]), -1, float(future_usage_ram[0])

def predict_if_gpu(data: ndarray, ut: int):
    future_temp_cpu = np.clip(predict_temp_cpu_if_gpu(data), a_min=0, a_max=100)
    future_usage_cpu = np.clip(predict_usage_cpu_if_gpu(data), a_min=0, a_max=100)
    future_usage_ram = np.clip(predict_ram_if_gpu(data), a_min=0, a_max=100)
    return ut, float(future_temp_cpu[0]), 100, -1, float(future_usage_cpu[0]), -1, float(future_usage_ram[0])

def predict(data: ndarray, ut: int) -> tuple[int, float, float, float]:
    data = np.delete(data, 2, axis = 1)
    flag_give_gpu = bool(data[1, 2] + 1)

    if not flag_give_gpu:
        data = np.delete(data, 2, axis=1)
        data = np.delete(data, 3, axis=1)
    
    if not flag_give_gpu:
        return predict_if_not_gpu(data, ut)
    else:
        return predict_if_gpu(data, ut)
