import numpy as np
from numpy import ndarray
from arbeitAI.ram_usage import predict_ram_if_not_gpu
from arbeitAI.temp_cpu import predict_temp_cpu_if_not_gpu
from arbeitAI.usage_cpu import predict_usage_cpu_if_not_gpu


def predict(data: ndarray, ut: int) -> tuple[int, float, float, float]:
    future_temp_cpu = np.clip(predict_temp_cpu_if_not_gpu(data), a_min=0, a_max=100)
    future_usage_cpu = np.clip(predict_usage_cpu_if_not_gpu(data), a_min=0, a_max=100)

    future_usage_ram = np.clip(predict_ram_if_not_gpu(data), a_min=0, a_max=100)
    return ut, float(future_temp_cpu[0]), float(future_usage_cpu[0, 0]), float(future_usage_ram[0, 0])
