from analysis import read_all, read_last
from arbeitAI import predict_temp_cpu_if_not_gpu, predict_usage_cpu_if_not_gpu, predict_ram_if_not_gpu, predict_usage_cpu_if_gpu, predict_ram_if_gpu, predict_temp_gpu, predict_temp_cpu_if_gpu, predict_usage_gpu
import sqlite3
import numpy as np
import sqlite3

from arbeitAI.temp_cpu import learn_model_temp_cpu_if_gpu
from arbeitAI.usage_cpu import learn_model_cpu_usage_if_gpu
from arbeitAI.utils import to_model
def arbeit_model():
    sl = np.array(read_all())


    kritical_temperature_cpu = sl[1, 2]
    
    sl = np.delete(sl, 2, axis = 1)
    flag_give_gpu = bool(sl[1, 2] + 1)

    if not flag_give_gpu:
        sl = np.delete(sl, 2, axis=1)
        sl = np.delete(sl, 3, axis=1)
    
    # Таблица без GPU После изменений принимает вид:
    
    # Uptime - Temp_CPU - CPU_usage - RAM_usage 
    
    if not flag_give_gpu:
        all_predicter_if_not_gpu(sl)
    else:
        all_predicter_if_gpu(sl)
    

def all_predicter_if_not_gpu(sl):
    future_temp_cpu = predict_temp_cpu_if_not_gpu(sl)
    
    future_usage_cpu = predict_usage_cpu_if_not_gpu(sl)
    
    future_usage_ram = predict_ram_if_not_gpu(sl)
    
    tutime = read_last()[0]
    loh = [tutime+1000, float(future_temp_cpu[0]), float(future_usage_cpu[0]), float(future_usage_ram[0])]
    tutime += 1000
    for _ in range(6000):
        print(loh)
        sl = np.vstack((sl, loh))
        
        future_temp_cpu = np.clip(predict_temp_cpu_if_not_gpu(sl), a_min=0, a_max=100)
    
        future_usage_cpu = np.clip(predict_usage_cpu_if_not_gpu(sl), a_min=0,a_max=100)
    
        future_usage_ram = np.clip(predict_ram_if_not_gpu(sl),a_min=0,a_max=100)
        
        loh = [tutime+1000, float(future_temp_cpu[0]), float(future_usage_cpu[0]), float(future_usage_ram[0])]

        tutime+=1000
def all_predicter_if_gpu(sl):
    print(predict_temp_cpu_if_gpu(sl))
    print(predict_usage_cpu_if_gpu(sl))
    print(predict_ram_if_gpu(sl))
    print(predict_usage_gpu(sl))
    print(predict_temp_gpu(sl))
    
arbeit_model()