import psutil
import time
import GPUtil
import pandas as pd

# Create dataframe if not exist
try:
    pd.read_csv('data.csv')
except pd.errors.EmptyDataError:
    df = pd.DataFrame({
        'uptime': [],
        'temp_cpu': [],
        'crit_temp_cpu': [],
        'temp_gpu': [],
        'cpu_usage': [],
        'gpu_usage': [],
        'ram_usage': []
    })
    df.to_csv('data.csv', index=False)

def write(uptime, temp_cpu, crit_temp_cpu, temp_gpu, cpu_usage, gpu_usage, ram_usage):
    df = pd.read_csv('data.csv')
    df.loc[len(df)] = {
        'uptime': uptime,
        'temp_cpu': temp_cpu,
        'crit_temp_cpu': crit_temp_cpu,
        'temp_gpu': temp_gpu,
        'cpu_usage': cpu_usage,
        'gpu_usage': gpu_usage,
        'ram_usage': ram_usage
    }
    df.to_csv('data.csv', index=False)

def read_all() -> list[tuple[int, float, float, float, float, float, float]]:
    df = pd.read_csv('data.csv')
    return [list(df.loc[i]) for i in range(len(df))]

def read_last() -> tuple[int, float, float, float, float, float, float]:
    df = pd.read_csv('data.csv')
    return tuple(df.loc[len(df)-1]) # get last series
     
class State:
    gpu_not_exist_warn_showed = False
s = State()

def analysis():
    uptime = int((time.time() - psutil.boot_time()) * 10**3)
    sensors_temps = psutil.sensors_temperatures()

    curr = [i.current for i in sensors_temps['coretemp']]
    crit = [i.critical for i in sensors_temps['coretemp']]
    
    # Average CPU temperature
    cpu_temp = sum(curr) / len(curr)
    avg_crit = sum(crit) / len(crit)
    
    try:
        gpu_temp = GPUtil.getGPUs()[0].temperature
        gpu_usage = GPUtil.getGPUs()[0].load
    except IndexError:
        if not s.gpu_not_exist_warn_showed:
            print('[WARN] GPU is missing')
            s.gpu_not_exist_warn_showed = True
        gpu_temp = -1
        gpu_usage = -1


    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
 
    write(uptime, cpu_temp, avg_crit, gpu_temp, cpu_usage, gpu_usage, ram_usage)

if __name__ == '__main__':
    analysis()
