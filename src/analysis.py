import psutil
import time
import GPUtil
import pandas as pd
import os

from arbeitAI.predict import DATA_PATH

# Create dataframe if not exist

try:
    pd.read_csv(DATA_PATH)
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
    df.to_csv(DATA_PATH, index=False)

def write(uptime, temp_cpu, crit_temp_cpu, temp_gpu, cpu_usage, gpu_usage, ram_usage):
    df = pd.read_csv(DATA_PATH)
    df.loc[len(df)] = {
        'uptime': uptime,
        'temp_cpu': temp_cpu,
        'crit_temp_cpu': crit_temp_cpu,
        'temp_gpu': temp_gpu,
        'cpu_usage': cpu_usage,
        'gpu_usage': gpu_usage,
        'ram_usage': ram_usage
    }
    df.to_csv(DATA_PATH, index=False)

def read_all() -> list[tuple[int, float, float, float, float, float, float]]:
    df = pd.read_csv(DATA_PATH)
    return [list(df.loc[i]) for i in range(len(df))]

def read_last() -> tuple[int, float, float, float, float, float, float]:
    df = pd.read_csv(DATA_PATH)
    return tuple(df.loc[len(df)-1]) # get last series
     
class State:
    gpu_not_exist_warn_showed = False
s = State()

def get_cpu_temp_js() -> float:
    if os.name == 'nt': # If windows
        command = "node cpu_temp.js"
        import ctypes
        try:
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", "cmd.exe", f"/c -WindowStyle Hidden {command}", None, 1
            )
        except Exception as e:
            print("Ошибка:", e)
    else:
        os.system("sudo node cpu_temp.js")
    # time.sleep(1)
    
    import ujson
    
    with open('cpu_temp.res', encoding='utf8') as f:
        r = f.read()
    # os.remove('cpu_temp.res')
    return float(r)

def analysis():
    uptime = int((time.time() - psutil.boot_time()) * 10**3)

    if os.name == 'nt':
        # Windows
        cpu_temp = get_cpu_temp_js()
        avg_crit = 100
    else:
        # Other (Linux + MacOs)
        sensors_temps = psutil.sensors_temperatures()
        
        curr = [i.current for i in sensors_temps['coretemp']]
        crit = [i.critical for i in sensors_temps['coretemp']]
        
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
