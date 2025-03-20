import sqlite3
import psutil
import time
import GPUtil
from pprint import pprint
conn = sqlite3.connect('vanya.db')
conn.execute('CREATE TABLE IF NOT EXISTS vanya (time int, temperature_cpu float, temperature_gpu float, processor_usage float, gpu_usage float, ram_usage float, disk_usage float)')
conn.commit()
conn.close()

def write(time, temperature_cpu, temperature_gpu, processor_usage, gpu_usage, ram_usage, disk_usage):
    conn = sqlite3.connect('vanya.db')
    conn.execute('INSERT INTO vanya VALUES (?, ?, ?, ?, ?, ?, ?)', (time, temperature_cpu, temperature_gpu, processor_usage, gpu_usage, ram_usage, disk_usage))
    conn.commit()
    conn.close()
        
        


def main():
    # CPU temperature
    temperature = psutil.sensors_temperatures()

    curr = [i.current for i in temperature['coretemp']]
    crit = [i.critical for i in temperature['coretemp']]
    
    # Average CPU temperature
    avg_curr = sum(curr) / len(curr)
    avg_crit = sum(crit) / len(crit)
    
    print(avg_curr, avg_crit)
    
    try:
        #Температура видеокарты
        temperature_gpu = GPUtil.getGPUs()[0].temperature
        #Загрузка видеокарты
        gpu_usage = GPUtil.getGPUs()[0].load
    except IndexError:
        print('Видеокарта не обнаружена')
        temperature_gpu = 0
        gpu_usage = 0


    #Загрузка процессора
    processor_usage = psutil.cpu_percent(interval=1)
    
    t = int((time.time() - psutil.boot_time()) * 10**3)
    ram_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    write(t, avg_curr, temperature_gpu, processor_usage, gpu_usage, ram_usage, disk_usage)


main()