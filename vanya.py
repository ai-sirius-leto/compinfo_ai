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
    processor_usage = psutil.cpu_percent(interval=0.5)
    temperature = psutil.sensors_temperatures()

    curr = [i.current for i in temperature['coretemp']]
    crit = [i.critical for i in temperature['coretemp']]
    
    # Average CPU temperature
    avg_curr = sum(curr) / len(curr)
    avg_crit = sum(crit) / len(crit)
    
    print(avg_curr, avg_crit, processor_usage)
    
    #Температура видеокарты
    temperature_gpu = GPUtil.getGPUs()[0].temperature
    
    #Загрузка видеокарты
    gpu_usage = GPUtil.getGPUs()[0].load


    #Загрузка процессора
    processor_usage = psutil.cpu_percent(interval=1)


main()