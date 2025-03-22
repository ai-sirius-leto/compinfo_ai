import sqlite3
from typing import Any
import psutil
import time
import GPUtil

conn = sqlite3.connect('data.db')
conn.execute('CREATE TABLE IF NOT EXISTS compinfo (uptime int, temp_cpu float, crit_temp_cpu float, temp_gpu float, cpu_usage float, gpu_usage float, ram_usage float)')
conn.commit()
conn.close()

def write(uptime, temp_cpu, crit_temp_cpu, temp_gpu, cpu_usage, gpu_usage, ram_usage):
    with sqlite3.connect('data.db') as conn:
        conn.execute('INSERT INTO compinfo VALUES (?, ?, ?, ?, ?, ?, ?)', (uptime, temp_cpu, crit_temp_cpu, temp_gpu, cpu_usage, gpu_usage, ram_usage))
        conn.commit()

def read_all() -> list[tuple[int, float, float, float, float, float, float]]:
    with sqlite3.connect('data.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM compinfo')
        r = cur.fetchall()
        cur.close()
    return r

def read_last() -> tuple[int, float, float, float, float, float, float]:
    return read_all()[-1]
        
def analys():
    # CPU temperature
    temperature = psutil.sensors_temperatures()

    curr = [i.current for i in temperature['coretemp']]
    crit = [i.critical for i in temperature['coretemp']]
    
    # Average CPU temperature
    avg_curr = sum(curr) / len(curr)
    avg_crit = sum(crit) / len(crit)
    
    temperature_cpu = avg_curr
    
    try:
        #Температура видеокарты
        temperature_gpu = GPUtil.getGPUs()[0].temperature
        #Загрузка видеокарты
        gpu_usage = GPUtil.getGPUs()[0].load
    except IndexError:
        print('Видеокарта не обнаружена')
        temperature_gpu = -1
        gpu_usage = -1


    #Загрузка процессора
    processor_usage = psutil.cpu_percent(interval=1)
    
    #Текущее время работы
    uptime = int((time.time() - psutil.boot_time()) * 10**3)
    
    #Загруженность диска и ОЗУ
    ram_usage = psutil.virtual_memory().percent
 
    write(uptime, temperature_cpu, avg_crit, temperature_gpu, processor_usage, gpu_usage, ram_usage)

if __name__ == '__main__':
    analys()