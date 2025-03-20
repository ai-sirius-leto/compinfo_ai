import sqlite3
import psutil
import time
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
    processor_usage = psutil.cpu_percent(interval=0.5)

    temperature = psutil.sensors_temperatures()
    pprint(temperature)
    
    # TODO: bleudev: сделать средние
    avg_curr = -1
    avg_crit = -1
    for d in temperature['coretemp']:
        if avg_curr == -1:
            avg_curr = d.current
        else:
            avg_curr = (avg_curr + d.current) / 2
        print(d.current, d.critical)





main()