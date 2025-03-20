import sqlite3
import psutil
import time
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
    
    





main()