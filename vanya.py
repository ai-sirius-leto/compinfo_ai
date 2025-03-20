import sqlite3

conn = sqlite3.connect('vanya.db')
conn.execute('CREATE TABLE IF NOT EXIST vanya (temperature float, processor_usage float, gpu_usage float, ram_usage float, disk_usage float)')
conn.commit()
conn.close()
class vanya:
    def __init__(self, temperature, processor_usage, gpu_usage, ram_usage, disk_usage):
        self.temperature = temperature
        self.processor_usage = processor_usage
        self.gpu_usage = gpu_usage
        self.ram_usage = ram_usage
        self.disk_usage = disk_usage
    def write(self):
        conn = sqlite3.connect('vanya.db')
        conn.execute('INSERT INTO vanya VALUES (?, ?, ?, ?, ?)', (self.temperature, self.processor_usage, self.gpu_usage, self.ram_usage, self.disk_usage))
        conn.commit()
        conn.close()
        
        


def main():
    print(1)
    





main()