
import os
import time

def get_cpu_temp():
    if os.name == 'nt': # If windows
        command = "node cpu_temp.js"
        import ctypes
        try:
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", "cmd.exe", f"/c {command}", None, 1
            )
        except Exception as e:
            print("Ошибка:", e)
    else:
        os.system("sudo node cpu_temp.js")
    time.sleep(1)
    
    import ujson
    
    with open('cpu_temp.res') as f:
        r = f.read()
    os.remove('cpu_temp.res')
    return float(r)

print(get_cpu_temp())
