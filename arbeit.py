from pprint import pprint
import numpy as np
import pandas as pd
import flet as ft
import sqlite3
import time
import psutil
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
def to_model(sp: np.ndarray):
    
    if len(sp.shape) == 1:
        sp = [sp]
    #sp = np.ndarray(sp)
    return Normalizer().fit(sp).transform(sp)

def arbeit_model():
    with sqlite3.connect('data.db') as conn:
        cur = conn.cursor()
        cur.execute("select max(uptime) from compinfo")
        ut = cur.fetchone()[0]
        cur.execute("select * from compinfo where uptime > ?", (ut - 120000,))
        r = cur.fetchall()
        cur.close()
    
    sl = np.array(r)
    kritical_temperature_cpu = sl[1, 2]
    
    sl = np.delete(sl, 2, axis = 1)
    flag_give_gpu = bool(sl[1, 2] + 1)

    if not flag_give_gpu:
        sl = np.delete(sl, 2, axis=1)
        sl = np.delete(sl, 3, axis=1)
    
    # Таблица без GPU После изменений принимает вид:
    
    # Uptime - Temp_CPU - CPU_usage - RAM_usage 
    
    if not flag_give_gpu:
        predict_temp_cpu_if_not_gpu(sl, kritical_temperature_cpu)
    #else:
    #    problem_two(sl, kritical_temperature_cpu)
    # Таблица с GPU После изменений имеет вид:
    
    # Uptime - Temp_CPU - Temp_GPU - CPU_usage - GPU_usage - RAM_usage 
    
def predict_temp_cpu_if_not_gpu(sl: np.ndarray) -> np.ndaraay:
    # Вид numpy-массива sl: Uptime - Temp_CPU - CPU_usage - RAM_usage 
    # kr_temp - критическая температура CPU
    stolb_cpu_temp = sl[:, 1:2]
    
    stolb_cpu_temp = stolb_cpu_temp[:-1, :]
    
    stolb_cpu_temp = np.vstack((np.array([0]), stolb_cpu_temp))

    neue = np.hstack((sl, stolb_cpu_temp))

    
    #Теперь пятый столбец - будущая температура (температура через секунду). 
    # У последней (первой) строки её, конечно, нет
    neue = neue[1:,:] #Удаление последней строки ввиду отсутствия информации будущей температуры на неё
    
    #С этого момента в массиве neue хранится текущая информация и будущая информация по температуре процессора в следующей строке  
    # По факту вид: Uptime - Temp_CPU - CPU_usage - RAM_usage - Temp_CPU_future


    X = neue[:, :-1]
    Y = neue[:,-1]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
    
    vanina_model = LinearRegression()
    vanina_model.fit(x_train, y_train)
    
    predictions = vanina_model.predict(x_test)

    

    print(y_test)
    print(predictions)


arbeit_model()
    
#def problem_two(sl):  
#arbeit_model()