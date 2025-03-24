import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression
from arbeitAI.utils import to_model
def learn_model_ram_if_not_gpu(sl: np.ndarray): #Функция возвращает обученную модель
    """На вход подаётся нумпи массив данных, на основе которых будет учиться модель"""
    # Вид numpy-массива sl: Uptime - Temp_CPU - CPU_usage - RAM_usage 
    """Первая строка - самая старая, последняя - последняя. Будущее значение последней нам неизвестно"""
    #sl = to_model(sl)
    stolb_ram = sl[:, -1:]
    
    
    
    stolb_ram = stolb_ram[1:, :]
    
    
    stolb_ram = np.vstack((stolb_ram, np.array([[0]])))
    
    neue = np.hstack((sl, stolb_ram))
    
    neue = neue[:-1, :]
    
    X = neue[:, :-1]
    Y = neue[:, -1:]
    
    model = LinearRegression()
    model.fit(X, Y)
    
    return model
def predict_ram_if_not_gpu(sl):
    
    van_model = learn_model_ram_if_not_gpu(sl)
    
    future = sl[-1:, :]
    
    return van_model.predict(future)