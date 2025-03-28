import numpy as np

from arbeitAI.model import modelka
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
   
    return modelka(X, Y)
def predict_ram_if_not_gpu(sl):
    
    van_model = learn_model_ram_if_not_gpu(sl)
    
    future = sl[-1:, :]
    
    return van_model.predict(future)

def learn_model_ram_if_gpu(sl: np.ndarray): #Функция возвращает модель, обученную
    """
    Вид массива sl:  Uptime - temp_cpu - temp_gpu - cpu_usage - gpu_usage - ram_usage
    """
    netwola = sl[:, -1:]
    
    netwola = netwola[1:,:]
    
    netwola = np.vstack((netwola,np.array([0])))
    
    neue = np.hstack((sl, netwola))
    
    neue = neue[:-1, :]
    
    X = neue[:, :-1]
    Y = neue[:, -1:]
    return modelka(X, Y)

def predict_ram_if_gpu(sl: np.ndarray):
    
    moka = learn_model_ram_if_gpu(sl)
    
    future = sl[-1:, :]
    return moka.predict(future)