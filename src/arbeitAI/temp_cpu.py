import numpy as np


from arbeitAI.model import modelka
def learn_model_temp_cpu_if_not_gpu(sl: np.ndarray): #Функция возвращает обученную модель
    
    """На вход подаётся нумпи массив данных, на основе которых будет учиться модель"""
    # Вид numpy-массива sl: Uptime - Temp_CPU - CPU_usage - RAM_usage 
    """Первая строка - самая старая, последняя - последняя. Будущее значение последней нам неизвестно"""
    
    #sl = to_model(sl)
    
    stolb_cpu_temp = sl[:, 1:2]
    
    stolb_cpu_temp = stolb_cpu_temp[1:, :]
    
    stolb_cpu_temp = np.append(stolb_cpu_temp, np.array([[0]]), axis=0)

    neue = np.hstack((sl, stolb_cpu_temp))

    #Теперь пятый столбец - будущая температура (температура через секунду). 
    # У первой строки её, конечно, нет

    neue = neue[:-1,:] #Удаление последней строки ввиду отсутствия информации будущей температуры на неё

    #С этого момента в массиве neue хранится текущая информация и будущая информация по температуре процессора в следующей строке  
    # По факту вид: Uptime - Temp_CPU - CPU_usage - RAM_usage - Temp_CPU_future


    X = neue[:, :-1]
    Y = neue[:,-1]
    
    return modelka(X, Y)
def predict_temp_cpu_if_not_gpu(sl: np.ndarray) -> np.ndarray: #Функция обучает модель и возвращает температуру CPU в будущую секунду
    # Вид numpy-массива sl: Uptime - Temp_CPU - CPU_usage - RAM_usage 
    # kr_temp - критическая температура CPU
    model = learn_model_temp_cpu_if_not_gpu(sl)
    
    future = sl[-1:,:]

    return model.predict(future)

def learn_model_temp_cpu_if_gpu(sl: np.ndarray):
    """
    Вид массива sl:  Uptime - temp_cpu - temp_gpu - cpu_usage - gpu_usage - ram_usage
    """
    noka = sl[:,1:2]

    
    noka = noka[1:,:]
    noka = np.vstack((noka, np.array([0])))
    
    neue = np.hstack((sl, noka))
    neue = neue[:-1,:]
    
    X = neue[:,:-1]
    Y = neue[:,-1:]
    return modelka(X, Y)
def predict_temp_cpu_if_gpu(sl: np.ndarray):
    tolya = learn_model_temp_cpu_if_gpu(sl)
    future = sl[-1:,:]
    return tolya.predict(future)