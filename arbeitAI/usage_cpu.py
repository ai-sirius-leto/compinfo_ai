import numpy as np
from sklearn.linear_model import LinearRegression 
from arbeitAI.utils import to_model
def learn_model_cpu_usage_if_not_gpu(sl:np.ndarray): #Функция возвращает обученную модель
    """На вход подаётся нумпи массив данных, на основе которых будет учиться модель"""
    # Вид numpy-массива sl: Uptime - Temp_CPU - CPU_usage - RAM_usage 
    """Первая строка - самая старая, последняя - последняя. Будущее значение последней нам неизвестно"""
    #sl = to_model(sl)
    stolb_cpu_temp = sl[:, 2:3]
    
    stolb_cpu_temp = stolb_cpu_temp[1:, :]
    stolb_cpu_temp = np.append(stolb_cpu_temp, np.array([[0]]), axis = 0)
    
    
    neue = np.hstack((sl, stolb_cpu_temp))
    
    neue = neue[:-1, :]
    
    X = neue[:, :-1]
    Y = neue[:, -1:]
    
    vanina_model = LinearRegression().fit(X, Y)
    return vanina_model
    
def predict_usage_cpu_if_not_gpu(sl: np.ndarray):
    model = learn_model_cpu_usage_if_not_gpu(sl)
    
    future = sl[-1:,:]
    return model.predict(future)
    