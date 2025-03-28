from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np


def modelka(X: np.ndarray, Y:np.ndarray):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.3)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_predskaz = model.predict(X_test)
    
    print("Коэффициент детерминации (R^2):", r2_score(Y_test, Y_predskaz))
    
    return model
    
    
    
    
def modelka_2(X: np.ndarray, Y: np.ndarray): #Обучает модель
    '''
    Функия обучает текущую модель. Х - данные, У - то, что должно предсказаться 
    (Иными словами У - данные, которые должны предсказаться)
    
    Массив Х - numpy-array, представляющий собой двумерную матрицу
    Вид массива X (зависит от наличия GPU у пользователя:
    
    Uptime - temp_cpu - temp_gpu - cpu_usage - gpu_usage - ram_usage
    
    Uptime - temp_cpu - cpu_usage - ram_usage
    
    uptime - Время с момента включения
    temp_cpu - температура CPU
    temp_gpu - температура GPU (Если есть)
    cpu_usage - Использование CPU
    gpu_usage - Использование GPU (Если есть)
    ram_usage - Использование ОЗУ
    )
    
    Массив Y - тоже numpy-array, тоже двумерный массив, но имеющий 1 столбец
    Y представляет собой информацию о будущей temp_cpu или cpu_usage или что то другое, всегда разное. 
    
    В общем то, что нужно предсказать
    '''
    
    
    
    
    