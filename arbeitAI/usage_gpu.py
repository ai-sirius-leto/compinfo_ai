import numpy as np
from arbeitAI import modelka
def learn_model_usage_gpu(sl: np.ndarray):
    """
    Вид массива sl:  Uptime - temp_cpu - temp_gpu - cpu_usage - gpu_usage - ram_usage
    """
    
    jopa = sl[:,4:5]
    
    jopa = jopa[1:,:]
    jopa = np.vstack((jopa, np.array([0])))
    
    neue = np.hstack((sl, jopa))
    neue = neue[:-1,:]
    
    X = neue[:,:-1]
    Y = neue[:,-1:]

    return modelka(X, Y)
def predict_usage_gpu(sl: np.ndarray):
    koks = learn_model_usage_gpu(sl)
    future = sl[-1:,:]
    return koks.predict(future)