import numpy as np
from arbeitAI import modelka
def learn_model_temp_gpu(sl: np.ndarray):
    """
    Вид массива sl:  Uptime - temp_cpu - temp_gpu - cpu_usage - gpu_usage - ram_usage
    """
    
    popa = sl[:,2:3]
    
    popa = popa[1:,:]
    
    popa = np.vstack((popa, np.array([0])))
    
    neue = np.hstack((sl, popa))
    neue = neue[:-1,:]
    
    X = neue[:,:-1]
    Y = neue[:,-1:]
    
    
    
    
    
    
    
    
    
    
    
    
    return modelka(X, Y)
def predict_temp_gpu(sl: np.ndarray):
    
    gosha = learn_model_temp_gpu(sl)
    future = sl[-1:,:]
    return gosha.predict(future)
