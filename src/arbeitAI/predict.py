# import numpy as np
# from numpy import ndarray
# from arbeitAI.ram_usage import predict_ram_if_gpu, predict_ram_if_not_gpu
# from arbeitAI.temp_cpu import predict_temp_cpu_if_gpu, predict_temp_cpu_if_not_gpu
# from arbeitAI.usage_cpu import predict_usage_cpu_if_gpu, predict_usage_cpu_if_not_gpu

# def predict_if_not_gpu(data: ndarray, ut: int):
#     future_temp_cpu = np.clip(predict_temp_cpu_if_not_gpu(data), a_min=0, a_max=100)
#     future_usage_cpu = np.clip(predict_usage_cpu_if_not_gpu(data), a_min=0, a_max=100)
#     future_usage_ram = np.clip(predict_ram_if_not_gpu(data), a_min=0, a_max=100)
#     return ut, float(future_temp_cpu[0]), 100, -1, float(future_usage_cpu[0]), -1, float(future_usage_ram[0])

# def predict_if_gpu(data: ndarray, ut: int):
#     future_temp_cpu = np.clip(predict_temp_cpu_if_gpu(data), a_min=0, a_max=100)
#     future_usage_cpu = np.clip(predict_usage_cpu_if_gpu(data), a_min=0, a_max=100)
#     future_usage_ram = np.clip(predict_ram_if_gpu(data), a_min=0, a_max=100)
#     return ut, float(future_temp_cpu[0]), 100, -1, float(future_usage_cpu[0]), -1, float(future_usage_ram[0])

# def predict(data: ndarray, ut: int) -> tuple[int, float, float, float]:
#     data = np.delete(data, 2, axis = 1)
#     flag_give_gpu = bool(data[1, 2] + 1)

#     if not flag_give_gpu:
#         data = np.delete(data, 2, axis=1)
#         data = np.delete(data, 3, axis=1)
#     
#     if not flag_give_gpu:
#         return predict_if_not_gpu(data, ut)
#     else:
#         return predict_if_gpu(data, ut)

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import adam_v2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from collections import deque
import os

def read_all() -> list[tuple[int, float, float, float, float, float, float]]:
    df = pd.read_csv('data.csv')
    return [list(df.loc[i]) for i in range(len(df))]

def read_last() -> tuple[int, float, float, float, float, float, float]:
    df = pd.read_csv('data.csv')
    return tuple(df.loc[len(df)-1]) # get last series
def create_and_train_model(model_filename, scaler_filename, target_column, time_steps=30, epochs=25):
    data = pd.read_csv('data.csv')
    feat = ['uptime', 'temp_cpu', 'cpu_usage', 'ram_usage']
    if read_last()[3] != -1:
        feat.extend(['temp_gpu', 'gpu_usage'])
    
    X = data[feat].values
    Y = data[target_column].values
    
    Y = Y[1:]
    X = X[:-1, :]
    
    
    
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    X_scaled = scaler.fit_transform(X)
    
    if 'usage' in target_column:
        Y = Y / 100
    
    
    
    X_lstm, Y_lstm = list(),list()
    
    for i in range(len(X_scaled) - time_steps):
        X_lstm.append(X_scaled[i:i + time_steps])
        Y_lstm.append(Y[i+time_steps])
    
    X_lstm = np.array(X_lstm)
    Y_lstm = np.array(Y_lstm)
    
    if len(X_lstm.shape) == 2:
        X_lstm = np.expand_dims(X_lstm, axis=2)

    #С этого момента данные правильно преобразованы
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_lstm, Y_lstm, test_size=0.3, random_state=42)

    '''СОЗДАНИЕ МОДЕЛИ LSTM (НАСТРАИВАЕМО)'''
    model = Sequential([
        LSTM(
            128,
            input_shape=(time_steps, X_train.shape[2]),
            return_sequences=True,
            kernel_regularizer=l2(0.001)
        ),
        
        Dropout(0.3),
        
        LSTM(
            64,
            return_sequences=False
        ),
        
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        
        
        Dense(1, activation='linear' if 'temp' in target_column else 'sigmoid')
    ])
    '''
    model = Sequential([
        LSTM(64, input_shape=(time_steps, X_train.shape[2])),
        Dropout(0.2),
        Dense(32,activation='relu'),
        Dense(1, activation='linear' if 'temp' in target_column else 'sigmoid')
    ])
    '''
    model.compile(
        optimizer=adam_v2.Adam(learning_rate=0.001),
        loss='mse',
        metrics='mae'
    )
    model_path = os.path.join('LSTM/models', model_filename)
    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_mae',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_mae',
            patience=25,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=5,
            min_lr=1e-5
        )
    ]
    histore = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    scaler_path = os.path.join('LSTM/models/scalers', scaler_filename)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'Model has saved in {model_path}')
    print(f'Scaler file save in {scaler_path}')
    print('Vanya chort!')
    
    return model_path
def load_saved_model(model_filename, scaler_filename):
    model_dir = os.path.join('LSTM/models', model_filename)
    model = load_model(model_dir)
    
    scaler_path = os.path.join('LSTM/models/scalers', scaler_filename)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    time_steps = model.input_shape[1]
    return (model, scaler, time_steps)
def update_model(model_filename, scaler_filename, new_data_path, epochs=10):
    model_dir = os.path.join('LSTM/models', model_filename)
    model = load_model(model_dir)
    
    scaler_sir = os.path.join('LSTM/models/scalers', scaler_filename)
    with open(scaler_sir, 'rb') as f:
        scaler = pickle.load(f)
        
        
    new_data = pd.read_csv(new_data_path)
    festures = ['uptime', 'temp_cpu', 'cpu_usage', 'ram_usage']
    if read_last()[3] != -1:
        festures.extend(['temp_gpu', 'gpu_usage'])
        
    X_new = new_data[festures].values
    Y_new = new_data[model.output_names[0]].values
    X_new_scaled = scaler.transform(X_new)
    
    time_steps = model.input_shape[1]
    X_lstm, Y_lstm = list(), list()
    
    for i in range(len(X_new_scaled) - time_steps):
        X_lstm.append(X_new_scaled[i:i + time_steps])
        Y_lstm.append(Y_new[i+time_steps])
    
    history = model.fit(
        X_lstm, Y_lstm,
        epochs = epochs,
        batch_size = 32,
        validation_split=0.2,
        verbose=1,
    )
    
    model.save(model_dir)
    return model_dir

def predict_future(target_column, model, scaler, initial_data, steps = 30, base_valotility=0.5):
    has_gpu = initial_data.shape[1] == 6
    
    if has_gpu:
        col_map = {
            'uptime':0,
            'temp_cpu':1,
            'temp_gpu':2,
            'cpu_usage':3,
            'gpu_usage':4,
            'ram_usage':5
        }
    else:
        col_map = {
            'uptime':0,
            'temp_cpu':1,
            'cpu_usage':2,
            'ram_usage':3
        }
        
    target_index = col_map[target_column]
    
    volatility_params = {
        'temp_cpu':{'base':0.03, 'spike_prob':0.1, 'spike_mult':2.0},
        'temp_gpu':{'base':0.03, 'spike_prob':0.07, 'spike_mult':1.8},
        'cpu_usage':{'base':0.025, 'spike_prob':0.1, 'spike_mult':1.8},
        'gpu_usage':{'base':0.03, 'spike_prob':0.12, 'spike_mult':2.0},
        'ram_usage':{'base':0.001, 'spike_prob':0.002, 'spike_mult':1.0},
    }
    
    params = volatility_params[target_column]
    current_window = initial_data.copy()
    predictions = list()
    
    for _ in range(steps):
        window_scaled = scaler.transform(current_window)
        pred_norm = model.predict(window_scaled[np.newaxis, :, :], verbose=0)[0][0]
        
        noise = np.random.normal(scale=params['base'] * 0.5)
        
        if np.random.rand() < params['spike_prob']:
            noise += params['spike_mult'] * np.random.rand()
            
        pred_norm += noise

        if 'usage' in target_column:
            pred_norm = np.clip(pred_norm, 0, 1)
            pred_original = pred_norm*100
        else:
                    
            dummy_row = np.zeros((1, scaler.n_features_in_))
            dummy_row[0, target_index] = pred_norm
            pred_original = scaler.inverse_transform(dummy_row)[0, target_index]
            pred_original = np.clip(pred_norm, 30, 90)
        
        predictions.append(pred_original)
        
        new_row = current_window[-1].copy()
        new_row[0] += 1000
        new_row[target_index] = pred_original
        
        for i in range(1, len(new_row)):
            if i != target_index:
                decay = 0.98 if ('gpu' in target_column) else 0.995
                new_row[i] *= decay
        
        current_window = np.vstack([current_window[1:], new_row])
    return np.array(predictions)
    
    
def all_model_remove():
    try:
        os.remove('LSTM/models/temp_cpu_if_gpu.h5')
        os.remove('LSTM/models/scalers/temp_cpu_if_gpu_scaler.pkl')
    except:
        pass
    try:
        os.remove('LSTM/models/cpu_usage_if_gpu.h5')
        os.remove('LSTM/models/scalers/cpu_usage_if_gpu_scaler.pkl')
    except:
        pass
    try:
        os.remove('LSTM/models/gpu_usage.h5')
        os.remove('LSTM/models/scalers/gpu_usage_scaler.pkl')
    except:
        pass
    try:
        os.remove('LSTM/models/temp_gpu.h5')
        os.remove('LSTM/models/scalers/temp_cpu_scaler.pkl')
    except:
        pass
    try:
        os.remove('LSTM/models/ram_usage_if_gpu.h5')
        os.remove('LSTM/models/scalers/ram_usage_if_gpu_scaler.pkl')
    except:
        pass
    try:
        os.remove('LSTM/models/ram_usage_if_not_gpu.h5')
        os.remove('LSTM/models/scalers/ram_usage_if_not_gpu_scaler.pkl')
    except:
        pass
    try:
        os.remove('LSTM/models/cpu_usage_if_not_gpu.h5')
        os.remove('LSTM/models/scalers/cpu_usage_if_not_gpu_scaler.pkl')
    except:
        pass
    try:
        os.remove('LSTM/models/temp_cpu_if_not_gpu.h5')
        os.remove('LSTM/models/scalers/temp_cpu_if_not_gpu_scaler.pkl')
    except:
        pass

def all_predict(has_gpu, time_steps):
    n = pd.read_csv('data.csv')
        
    data = n.values[-30:, :]
        
    if has_gpu:
        
        data = np.hstack((data[:, :2], data[:, 3:]))
        model, scaler, time_steps = load_saved_model('temp_cpu_if_gpu.h5', 'temp_cpu_if_gpu_scaler.pkl')
 
        
        future_temp_cpu = predict_future('temp_cpu', model, scaler, data, time_steps).reshape(-1,1)
        
        model, scaler, time_steps = load_saved_model('cpu_usage_if_gpu.h5', 'cpu_usage_if_gpu_scaler.pkl')
        
        future_cpu_usage = predict_future('cpu_usage', model, scaler, data, time_steps).reshape(-1,1)
        
        model,scaler, time_steps = load_saved_model('temp_gpu.h5', 'temp_gpu_scaler.pkl')
        
        future_temp_gpu = predict_future('temp_gpu', model, scaler, data, time_steps).reshape(-1,1)
        
        model,scaler, time_steps = load_saved_model('gpu_usage.h5', 'gpu_usage_scaler.pkl')
        
        future_gpu_usage = predict_future('gpu_usage', model, scaler, data, time_steps).reshape(-1,1)
        
        model,scaler, time_steps = load_saved_model('ram_usage_if_gpu.h5', 'ram_usage_if_gpu_scaler.pkl')
        
        future_ram_usage = predict_future('ram_usage', model, scaler, data, time_steps).reshape(-1,1)
        
        
        
        all_pred = np.hstack((data[:, :1] + 1000, future_temp_cpu, future_temp_gpu, future_cpu_usage, future_gpu_usage, future_ram_usage))
        
        return all_pred
    else:
        data = np.hstack((data[:, :1], data[:, 1:2], data[:, 4:5], data[:, 6:]))


        model, scaler, time_steps = load_saved_model('temp_cpu_if_not_gpu.h5', 'temp_cpu_if_not_gpu_scaler.pkl')
        
        future_temp_cpu = predict_future('temp_cpu', model, scaler, data, time_steps).reshape(-1,1)
        
        model, scaler, time_steps = load_saved_model('cpu_usage_if_not_gpu.h5', 'cpu_usage_if_not_gpu_scaler.pkl')
        
        future_cpu_usage = predict_future('cpu_usage', model, scaler, data, time_steps).reshape(-1,1)
        
        model, scaler, time_steps = load_saved_model('ram_usage_if_not_gpu.h5', 'ram_usage_if_not_gpu_scaler.pkl')
        
        future_ram_usage = predict_future('ram_usage', model, scaler, data, time_steps).reshape(-1,1)
        
        return np.hstack((data[:, :1] + 1000, future_temp_cpu, 100, -1, future_cpu_usage, -1, future_ram_usage))
    
def all_model_reset(has_gpu=False):
    all_model_remove()
    if has_gpu:
        create_and_train_model('temp_cpu_if_gpu.h5', 'temp_cpu_if_gpu_scaler.pkl', 'temp_cpu', 30, 500)
        create_and_train_model('cpu_usage_if_gpu.h5', 'cpu_usage_if_gpu_scaler.pkl', 'cpu_usage', 30, 500)
        create_and_train_model('temp_gpu.h5', 'temp_gpu_scaler.pkl', 'temp_gpu', 30, 500)
        create_and_train_model('gpu_usage.h5', 'gpu_usage_scaler.pkl', 'gpu_usage', 30, 500)
        create_and_train_model('ram_usage_if_gpu.h5', 'ram_usage_if_gpu_scaler.pkl', 'ram_usage', 30, 500)
    else:
        create_and_train_model('temp_cpu_if_not_gpu.h5', 'temp_cpu_if_not_gpu_scaler.pkl', 'temp_cpu', 30, 500)
        create_and_train_model('cpu_usage_if_not_gpu.h5', 'cpu_usage_if_not_gpu_scaler.pkl', 'cpu_usage', 30, 500)
        create_and_train_model('ram_usage_if_not_gpu.h5', 'ram_usage_if_not_gpu_scaler.pkl', 'ram_usage', 30, 500)



