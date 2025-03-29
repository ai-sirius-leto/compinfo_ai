import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.layers import InputLayer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from collections import deque
import os

def create_and_save_joint_model(model_filename, scaler_filename, time_steps = 30, epochs = 25, has_gpu = True):
    data = pd.read_csv('src/data.csv')
    
    model_filename = os.path.join('Models_new_LSTM', model_filename)
    scaler_filename = os.path.join('Models_new_LSTM/scalers', scaler_filename)
    
    if has_gpu:
        features = ['uptime', 'temp_cpu', 'temp_gpu', 'cpu_usage', 'gpu_usage', 'ram_usage']
        features_to_predict = ['temp_cpu', 'temp_gpu', 'cpu_usage', 'gpu_usage', 'ram_usage']
    else:
        features = ['uptime', 'temp_cpu', 'cpu_usage', 'ram_usage'] 
        features_to_predict = ['temp_cpu', 'cpu_usage', 'ram_usage']
    
    all_data = data[features].values
    data_to_predict = data[features_to_predict].values
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_predict)
    
    noise = 0.05
    scaled_data = scaled_data + np.random.normal(0, noise, scaled_data.shape)
    
    
    X, Y = [], []
    for i in range(len(scaled_data) - time_steps - 1):
        X.append(scaled_data[i:i+time_steps])
        Y.append(scaled_data[i+time_steps])
    X = np.array(X)
    Y = np.array(Y)
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    
    
    
    model = Sequential([
        LSTM(128, input_shape=(time_steps, X_train.shape[2])),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(Y_train.shape[1])
    ])
    
    model.compile(optimizer=adam_v2.Adam(learning_rate=0.005), loss = 'huber_loss', metrics=['mae'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15),
        ReduceLROnPlateau(monitore='val_loss', factor=0.5, patience=5),
        ModelCheckpoint(
            filepath = model_filename, monitor='val_loss', save_best_only=True
        )
    ]
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs = epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    return model
    
    
    
def model_remove(s):
    if s:
        try:
            os.remove('Models_new_LSTM/model_if_gpu.h5')
            os.remove('Models_new_LSTM/scalers/scaler_if_gpu.pkl')
        except:
            pass
    else:
        try:
            os.remove('Models_new_LSTM/model_if_not_gpu.h5')
            os.remove('Models_new_LSTM/scalers/scaler_if_not_gpu.pkl')
        except:
            pass

def predict_joint_future(model, scaler, initial_data, steps=30):
    
    uptime = initial_data[:, 0]
    features = initial_data[:, 1:]
    
    scaled_features = scaler.transform(features)
    
    current_window = scaled_features[-30:]
    predictions = list()
    
    noise_level = 0.05
    volatility_factor = 0.05
    
    for _ in range(steps):
        pred_scaled = model.predict(current_window[np.newaxis, :, :], verbose = 0)[0]
        
        pred_scaled += np.random.normal(0, noise_level, pred_scaled.shape)
        
        if len(predictions) > 0:
            last_diff = pred_scaled - current_window[-1]
            pred_scaled += last_diff * volatility_factor * np.random.randn()

        pred_original = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
        
        predictions.append(pred_original)
        
        current_window = np.vstack([current_window[1:], pred_scaled])

    last_uptime = uptime[-1]
    predicted_uptime = [last_uptime+1000*(i+1) for i in range(steps)]
    
    full_predictions = np.column_stack([predicted_uptime, np.array(predictions)])
    return full_predictions

def load_joint_model(has_gpu=True):
    if has_gpu:
        model_path = 'Models_new_LSTM/model_if_gpu.h5'
        scaler_path = 'Models_new_LSTM/scalers/scaler_if_gpu.pkl'
    else:
        model_path = 'Models_new_LSTM/model_if_not_gpu.h5'
        scaler_path = 'Models_new_LSTM/scalers/scaler_if_not_gpu.pkl'
        
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)    
    return model, scaler
def model_reset(has_gpu=True):
    if has_gpu:
        model_remove(True)
        create_and_save_joint_model('model_if_gpu.h5', 'scaler_if_gpu.pkl', epochs=500, has_gpu=True)
    else:
        model_remove(False)
        create_and_save_joint_model('model_if_not_gpu.h5', 'scaler_if_not_gpu.pkl', epochs=500, has_gpu=False)
        
def jojo(n, has_gpu=True):
    if has_gpu:
        model, scaler = load_joint_model(True)
        
        data = pd.read_csv('src/data.csv')
        
        features = ['uptime', 'temp_cpu', 'temp_gpu', 'cpu_usage', 'gpu_usage', 'ram_usage']
        
        last_points = data[features].values[-30:]
        
        predictions = predict_joint_future(model, scaler, last_points, n)
        
        kriti = data.values[-n:, 2:3]
        np.set_printoptions(linewidth=np.inf)
        
        s1 = predictions[:, :2]
        s2 = predictions[:, 2:]
        
        answer = np.hstack((s1, kriti, s2))
        return answer
    else:
        model, scaler = load_joint_model(False)
        
        data = pd.read_csv('src/data.csv')
        
        features = ['uptime', 'temp_cpu', 'cpu_usage', 'ram_usage']
        
        last_points = data[features].values[-30:]
        
        predictions = predict_joint_future(model, scaler, last_points, n)
        
        kriti = data.values[-n:, 2:3]
        np.set_printoptions(linewidth=np.inf)
        
        notigs = n.values[-30:, 3:4]
        
        time_temp_cpu = predictions[:, :2]
        cpu_usage = predictions[:, 2:3]
        ram_usage = predictions[:, 3:]
        
        answer = np.hstack((time_temp_cpu, kriti, notigs, cpu_usage, notigs, ram_usage))
        return answer
    
#Функции и их описание
'''
Во всех функциях flag - показатель наличия видеокарты

model_remove(flag) - убивает модель этого флага, если они имеются
model_reset(flag) сначала проводит model_remove(flag), а создает и обучает модель этого флага на текущих данных
jojo(n, flag):

Возвращает numpy-array - продолжение таблицы на основе модели LSTM данного флага (если модели нет выведет ошибку)
'''
