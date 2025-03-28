import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
#import xgboost as xgb
import pickle

def prepare_data(filename, target_column, has_gpu=False):
    data = pd.read_csv(filename)
    if has_gpu:
        r = ['uptime','temp_cpu','temp_gpu','cpu_usage','gpu_usage','ram_usage']
    else:
        r = ['uptime','temp_cpu','cpu_usage','ram_usage']
    X = data[r].values
    Y = data[target_column].values
    return X, Y

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def start_train_and_save_model(X, Y, filename, test_sizer = 0.2, random_stater = 42):
    '''
    X - массив признаков
    Y - массив целевых элементов
    '''
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=test_sizer,
        random_state=random_stater,
    )
    '''
    unsere_model = xgb.XGBRegressor(
        objective = 'reg:squarederror',
        n_estimators = 100000,
        learning_rate = 0.05,
        max_depth = 6,
        subsamble = 0.9,
        colsamble_bytree = 0.9,
    )
    '''
    unsere_model = GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.001,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        loss='huber',
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_stater
        
    )
    unsere_model.fit(
        x_train, y_train
    )
    
    y_predict = unsere_model.predict(x_test)
    joja = mean_absolute_error(y_test, y_predict)
    print(f'Средняя абсолютная ошибка(MAE){joja:.2f}')
    
    with open(filename, 'wb') as f:
        pickle.dump(unsere_model, f)
    return unsere_model

def predict_next_value(model, X):
    input_data = np.array(X).reshape(1, -1)
    
    predskaz = model.predict(input_data)
    return predskaz

def update_model(model_filename, new_data_filename, target_column, has_gpu=False,epoch=10):
    
    X, Y = prepare_data(new_data_filename, target_column, has_gpu=has_gpu)
    
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42,
    )
    model.set_params(
        n_estimators=epoch,
        warm_start = True,
        learning_rate = 0.05
    )
    model.fit(
        X_train, Y_train,
        eval_set=[(X_val, Y_val)],
        verbose = True,
        xgb_model=model
    )
    y_predskaz = model.predict(X_val)
    mae = mean_absolute_error(Y_val, y_predskaz)
    print(f'Средняя абсолютная ошибка после дообучения (MAE){mae:.2f}')
    return model #Возвращаем дообученную модель
def future_if_not_gpu():
    X = read_last()
    return np.array([X[0]] + [X[1]] + [X[4]] + [X[6]])
def future_if_gpu():
    X = read_last()
    return np.array(X[:2]+X[3:])
def read_all() -> list[tuple[int, float, float, float, float, float, float]]:
    df = pd.read_csv('data.csv')
    return [list(df.loc[i]) for i in range(len(df))]

def read_last() -> tuple[int, float, float, float, float, float, float]:
    df = pd.read_csv('data.csv')
    return tuple(df.loc[len(df)-1]) # get last series  

def help():
    print(
        """

        0 ячейка - uptime (Не предсказывается)
        1 ячейка - temp_cpu
        2 - krit_temp(Бесполезна)
        3 - cpu_usage
        4 - ram_usage

        При наличии GPU:
        0 - uptime (Не предсказывается)
        1 - temp_cpu
        2 - krit_temp (Бесполезна)
        3 - temp_gpu
        4 - cpu_usage
        5 - gpu_usage
        6 - ram_usage

        """)


def predict_future(column_number, has_gpu=False):
    if not has_gpu:
        X = future_if_not_gpu()
        if column_number == 1:
            model = load_model('vanya_analizator/models/model_temp_cpu_if_not_gpu.pkl')
        elif column_number == 3:
            model = load_model('vanya_analizator/models/model_cpu_usage_if_not_gpu.pkl')
        elif column_number == 4:
            model = load_model('vanya_analizator/models/model_ram_usage_if_not_gpu.pkl')
        else:
            raise IndexError('Это предсказать нельзя')
    else:
        X = future_if_gpu()
        if column_number == 1:
            model = load_model('vanya_analizator/models/model_temp_cpu_if_gpu.pkl')
        elif column_number == 3:
            model = load_model('vanya_analizator/models/model_temp_gpu_if_gpu.pkl')
        elif column_number == 4:
            model = load_model('vanya_analizator/models/model_cpu_usage_if_gpu.pkl')
        elif column_number == 5:
            model = load_model('vanya_analizator/models/model_gpu_usage_if_gpu.pkl')
        elif column_number == 6:
            model = load_model('vanya_analizator/models/model_ram_usage_if_gpu.pkl')
        else:
            raise IndexError('Это предсказать нельзя!')
    return predict_next_value(model, X)


def first_predict_all_future(has_gpu=False):
    kr_temp = np.array([read_last()[2]])
    if not has_gpu:
        hoh = np.array([-1])
        X = future_if_not_gpu()
        predict_cpu_temp = predict_future(1, False)
        predict_cpu_usage = predict_future(3, False)
        predict_ram_usage = predict_future(4, False)
        
        goga = np.array([X[0] + 1000])
        
        fofa = np.concatenate([goga, predict_cpu_temp, kr_temp, hoh, predict_cpu_usage, hoh, predict_ram_usage])
        
        return fofa
    else:
        predict_cpu_temp = predict_future(1, True)
        predict_cpu_usage = predict_future(4, True)
        predict_gpu_temp = predict_future(3, True)
        predict_gpu_usage = predict_future(5, True)
        predict_ram_usage = predict_future(6, True)
        X = future_if_gpu()
        goga = np.array([X[0]+1000])
        fofa = np.concatenate([goga, predict_cpu_temp, kr_temp, predict_gpu_temp, predict_cpu_usage, predict_gpu_usage, predict_ram_usage])
        return fofa
def next_all_predict(X: np.array, has_gpu=False):
    if not has_gpu:
        kr_temp = np.array([X[2]])
        array = np.array([X[0], X[1], X[4], X[6]])
        hoh = np.array([-1])
        model = load_model('vanya_analizator/models/model_temp_cpu_if_not_gpu.pkl')
        future_cpu_temp = predict_next_value(model, array)
        model_2 = load_model('vanya_analizator/models/model_cpu_usage_if_not_gpu.pkl')
        future_cpu_usage = predict_next_value(model_2, array)
        model_3 = load_model('vanya_analizator/models/model_ram_usage_if_not_gpu.pkl')
        future_ram_usage = predict_next_value(model_3, array)
        next_time = np.array([X[0]+1000])
        lolp = np.concatenate([next_time, future_cpu_temp, kr_temp, hoh, future_cpu_usage, hoh, future_ram_usage])
        return lolp
    else:
        kr_temp = np.array([X[2]])
        array = np.concatenate([X[:2],X[3:]])
        model = load_model('vanya_analizator/models/model_temp_cpu_if_gpu.pkl')
        future_cpu_temp = predict_next_value(model, array)
        model_2 = load_model('vanya_analizator/models/model_cpu_usage_if_gpu.pkl')
        future_cpu_usage = predict_next_value(model_2, array)
        model_3 = load_model('vanya_analizator/models/model_temp_gpu_if_gpu.pkl')
        future_temp_gpu = predict_next_value(model_3, array)
        model_4 = load_model('vanya_analizator/models/model_gpu_usage_if_gpu.pkl')
        future_gpu_usage = predict_next_value(model_4, array)
        model_5 = load_model('vanya_analizator/models/model_ram_usage_if_gpu.pkl')
        future_ram_usage = predict_next_value(model_5, array)
        next_time = np.array([X[0]+1000])
        '''
        print(next_time)
        print(future_cpu_temp)
        print(kr_temp)
        print(future_temp_gpu)
        print(future_cpu_usage)
        print(future_gpu_usage)
        print(future_ram_usage)
        print('\n\n\n')
        '''
        
        lolp = np.concatenate([next_time, future_cpu_temp, kr_temp, future_temp_gpu, future_cpu_usage, future_gpu_usage, future_ram_usage])
        
        print(f'На массив \n{array}\n предсказан массив \n{lolp}\n')
        return lolp
def all_start(has_gpu=False):
    fuk = 'data.csv'
    if not has_gpu:
        X, Y = prepare_data(fuk, 'temp_cpu', has_gpu)
        start_train_and_save_model(X, Y, 'vanya_analizator/models/model_temp_cpu_if_not_gpu.pkl')
        X2, Y2 = prepare_data(fuk, 'cpu_usage', has_gpu)
        start_train_and_save_model(X2, Y2, 'vanya_analizator/models/model_cpu_usage_if_not_gpu.pkl')
        X3, Y3 = prepare_data(fuk, 'ram_usage', has_gpu)
        start_train_and_save_model(X3, Y3, 'vanya_analizator/models/model_ram_usage_if_not_gpu.pkl')
    else:
        X4, Y4 = prepare_data(fuk, 'temp_cpu', has_gpu)
        start_train_and_save_model(X4, Y4, 'vanya_analizator/models/model_temp_cpu_if_gpu.pkl')
     
        X5, Y5 = prepare_data(fuk, 'temp_gpu', has_gpu)
        start_train_and_save_model(X5, Y5, 'vanya_analizator/models/model_temp_gpu_if_gpu.pkl')
        
        X6, Y6 = prepare_data(fuk, 'cpu_usage', has_gpu)
        start_train_and_save_model(X6, Y6, 'vanya_analizator/models/model_cpu_usage_if_gpu.pkl')
        
        X7, Y7 = prepare_data(fuk, 'gpu_usage', has_gpu)
        start_train_and_save_model(X7, Y7, 'vanya_analizator/models/model_gpu_usage_if_gpu.pkl')
        
        X8, Y8 = prepare_data(fuk, 'ram_usage', has_gpu)
        start_train_and_save_model(X8, Y8, 'vanya_analizator/models/model_ram_usage_if_gpu.pkl')
def all_model_reset():
    import os
    try:
        os.remove("vanya_analizator/models/model_temp_cpu_if_not_gpu.pkl")
    except:
        pass
    try:
        os.remove("vanya_analizator/models/model_cpu_usage_if_not_gpu.pkl")
    except:
        pass
    try:
        os.remove("vanya_analizator/models/model_ram_usage_if_not_gpu.pkl")
    except:
        pass
    try:
        os.remove("vanya_analizator/models/model_temp_cpu_if_gpu.pkl")
    except:
        pass
    try:
        os.remove("vanya_analizator/models/model_cpu_usage_if_gpu.pkl")
    except:
        pass
    try:
        os.remove("vanya_analizator/models/model_temp_gpu_if_gpu.pkl")
    except:
        pass
    try:
        os.remove("vanya_analizator/models/model_gpu_usage_if_gpu.pkl")
    except:
        pass
    try:
        os.remove("vanya_analizator/models/model_ram_usage_if_gpu.pkl")
    except:
        pass
