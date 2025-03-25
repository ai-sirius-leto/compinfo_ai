from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
def modelka(X: np.ndarray, Y: np.ndarray):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_scaled, Y)
    
    return model