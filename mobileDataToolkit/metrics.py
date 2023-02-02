from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
import numpy as np

def absolute_percentage_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(absolute_percentage_error(y_true, y_pred))


def max_absolute_percentage_error(y_true, y_pred):
    return np.max(absolute_percentage_error(y_true, y_pred))


def total_absolute_percentage_error(y_true, y_pred):
    return np.sum(absolute_percentage_error(y_true, y_pred))


def evaluate(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAD': median_absolute_error(y_true, y_pred),
        #'R2': r2_score(y_true, y_pred, ts),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'MAXAPE': max_absolute_percentage_error(y_true, y_pred),
        'TAPE': total_absolute_percentage_error(y_true, y_pred)
    }

def average_eval(y_true_lat, y_true_lon, y_pred_lat, y_pred_lon):
    eval1 = evaluate(y_true_lat, y_pred_lat)
    eval2 = evaluate(y_true_lon, y_pred_lon)
    
    averaged = list()
    for i, j in zip(eval1.values(), eval2.values()):
        averaged.append(np.sqrt(i**2 + j**2))
        
    return {
        'MAE': averaged[0],
        'RMSE': averaged[1],
        'MAD': averaged[2],
        #'R2': r2_score(y_true, y_pred, ts),
        'MAPE': averaged[3],
        'MAXAPE': averaged[4],
        'TAPE': averaged[5]
    }