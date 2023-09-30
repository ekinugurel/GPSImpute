from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
import numpy as np
import similaritymeasures as sm
import pandas as pd

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

def evaluate_similarity(lat_tc, pred_mean, y_test_scaled):
    preds_lat = np.hstack((pd.Series(lat_tc.index).values.reshape(-1,1), pred_mean[:,0].reshape(-1,1)))
    test_lat = np.hstack((pd.Series(lat_tc.index).values.reshape(-1,1), y_test_scaled[:,0].reshape(-1,1)))

    preds_lon = np.hstack((pd.Series(lat_tc.index).values.reshape(-1,1), pred_mean[:,1].reshape(-1,1)))
    test_lon = np.hstack((pd.Series(lat_tc.index).values.reshape(-1,1), y_test_scaled[:,1].reshape(-1,1)))

    # quantify the difference between the two curves using PCM
    pcm_lat = sm.pcm(preds_lat, test_lat)
    pcm_lon = sm.pcm(preds_lon, test_lon)

    # quantify the difference between the two curves using
    # Discrete Frechet distance
    df_lat = sm.frechet_dist(preds_lat, test_lat)
    df_lon = sm.frechet_dist(preds_lon, test_lon)

    # quantify the difference between the two curves using
    # area between two curves
    area_lat = sm.area_between_two_curves(preds_lat, test_lat)
    area_lon = sm.area_between_two_curves(preds_lon, test_lon)

    # quantify the difference between the two curves using
    # Curve Length based similarity measure
    cl_lat = sm.curve_length_measure(preds_lat, test_lat)
    cl_lon = sm.curve_length_measure(preds_lon, test_lon)

    # quantify the difference between the two curves using
    # Dynamic Time Warping distance
    dtw_lat, d_lat = sm.dtw(preds_lat, test_lat)
    dtw_lon, d_lon = sm.dtw(preds_lon, test_lon)

    # mean absolute error
    mae_lat = sm.mae(preds_lat, test_lat)
    mae_lon = sm.mae(preds_lon, test_lon)

    # mean squared error
    mse_lat = sm.mse(preds_lat, test_lat)
    mse_lon = sm.mse(preds_lon, test_lon)

    # Take the average of the metrics
    return {
        'PCM': (pcm_lat + pcm_lon) / 2,
        'DF': (df_lat + df_lon) / 2,
        'AREA': (area_lat + area_lon) / 2,
        'CL': (cl_lat + cl_lon) / 2,
        'DTW': (dtw_lat + dtw_lon) / 2,
        'MAE': (mae_lat + mae_lon) / 2,
        'MSE': (mse_lat + mse_lon) / 2
    }