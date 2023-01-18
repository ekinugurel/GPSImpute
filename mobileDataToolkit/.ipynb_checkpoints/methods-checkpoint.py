import numpy as np
import math
from sklearn.metrics import mean_squared_error

def LI(X_train, X_test, y_train, y_test):
    preds_lat = []
    preds_long = []
    for i,j in enumerate(X_test):
        preds_lat.append(np.interp(j, X_train, y_train[:,0]))
        preds_long.append(np.interp(j, X_train, y_train[:,1]))

    rmse_lat = mean_squared_error(preds_lat, y_test[:,0], squared=False)
    rmse_long = mean_squared_error(preds_long, y_test[:,1], squared=False)
    rmse = math.sqrt(rmse_lat**2 + rmse_long**2)
    return rmse