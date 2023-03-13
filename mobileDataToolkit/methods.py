import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib
from matplotlib import pyplot as plt, cm
import utils.helper_func as helper_func
import pandas as pd
import torch
from scipy.spatial.distance import cdist
import mobileDataToolkit.metrics as metrics

def mobVisualize(data, axes=None, **kwargs):
    """
    Visualizes the trajectory of a mobile device.

    Parameters
    ----------
    data : pd.DataFrame
        Trajectory data that needs to be visualized.
    axes : matplotlib.axes._subplots.AxesSubplot, optional
        Axes to plot on. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    f, (y1_ax, y2_ax) = plt.subplots(2, 1, constrained_layout = True)
    if 'user_ID' in list(data.columns):
        if len(data.user_ID.unique()) == 1:
            f.suptitle('User ID: ' + str(data.user_ID.unique().item()), fontsize = 10)
    elif 'UID' in list(data.columns):
        if len(data.UID.unique()) == 1:
            f.suptitle('User ID: ' + str(data.UID.unique().item()), fontsize = 10)
    if 'datetime' in list(data.columns):
        data = data.rename(columns={'datetime': 'Date_Time'})

    times = [date_obj.strftime('%H:%M:%S') for date_obj in pd.to_datetime(data.Date_Time).dt.time]

    y1_ax.scatter(times, data['norm_lat'], marker='.', c='blue', **kwargs)
    y1_ax.set_title('(Normalized) Latitude', fontsize = 10, **kwargs)
    y1_ax.set_xticks([])

    y2_ax.scatter(times, data['norm_long'], marker='.', c='blue', **kwargs)
    y2_ax.set_title('(Normalized) Longitude', fontsize = 10, **kwargs)

    try:
        y1_ax.fill_between(data['Date_Time'], 0, 1, 
                           where=((data['Date_Time'] >= test_start_date) & (data['Date_Time'] <= test_end_date)), 
                           color='pink', alpha=0.5, label = 'Testing period', transform=y1_ax.get_xaxis_transform())
        y2_ax.fill_between(data['Date_Time'], 0, 1, 
                           where=(data['Date_Time'] >= test_start_date) & (data['Date_Time'] <= test_end_date), 
                           color='pink', alpha=0.5, label = 'Testing period', transform=y2_ax.get_xaxis_transform())
    except AttributeError:
        pass
    except NameError:
        pass
    #y1_ax.set_ylim([min(data['orig_lat']), max(data['orig_lat'])])
    #y2_ax.set_ylim([min(data['orig_long']), max(data['orig_long'])])

    #y1_ax.legend(bbox_to_anchor=(0, 1.05, 1, 0.2), loc="lower left",
    #        borderaxespad=0)
    #y2_ax.legend(bbox_to_anchor=(0, 1.05, 1, 0.2), loc="lower left",
    #        borderaxespad=0)
    plt.xticks(rotation = 30, fontsize = 8)
    plt.xlabel('Date', fontsize=10)

    plt.show()

def homeLoc(data, m_threshold = 50):
    home_lat = data[(data['Hour'] <= 6) | (data['Hour'] >= 22)]['orig_lat'].mode().item()
    home_long = data[(data['Hour'] <= 6) | (data['Hour'] >= 22)]['orig_long'].mode().item()

    upper_b_lat, upper_b_long = helper_func.newCoords(home_lat, home_long, m_threshold, 0)
    right_b_lat, right_b_long = helper_func.newCoords(home_lat, home_long, 0, m_threshold)
    lower_b_lat, lower_b_long = helper_func.newCoords(home_lat, home_long, -m_threshold, 0)
    left_b_lat, left_b_long = helper_func.newCoords(home_lat, home_long, 0, -m_threshold)

    home = []

    for i, j in enumerate(data['orig_lat']):
        if (
            (data['orig_lat'][i] > upper_b_lat) or
            (data['orig_long'][i] > right_b_long) or
            (data['orig_lat'][i] < lower_b_lat) or
            (data['orig_long'][i] < left_b_long)
        ):
            home.append(0)
        else:
            home.append(1)
    data['home'] = home

    return home_lat, home_long

def homeLocv2(data, m_threshold = 50):
    # Convert latitudes and longitudes to radians
    orig_lat_rad = np.radians(data['orig_lat'])
    orig_long_rad = np.radians(data['orig_long'])

    # Compute home location coordinates
    home_lat_rad = np.radians(data[(data['Hour'] <= 6) | (data['Hour'] >= 22)]['orig_lat'].mode().item())
    home_long_rad = np.radians(data[(data['Hour'] <= 6) | (data['Hour'] >= 22)]['orig_long'].mode().item())

    # Calculate distance between each coordinate and the home location
    dist = cdist([(home_lat_rad, home_long_rad)], np.column_stack((orig_lat_rad, orig_long_rad)))
    in_bounds = (dist <= m_threshold).flatten().astype(int)

    data['home'] = in_bounds

    return np.degrees(home_lat_rad), np.degrees(home_long_rad)

def LI(X_train, X_test, y_train, y_test):
    preds_lat = []
    preds_long = []
    for i,j in enumerate(X_test):
        preds_lat.append(np.interp(j, X_train, y_train[:,0]))
        preds_long.append(np.interp(j, X_train, y_train[:,1]))

    preds_lat = np.array(preds_lat)
    preds_long = np.array(preds_long)

    return preds_lat, preds_long

def Multi_Trip_TrainTestSplit(data, test_start_date, test_end_date, output = 'coords'):    

    # IMPORTANT: these lines find the index at which the temporal input dimensions are located within the full dataframe
    weeks_col = pd.get_dummies(data['Week_of_Month']).columns
    data = data.rename(columns={'datetime': 'Date_Time', 'lat': 'orig_lat', 'lng': 'orig_long' })

    inputstart = "unix_start_t_min"
    inputend = "week_" + str(weeks_col[-1])

    time_start_loc = data.columns.get_loc(inputstart)
    time_end_loc = data.columns.get_loc(inputend) + 1

    test_start_date = test_start_date
    test_end_date = test_end_date

    X_train = torch.tensor(np.asarray(data.iloc[:, time_start_loc:time_end_loc][
        (data['Date_Time'] < test_start_date) | (data['Date_Time'] > test_end_date)
        ]).astype(np.float))
    X_test = torch.tensor(np.asarray(data.iloc[:, time_start_loc:time_end_loc][
        (data['Date_Time'] >= test_start_date) & (data['Date_Time'] <= test_end_date)
        ]).astype(np.float))
    y_train_lat = torch.tensor(np.asarray(data['orig_lat'][
        (data['Date_Time'] < test_start_date) | (data['Date_Time'] > test_end_date)
        ]).astype(np.float))
    y_test_lat = torch.tensor(np.asarray(data['orig_lat'][
        (data['Date_Time'] >= test_start_date) & (data['Date_Time'] <= test_end_date)
        ]).astype(np.float))
    y_train_long = torch.tensor(np.asarray(data['orig_long'][
        (data['Date_Time'] < test_start_date) | (data['Date_Time'] > test_end_date)
        ]).astype(np.float))
    y_test_long = torch.tensor(np.asarray(data['orig_long'][
        (data['Date_Time'] >= test_start_date) & (data['Date_Time'] <= test_end_date)
        ]).astype(np.float))
    glob_t_train = torch.tensor(np.asarray(data['unix_start_t_min'][
        (data['Date_Time'] < test_start_date) | (data['Date_Time'] > test_end_date)
        ]).astype(np.float))
    glob_t_test = torch.tensor(np.asarray(data['unix_start_t_min'][
        (data['Date_Time'] >= test_start_date) & (data['Date_Time'] <= test_end_date)
        ]).astype(np.float))
    date_train = data['Date_Time'][
        (data['Date_Time'] < test_start_date) | (data['Date_Time'] > test_end_date)
        ]
    date_test = data['Date_Time'][
        (data['Date_Time'] >= test_start_date) & (data['Date_Time'] <= test_end_date)
        ]
    #vel_train = torch.tensor(np.asarray(data['vel'][
    #    (data['Date_Time'] < test_start_date) | (data['Date_Time'] > test_end_date)
    #    ]).astype(np.float))
    #vel_test = torch.tensor(np.asarray(data['vel'][
    #    (data['Date_Time'] >= test_start_date) & (data['Date_Time'] > test_end_date)
    #    ]).astype(np.float))
    #angle_train = torch.tensor(np.asarray(data['angle'][
    #    (data['Date_Time'] < test_start_date) | (data['Date_Time'] > test_end_date)
    #    ]).astype(np.float))
    #angle_test = torch.tensor(np.asarray(data['angle'][
    #    (data['Date_Time'] >= test_start_date) & (data['Date_Time'] > test_end_date)
    #    ]).astype(np.float))
    prec_train = torch.tensor(np.asarray(data['orig_unc'][
        (data['Date_Time'] < test_start_date) | (data['Date_Time'] > test_end_date)
        ]).astype(np.float))
    prec_test = torch.tensor(np.asarray(data['orig_unc'][
        (data['Date_Time'] >= test_start_date) & (data['Date_Time'] > test_end_date)
        ]).astype(np.float))

    if output == 'coords':
        y_train = torch.stack(
            [
                y_train_lat, y_train_long 
                ], -1)
        y_test = torch.stack(
            [
                y_test_lat, y_test_long
                ], -1)
    elif output == 'home':
        y_train = torch.tensor(np.asarray(data['home'][
            (data['Date_Time'] < test_start_date) | (data['Date_Time'] > test_end_date)
            ]).astype(np.float))
        y_test = torch.tensor(np.asarray(data['home'][
            (data['Date_Time'] >= test_start_date) & (data['Date_Time'] <= test_end_date)
            ]).astype(np.float))
    #else:
    #    MO_train = torch.stack([dist_train, angle_train], -1)
    #    MO_test = torch.stack([dist_test, angle_test], -1)

    train = pd.DataFrame({'unix_start_t_min': glob_t_train,
                               'date': date_train,
                               'lat': [y_train_lat[i].item() for i in range(0, len(y_train_lat))],
                               'long': [y_train_long[i].item() for i in range(0, len(y_train_long))]},
                              columns = ['unix_start_t_min', 'date', 'lat', 'long'])
    train = train.sort_values(by=['unix_start_t_min'])

    test = pd.DataFrame({'unix_start_t_min': glob_t_test,
                       'date': date_test,
                       'lat': [y_test_lat[i].item() for i in range(0, len(y_test_lat))],
                       'long': [y_test_long[i].item() for i in range(0, len(y_test_long))]},
                      columns = ['unix_start_t_min', 'date', 'lat', 'long'])
    test = test.sort_values(by=['unix_start_t_min'])
    
    return train, test, X_train, y_train, X_test, y_test, date_train, date_test