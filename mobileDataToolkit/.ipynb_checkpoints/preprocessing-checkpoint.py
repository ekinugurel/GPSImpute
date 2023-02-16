import pandas as pd
import datetime
from datetime import datetime
import numpy as np
import holidays
import matplotlib
from matplotlib import pyplot as plt, cm
import math
import torch
import utils.helper_func as helper_func

us_holidays = holidays.US()

class dp_MultiTrip():
    def __init__(self, data=None, file_path=None, random_state=None):
        self.file_path = file_path
        self.random_state = random_state
        
        if file_path is not None:
            self.read_data()
        elif data is not None:
            self.data = data

    def read_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def add_DateTime(self):
        Date_Time = list()
        for i in self.data['unix_start_t']:
            Date_Time.append(datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S'))
        self.data['Date_Time'] = Date_Time
        self.data['Date_Time'] = pd.to_datetime(self.data['Date_Time'])
    
    def chooseUser(self, user_ID):
        self.user_ID = user_ID
        if self.data.user_ID in self.data.columns:
            self.data = self.data[self.data.user_ID == self.user_ID]
        elif self.data.UID in self.data.columns:
            self.data = self.data[self.data.UID == self.user_ID]
        
    def sample(self, npoints, random_state):
        self.samples = self.data.sample(n = npoints, random_state = random_state)
        return self.samples
        
    def subsetByTripID(self, trip_ID):
        self.data = self.data[self.data['trip_ID'] == trip_ID]
        self.data = self.data.reset_index()
        return self.data    
        
    def subsetByTime(self, starttime, endtime):
        self.data = self.data[(self.data['Date_Time'] >= starttime) & (self.data['Date_Time'] <= endtime)]
        self.data = self.data.reset_index()
        return self.data
    
    def subset(self, TOD_start, TOD_end, DOW, TOD_b=True, DOW_b=True):
        if DOW_b == True:
            self.data = self.data[(self.data['Day of Week'] == DOW)]
        if TOD_b == True:
            self.data = self.data[(self.data['sec_after_midnight'] >= TOD_start) & (self.data['sec_after_midnight'] <= TOD_end)]
        return self.data
    
    def tempOcp(self, test = True, bin_len = 5):
        """
        Calculates the temporal occupancy of a given mobile data sequence.

        Parameters
        ----------
        bin_len : INT, optional
            Number of minutes in a time interval (bin). The default is 5.
        test: INT, optional
            Boolean signifying whether to obtain the temporal occupancy of the test set only (if it exists)

        Returns
        -------
        temp_ocp : FLOAT
            Temporal occupancy metric.

        """
        if test == True:
            data = self.subsetByTime(self.test_start_date, self.test_end_date)
        else:
            data = self.data
            
        bins = np.arange(min(data['unix_start_t_min']), max(data['unix_start_t_min'])+1, bin_len )
        Nb = len(bins)
        obs = list()
        for i, j in enumerate(bins):
            if i == Nb:
                break
            hi = list()
            for k in range(bins[i], bins[i+1]):
                condition = [k in data['unix_start_t_min'].values]
                hi.append(any(condition))
            if any(hi):
                obs.append(1)
            if i == (Nb-2):
                break
        obs.append(1)
        temp_ocp = float(len(obs) / len(bins))
        return temp_ocp
    
    def Multi_Trip_Preprocess(self):
        self.data = self.data[self.data.user_ID == self.user_ID]

        self.data['Day of Week'] = self.data['Date_Time'].dt.dayofweek
        self.data['Year'] = self.data['Date_Time'].dt.year
        self.data['Month'] = self.data['Date_Time'].dt.month
        self.data['Day'] = self.data['Date_Time'].dt.day
        self.data['Week'] = self.data['Date_Time'].dt.week
        self.data['Hour'] = self.data['Date_Time'].dt.hour
        self.data['Week_of_Month'] = pd.to_numeric(self.data['Date_Time'].dt.day/7).apply(lambda x: math.ceil(x))

            
        self.n_weeks = self.data.Week.nunique()
        self.prec = np.asarray(self.data['orig_unc'])
        
        mean_lat = self.data['orig_lat'].mean()
        mean_long = self.data['orig_long'].mean()
        stdev_lat = self.data['orig_lat'].std()
        stdev_long = self.data['orig_long'].std()
            
        self.data['norm_lat'] = [np.array(((i - mean_lat) / stdev_lat), dtype=np.float32) for i in self.data['orig_lat']]
        self.data['norm_long'] = [np.array(((i - mean_long) / stdev_long), dtype=np.float32) for i in self.data['orig_long']]
        
        self.data['unix_start_t_min'] = ( (self.data['unix_start_t'] - min(self.data['unix_start_t'])) / 60 ).astype(int) #
        self.data['sec_after_midnight'] = (self.data['Date_Time'].dt.hour*3600)+(self.data['Date_Time'].dt.minute*60)+(self.data['Date_Time'].dt.second)
        #self.data['minute'] = self.data['Date_Time'].dt.hour * 60 + self.data['Date_Time'].dt.minute
        #self.data['15_mins'] = math.ceil(self.data['minute'] / 15)
        
        holiday = list()
        for i in self.data['Date_Time']:
            holiday.append(1 if (i in us_holidays) else 0)
        self.data['Holiday'] = holiday
        
        weekend = list()
        for i in self.data['Date_Time']:
            weekend.append(1 if (i.weekday() >= 5) else 0)
        self.data['weekend'] = weekend
        
        AM_peak = list()
        for i in self.data['Date_Time']:
            AM_peak.append(1 if ((i.hour >= 6) & (i.hour < 10)) else 0)
        self.data['AM_peak'] = AM_peak
        
        PM_peak = list()
        for i in self.data['Date_Time']:
            PM_peak.append(1 if ((i.hour >= 15) & (i.hour < 19)) else 0)
        self.data['PM_peak'] = PM_peak
        
        days = pd.get_dummies(self.data['Day of Week']).to_numpy()
        self.days_col = pd.get_dummies(self.data['Day of Week']).columns
        days_ind = []
        for i in self.days_col:
            days_ind.append("day_" + str(i))
            
        self.data[days_ind] = days

        weeks = pd.get_dummies(self.data['Week_of_Month']).to_numpy()
        self.weeks_col = pd.get_dummies(self.data['Week_of_Month']).columns
        week_ind = []
        for i in self.weeks_col:
            week_ind.append("week_" + str(i))
        
        self.data[week_ind] = weeks
            
        months = pd.get_dummies(self.data['Month']).to_numpy()
        self.months_col = pd.get_dummies(self.data['Month']).columns
        month_ind = []
        for i in self.months_col:
            month_ind.append("month_" + str(i))
            
        self.data[month_ind] = months        
                
        self.data = self.data.reset_index()
    
    def mobVisualize(self, data):
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        f, (y1_ax, y2_ax) = plt.subplots(2, 1, constrained_layout = True)
        if len(self.data.user_ID.unique()) == 1:
            f.suptitle('User ID: ' + str(self.data.user_ID.unique().item()), fontsize = 10)
        
        y1_ax.scatter(data['Date_Time'], data['norm_lat'], marker='.', c='blue')
        y1_ax.set_title('(Normalized) Latitude', fontsize = 10)
        y1_ax.set_xticks([])
        
        y2_ax.scatter(data['Date_Time'], data['norm_long'], marker='.', c='blue')
        y2_ax.set_title('(Normalized) Longitude', fontsize = 10)
        
        try:
            y1_ax.fill_between(data['Date_Time'], 0, 1, 
                               where=((data['Date_Time'] >= self.test_start_date) & (data['Date_Time'] <= self.test_end_date)), 
                               color='pink', alpha=0.5, label = 'Testing period', transform=y1_ax.get_xaxis_transform())
            y2_ax.fill_between(data['Date_Time'], 0, 1, 
                               where=(data['Date_Time'] >= self.test_start_date) & (data['Date_Time'] <= self.test_end_date), 
                               color='pink', alpha=0.5, label = 'Testing period', transform=y2_ax.get_xaxis_transform())
        except AttributeError:
            pass
        #y1_ax.set_ylim([min(self.data['orig_lat']), max(self.data['orig_lat'])])
        #y2_ax.set_ylim([min(self.data['orig_long']), max(self.data['orig_long'])])

        #y1_ax.legend(bbox_to_anchor=(0, 1.05, 1, 0.2), loc="lower left",
        #        borderaxespad=0)
        #y2_ax.legend(bbox_to_anchor=(0, 1.05, 1, 0.2), loc="lower left",
        #        borderaxespad=0)
        plt.xticks(rotation = 30, fontsize = 8)
        plt.xlabel('Date', fontsize=10)
        
        plt.show()
    
    def homeLoc(self, m_threshold = 50):
        self.home_lat = self.data[(self.data['Hour'] <= 6) | (self.data['Hour'] >= 22)]['orig_lat'].mode().item()
        self.home_long = self.data[(self.data['Hour'] <= 6) | (self.data['Hour'] >= 22)]['orig_long'].mode().item()
        
        upper_b_lat, upper_b_long = helper_func.newCoords(self.home_lat, self.home_long, m_threshold, 0)
        right_b_lat, right_b_long = helper_func.newCoords(self.home_lat, self.home_long, 0, m_threshold)
        lower_b_lat, lower_b_long = helper_func.newCoords(self.home_lat, self.home_long, -m_threshold, 0)
        left_b_lat, left_b_long = helper_func.newCoords(self.home_lat, self.home_long, 0, -m_threshold)
        
        home = []
        
        for i, j in enumerate(self.data['orig_lat']):
            if (
                (self.data['orig_lat'][i] > upper_b_lat) or
                (self.data['orig_long'][i] > right_b_long) or
                (self.data['orig_lat'][i] < lower_b_lat) or
                (self.data['orig_long'][i] < left_b_long)
            ):
                home.append(0)
            else:
                home.append(1)
        self.data['home'] = home

        return self.home_lat, self.home_long

        
    def Multi_Trip_TrainTestSplit(self, test_start_date, test_end_date, output = 'coords'):    
        
        # IMPORTANT: these lines find the index at which the temporal input dimensions are located within the full dataframe
        self.inputstart = "unix_start_t_min"
        self.inputend = "week_" + str(self.weeks_col[-1])
        
        time_start_loc = self.data.columns.get_loc(self.inputstart)
        time_end_loc = self.data.columns.get_loc(self.inputend) + 1
        
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        
        self.X_train = torch.tensor(np.asarray(self.data.iloc[:, time_start_loc:time_end_loc][
            (self.data['Date_Time'] < self.test_start_date) | (self.data['Date_Time'] > self.test_end_date)
            ]).astype(np.float))
        self.X_test = torch.tensor(np.asarray(self.data.iloc[:, time_start_loc:time_end_loc][
            (self.data['Date_Time'] >= self.test_start_date) & (self.data['Date_Time'] <= self.test_end_date)
            ]).astype(np.float))
        self.y_train_lat = torch.tensor(np.asarray(self.data['orig_lat'][
            (self.data['Date_Time'] < test_start_date) | (self.data['Date_Time'] > test_end_date)
            ]).astype(np.float))
        self.y_test_lat = torch.tensor(np.asarray(self.data['orig_lat'][
            (self.data['Date_Time'] >= self.test_start_date) & (self.data['Date_Time'] <= self.test_end_date)
            ]).astype(np.float))
        self.y_train_long = torch.tensor(np.asarray(self.data['orig_long'][
            (self.data['Date_Time'] < self.test_start_date) | (self.data['Date_Time'] > self.test_end_date)
            ]).astype(np.float))
        self.y_test_long = torch.tensor(np.asarray(self.data['orig_long'][
            (self.data['Date_Time'] >= self.test_start_date) & (self.data['Date_Time'] <= self.test_end_date)
            ]).astype(np.float))
        self.glob_t_train = torch.tensor(np.asarray(self.data['unix_start_t_min'][
            (self.data['Date_Time'] < self.test_start_date) | (self.data['Date_Time'] > self.test_end_date)
            ]).astype(np.float))
        self.glob_t_test = torch.tensor(np.asarray(self.data['unix_start_t_min'][
            (self.data['Date_Time'] >= self.test_start_date) & (self.data['Date_Time'] <= self.test_end_date)
            ]).astype(np.float))
        self.date_train = self.data['Date_Time'][
            (self.data['Date_Time'] < self.test_start_date) | (self.data['Date_Time'] > self.test_end_date)
            ]
        self.date_test = self.data['Date_Time'][
            (self.data['Date_Time'] >= self.test_start_date) & (self.data['Date_Time'] <= self.test_end_date)
            ]
        #self.vel_train = torch.tensor(np.asarray(self.data['vel'][
        #    (self.data['Date_Time'] < self.test_start_date) | (self.data['Date_Time'] > self.test_end_date)
        #    ]).astype(np.float))
        #self.vel_test = torch.tensor(np.asarray(self.data['vel'][
        #    (self.data['Date_Time'] >= self.test_start_date) & (self.data['Date_Time'] > self.test_end_date)
        #    ]).astype(np.float))
        #self.angle_train = torch.tensor(np.asarray(self.data['angle'][
        #    (self.data['Date_Time'] < self.test_start_date) | (self.data['Date_Time'] > self.test_end_date)
        #    ]).astype(np.float))
        #self.angle_test = torch.tensor(np.asarray(self.data['angle'][
        #    (self.data['Date_Time'] >= self.test_start_date) & (self.data['Date_Time'] > self.test_end_date)
        #    ]).astype(np.float))
        self.prec_train = torch.tensor(np.asarray(self.data['orig_unc'][
            (self.data['Date_Time'] < self.test_start_date) | (self.data['Date_Time'] > self.test_end_date)
            ]).astype(np.float))
        self.prec_test = torch.tensor(np.asarray(self.data['orig_unc'][
            (self.data['Date_Time'] >= self.test_start_date) & (self.data['Date_Time'] > self.test_end_date)
            ]).astype(np.float))
        
        if output == 'coords':
            self.y_train = torch.stack(
                [
                    self.y_train_lat, self.y_train_long 
                    ], -1)
            self.y_test = torch.stack(
                [
                    self.y_test_lat, self.y_test_long
                    ], -1)
        elif output == 'home':
            self.y_train = torch.tensor(np.asarray(self.data['home'][
                (self.data['Date_Time'] < test_start_date) | (self.data['Date_Time'] > test_end_date)
                ]).astype(np.float))
            self.y_test = torch.tensor(np.asarray(self.data['home'][
                (self.data['Date_Time'] >= self.test_start_date) & (self.data['Date_Time'] <= self.test_end_date)
                ]).astype(np.float))
        #else:
        #    self.MO_train = torch.stack([self.dist_train, self.angle_train], -1)
        #    self.MO_test = torch.stack([self.dist_test, self.angle_test], -1)
        
        self.train = pd.DataFrame({'unix_start_t_min': self.glob_t_train,
                                   'date': self.date_train,
                                   'lat': [self.y_train_lat[i].item() for i in range(0, len(self.y_train_lat))],
                                   'long': [self.y_train_long[i].item() for i in range(0, len(self.y_train_long))]},
                                  columns = ['unix_start_t_min', 'date', 'lat', 'long'])
        self.train = self.train.sort_values(by=['unix_start_t_min'])
        
        self.test = pd.DataFrame({'unix_start_t_min': self.glob_t_test,
                           'date': self.date_test,
                           'lat': [self.y_test_lat[i].item() for i in range(0, len(self.y_test_lat))],
                           'long': [self.y_test_long[i].item() for i in range(0, len(self.y_test_long))]},
                          columns = ['unix_start_t_min', 'date', 'lat', 'long'])
        self.test = self.test.sort_values(by=['unix_start_t_min'])