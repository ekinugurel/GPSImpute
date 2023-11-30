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
    # This class is for data preprocessing for multi-trip data
    # It is used to preprocess the data for the multi-trip model
    def __init__(self, data=None, file_path=None, random_state=None):
        '''
        data: pandas dataframe
        file_path: path to the csv file
        random_state: random state for sampling
        '''
        self.file_path = file_path
        self.random_state = random_state
        
        if file_path is not None:
            self.read_data(file_path)
        elif data is not None:
            self.data = data

    def read_data(self, file_path):
        '''
        read data from csv file
        '''
        self.data = pd.read_csv(file_path)
        return self.data
    
    def add_DateTime(self, unix_col='unix_start_t'):
        date = list()
        for i in self.data[unix_col]:
            date.append(datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S'))
        self.data['datetime'] = date
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])

    def add_NormCoords(self):    
        mean_lat = self.data['orig_lat'].mean()
        mean_long = self.data['orig_long'].mean()
        stdev_lat = self.data['orig_lat'].std()
        stdev_long = self.data['orig_long'].std()
            
        self.data['norm_lat'] = [np.array(((i - mean_lat) / stdev_lat), dtype=np.float32) for i in self.data['orig_lat']]
        self.data['norm_long'] = [np.array(((i - mean_long) / stdev_long), dtype=np.float32) for i in self.data['orig_long']]
        
    
    def chooseUser(self, UID):
        return self.data[self.data.UID == UID]
        
    def sample(self, npoints, random_state):
        self.samples = self.data.sample(n = npoints, random_state = random_state)
        return self.samples
        
    def subsetByTripID(self, trip_ID):
        return self.data[self.data['trip_ID'] == trip_ID].reset_index()
    
    def subsetByTime(self, starttime, endtime):
        return self.data[(self.data['datetime'] >= starttime) & (self.data['datetime'] <= endtime)].reset_index()
    
    def subset(self, TOD_start, TOD_end, DOW, TOD_b=True, DOW_b=True):
        if DOW_b == True:
            self.data = self.data[(self.data['DoW'] == DOW)]
        if TOD_b == True:
            self.data = self.data[(self.data['SaM'] >= TOD_start) & (self.data['SaM'] <= TOD_end)]
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
            
        bins = np.arange(min(data['unix_min']), max(data['unix_min'])+1, bin_len )
        Nb = len(bins)
        obs = list()
        for i, j in enumerate(bins):
            if i == Nb:
                break
            hi = list()
            for k in range(bins[i], bins[i+1]):
                condition = [k in data['unix_min'].values]
                hi.append(any(condition))
            if any(hi):
                obs.append(1)
            if i == (Nb-2):
                break
        obs.append(1)
        temp_ocp = float(len(obs) / len(bins))
        return temp_ocp
    
    def Multi_Trip_Preprocess(self, datetime='datetime', lat='orig_lat', long='orig_long', 
    monthly_dummies=False, weekly_dummies=False):
        #self.data = self.data[self.data.UID == self.UID]
        self.data[datetime] = pd.to_datetime(self.data[datetime])

        self.data['DoW'] = self.data[datetime].dt.dayofweek
        self.data['Year'] = self.data[datetime].dt.year
        self.data['Month'] = self.data[datetime].dt.month
        self.data['Day'] = self.data[datetime].dt.day
        self.data['Week'] = self.data[datetime].dt.week
        self.data['Hour'] = self.data[datetime].dt.hour
        self.data['WoM'] = pd.to_numeric(self.data[datetime].dt.day/7).apply(lambda x: math.ceil(x))

        self.n_weeks = self.data.Week.nunique()
        self.prec = np.asarray(self.data['orig_unc'])
        
        mean_lat = self.data[lat].mean()
        mean_long = self.data[long].mean()
        stdev_lat = self.data[lat].std()
        stdev_long = self.data[long].std()
            
        self.data['norm_lat'] = [np.array(((i - mean_lat) / stdev_lat), dtype=np.float32) for i in self.data[lat]]
        self.data['norm_long'] = [np.array(((i - mean_long) / stdev_long), dtype=np.float32) for i in self.data[long]]
        
        # self.data['unix_min'] = ( (self.data['unix_start_t'] - min(self.data['unix_start_t'])) / 60 ).astype(int) #
        self.data['SaM'] = (self.data[datetime].dt.hour*3600)+(self.data[datetime].dt.minute*60)+(self.data[datetime].dt.second)
        #self.data['minute'] = self.data['datetime'].dt.hour * 60 + self.data['datetime'].dt.minute
        #self.data['15_mins'] = math.ceil(self.data['minute'] / 15)
        
        holiday = list()
        for i in self.data[datetime]:
            holiday.append(1 if (i in us_holidays) else 0)
        self.data['Holiday'] = holiday
        
        weekend = list()
        for i in self.data[datetime]:
            weekend.append(1 if (i.weekday() >= 5) else 0)
        self.data['weekend'] = weekend
        
        AM_peak = list()
        for i in self.data[datetime]:
            AM_peak.append(1 if ((i.hour >= 6) & (i.hour < 10)) else 0)
        self.data['AM_peak'] = AM_peak
        
        PM_peak = list()
        for i in self.data[datetime]:
            PM_peak.append(1 if ((i.hour >= 15) & (i.hour < 19)) else 0)
        self.data['PM_peak'] = PM_peak
        
        days = pd.get_dummies(self.data['DoW']).to_numpy()
        self.days_col = pd.get_dummies(self.data['DoW']).columns
        days_ind = []
        for i in self.days_col:
            days_ind.append("day_" + str(i))
            
        self.data[days_ind] = days
        if weekly_dummies == True:
            weeks = pd.get_dummies(self.data['WoM']).to_numpy()
            self.weeks_col = pd.get_dummies(self.data['WoM']).columns
            week_ind = []
            for i in self.weeks_col:
                week_ind.append("week_" + str(i))
            self.data[week_ind] = weeks

        if monthly_dummies == True:    
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
        if len(self.data.UID.unique()) == 1:
            f.suptitle('User ID: ' + str(self.data.UID.unique().item()), fontsize = 10)
        
        y1_ax.scatter(data['datetime'], data['norm_lat'], c='blue')
        y1_ax.set_title('(Normalized) Latitude', fontsize = 10)
        y1_ax.set_xticks([])
        
        y2_ax.scatter(data['datetime'], data['norm_long'], c='blue')
        y2_ax.set_title('(Normalized) Longitude', fontsize = 10)
        
        try:
            y1_ax.fill_between(data['datetime'], 0, 1, 
                               where=((data['datetime'] >= self.test_start_date) & (data['datetime'] <= self.test_end_date)), 
                               color='pink', alpha=0.5, label = 'Testing period', transform=y1_ax.get_xaxis_transform())
            y2_ax.fill_between(data['datetime'], 0, 1, 
                               where=(data['datetime'] >= self.test_start_date) & (data['datetime'] <= self.test_end_date), 
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

    
    def Multi_Trip_TrainTestSplit(self, test_start_date, test_end_date, training_index=set(), testing_index=set(),
                                  datetime = 'datetime', lat='orig_lat', 
                                  long='orig_long', output = 'coords', unix='unix_min', 
                                  inputstart = 'unix_min', inputend = "week_5"):    
        
        # IMPORTANT: these lines find the index at which the temporal input dimensions are located within the full dataframe
        self.inputstart = inputstart
        self.inputend = inputend
        
        time_start_loc = self.data.columns.get_loc(self.inputstart)
        time_end_loc = self.data.columns.get_loc(self.inputend) + 1
        
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date

        if (training_index is not None) & (test_start_date is not None) & (test_end_date is not None) & (testing_index is not None):
            self.X_train = torch.tensor(np.asarray(self.data.iloc[:, time_start_loc:time_end_loc][
                (self.data[datetime] < self.test_start_date) | (self.data[datetime] > self.test_end_date) |
                self.data[unix].isin(training_index) | ~self.data[unix].isin(testing_index)
                ]).astype(float))
            self.X_test = torch.tensor(np.asarray(self.data.iloc[:, time_start_loc:time_end_loc][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] <= self.test_end_date) &
                ~self.data[unix].isin(training_index) & (self.data[unix].isin(testing_index))
                ]).astype(float))
            self.y_train_lat = torch.tensor(np.asarray(self.data[lat][
                (self.data[datetime] < test_start_date) | (self.data[datetime] > test_end_date) |
                self.data[unix].isin(training_index) | ~self.data[unix].isin(testing_index)
                ]).astype(float))
            self.y_test_lat = torch.tensor(np.asarray(self.data[lat][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] <= self.test_end_date) &
                ~self.data[unix].isin(training_index) & (self.data[unix].isin(testing_index))
                ]).astype(float))
            self.y_train_long = torch.tensor(np.asarray(self.data[long][
                (self.data[datetime] < self.test_start_date) | (self.data[datetime] > self.test_end_date) |
                self.data[unix].isin(training_index) | ~self.data[unix].isin(testing_index)
                ]).astype(float))
            self.y_test_long = torch.tensor(np.asarray(self.data[long][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] <= self.test_end_date) &
                ~self.data[unix].isin(training_index) & (self.data[unix].isin(testing_index))
                ]).astype(float))
            self.glob_t_train = torch.tensor(np.asarray(self.data[unix][
                (self.data[datetime] < self.test_start_date) | (self.data[datetime] > self.test_end_date) |
                self.data[unix].isin(training_index) | ~self.data[unix].isin(testing_index)
                ]).astype(float))
            self.glob_t_test = torch.tensor(np.asarray(self.data[unix][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] <= self.test_end_date) &
                ~self.data[unix].isin(training_index) & (self.data[unix].isin(testing_index))
                ]).astype(float))
            self.date_train = self.data[datetime][
                (self.data[datetime] < self.test_start_date) | (self.data[datetime] > self.test_end_date) |
                self.data[unix].isin(training_index) | ~self.data[unix].isin(testing_index)
                ]
            self.date_test = self.data[datetime][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] <= self.test_end_date) &
                ~self.data[unix].isin(training_index) & (self.data[unix].isin(testing_index))
                ]
            self.prec_train = torch.tensor(np.asarray(self.data['orig_unc'][
                (self.data[datetime] < self.test_start_date) | (self.data[datetime] > self.test_end_date) |
                self.data[unix].isin(training_index) | ~self.data[unix].isin(testing_index)
                ]).astype(float))
            self.prec_test = torch.tensor(np.asarray(self.data['orig_unc'][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] > self.test_end_date) &
                ~self.data[unix].isin(training_index) & (self.data[unix].isin(testing_index))
                ]).astype(float))
        elif training_index is None:
            self.X_train = torch.tensor(np.asarray(self.data.iloc[:, time_start_loc:time_end_loc][
                (self.data[datetime] < self.test_start_date) | (self.data[datetime] > self.test_end_date)
                ]).astype(float))
            self.X_test = torch.tensor(np.asarray(self.data.iloc[:, time_start_loc:time_end_loc][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] <= self.test_end_date)
                ]).astype(float))
            self.y_train_lat = torch.tensor(np.asarray(self.data[lat][
                (self.data[datetime] < test_start_date) | (self.data[datetime] > test_end_date)
                ]).astype(float))
            self.y_test_lat = torch.tensor(np.asarray(self.data[lat][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] <= self.test_end_date)
                ]).astype(float))
            self.y_train_long = torch.tensor(np.asarray(self.data[long][
                (self.data[datetime] < self.test_start_date) | (self.data[datetime] > self.test_end_date)
                ]).astype(float))
            self.y_test_long = torch.tensor(np.asarray(self.data[long][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] <= self.test_end_date)
                ]).astype(float))
            self.glob_t_train = torch.tensor(np.asarray(self.data[unix][
                (self.data[datetime] < self.test_start_date) | (self.data[datetime] > self.test_end_date)
                ]).astype(float))
            self.glob_t_test = torch.tensor(np.asarray(self.data[unix][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] <= self.test_end_date)
                ]).astype(float))
            self.date_train = self.data[datetime][
                (self.data[datetime] < self.test_start_date) | (self.data[datetime] > self.test_end_date)
                ]
            self.date_test = self.data[datetime][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] <= self.test_end_date)
                ]
            self.prec_train = torch.tensor(np.asarray(self.data['orig_unc'][
                (self.data[datetime] < self.test_start_date) | (self.data[datetime] > self.test_end_date)
                ]).astype(float))
            self.prec_test = torch.tensor(np.asarray(self.data['orig_unc'][
                (self.data[datetime] >= self.test_start_date) & (self.data[datetime] > self.test_end_date)
                ]).astype(float))
        else:
            self.X_train = torch.tensor(np.asarray(self.data.iloc[:, time_start_loc:time_end_loc][
                self.data[unix].isin(training_index)
                ]).astype(float))
            self.X_test = torch.tensor(np.asarray(self.data.iloc[:, time_start_loc:time_end_loc][
                self.data[unix].isin(testing_index)
                ]).astype(float))
            self.y_train_lat = torch.tensor(np.asarray(self.data[lat][
                self.data[unix].isin(training_index)
                ]).astype(float))
            self.y_test_lat = torch.tensor(np.asarray(self.data[lat][
                self.data[unix].isin(testing_index)
                ]).astype(float))
            self.y_train_long = torch.tensor(np.asarray(self.data[long][
                self.data[unix].isin(training_index)
                ]).astype(float))
            self.y_test_long = torch.tensor(np.asarray(self.data[long][
                self.data[unix].isin(testing_index)
                ]).astype(float))
            self.glob_t_train = torch.tensor(np.asarray(self.data[unix][
                self.data[unix].isin(training_index)
                ]).astype(float))
            self.glob_t_test = torch.tensor(np.asarray(self.data[unix][
                self.data[unix].isin(testing_index)
                ]).astype(float))
            self.date_train = self.data[datetime][
                self.data[unix].isin(training_index)
                ]
            self.date_test = self.data[datetime][
                self.data[unix].isin(testing_index)
                ]
            self.prec_train = torch.tensor(np.asarray(self.data['orig_unc'][
                self.data[unix].isin(training_index)
                ]).astype(float))
            self.prec_test = torch.tensor(np.asarray(self.data['orig_unc'][
                self.data[unix].isin(testing_index)
                ]).astype(float))
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
                (self.data['datetime'] < test_start_date) | (self.data['datetime'] > test_end_date)
                ]).astype(float))
            self.y_test = torch.tensor(np.asarray(self.data['home'][
                (self.data['datetime'] >= self.test_start_date) & (self.data['datetime'] <= self.test_end_date)
                ]).astype(float))
        #else:
        #    self.MO_train = torch.stack([self.dist_train, self.angle_train], -1)
        #    self.MO_test = torch.stack([self.dist_test, self.angle_test], -1)
        
        self.train = pd.DataFrame({'unix_min': self.glob_t_train,
                                   'date': self.date_train,
                                   'lat': [self.y_train_lat[i].item() for i in range(0, len(self.y_train_lat))],
                                   'long': [self.y_train_long[i].item() for i in range(0, len(self.y_train_long))]},
                                  columns = ['unix_min', 'date', 'lat', 'long'])
        self.train = self.train.sort_values(by=['unix_min'])
        
        self.test = pd.DataFrame({'unix_min': self.glob_t_test,
                           'date': self.date_test,
                           'lat': [self.y_test_lat[i].item() for i in range(0, len(self.y_test_lat))],
                           'long': [self.y_test_long[i].item() for i in range(0, len(self.y_test_long))]},
                          columns = ['unix_min', 'date', 'lat', 'long'])
        self.test = self.test.sort_values(by=['unix_min'])