from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
import numpy as np
import similaritymeasures as sm
import pandas as pd
import glob 
import os


class Metrics:
    def __init__(self, time_gap, results_files, params_files, skmob_files_path):
        self.time_gap = time_gap
        self.results_files = results_files
        self.params_files = params_files
        self.skmob_files_path = skmob_files_path

    def classicalMetrics(self):
        results_ids = [f.split('_')[3].split('.')[0] for f in self.results_files]
        params_ids = [f.split('_')[2].split('.')[0] for f in self.params_files]

        self.mtgp_mae = np.array([])
        self.mtgp_rmse = np.array([])
        self.mtgp_mad = np.array([])
        self.mtgp_mape = np.array([])
        self.mtgp_maxape = np.array([])
        self.mtgp_tape = np.array([])
        self.mtgp_df = np.array([])
        self.mtgp_area = np.array([])
        self.mtgp_dtw = np.array([])
        self.rbf_mae = np.array([])
        self.rbf_rmse = np.array([])
        self.rbf_mad = np.array([])
        self.rbf_mape = np.array([])
        self.rbf_maxape = np.array([])
        self.rbf_tape = np.array([])
        self.rbf_df = np.array([])
        self.rbf_area = np.array([])
        self.rbf_dtw = np.array([])
        self.ses_mae = np.array([])
        self.ses_rmse = np.array([])
        self.ses_mad = np.array([])
        self.ses_mape = np.array([])
        self.ses_maxape = np.array([])
        self.ses_tape = np.array([])
        self.ses_df = np.array([])
        self.ses_area = np.array([])
        self.ses_dtw = np.array([])
        self.holt_mae = np.array([])
        self.holt_rmse = np.array([])
        self.holt_mad = np.array([])
        self.holt_mape = np.array([])
        self.holt_maxape = np.array([])
        self.holt_tape = np.array([])
        self.holt_df = np.array([])
        self.holt_area = np.array([])
        self.holt_dtw = np.array([])
        self.es_mae = np.array([])
        self.es_rmse = np.array([])
        self.es_mad = np.array([])
        self.es_mape = np.array([])
        self.es_maxape = np.array([])
        self.es_tape = np.array([])
        self.es_df = np.array([])
        self.es_area = np.array([])
        self.es_dtw = np.array([])
        self.arima_mae = np.array([])
        self.arima_rmse = np.array([])
        self.arima_mad = np.array([])
        self.arima_mape = np.array([])
        self.arima_maxape = np.array([])
        self.arima_tape = np.array([])
        self.arima_df = np.array([])
        self.arima_area = np.array([])
        self.arima_dtw = np.array([])
        self.sarima_mae = np.array([])
        self.sarima_rmse = np.array([])
        self.sarima_mad = np.array([])
        self.sarima_mape = np.array([])
        self.sarima_maxape = np.array([])
        self.sarima_tape = np.array([])
        self.sarima_df = np.array([])
        self.sarima_area = np.array([])
        self.sarima_dtw = np.array([])
        self.new_ocp = np.array([])
        self.init_lengthscale = np.array([])
        self.bic = np.array([])
            
        # Loop through each unique identifier and read corresponding files
        for id in set(results_ids):
            # First {} should be time gap, second {} should be id
            result_file = 'C:\\Users\\ekino\\OneDrive - UW\\GPR\\Sept_Results\\{}\\all_results\\results_{}.csv'.format(self.time_gap, id)
            params_file = 'C:\\Users\\ekino\\OneDrive - UW\\GPR\\Sept_Results\\{}\\all_parameters\\params_{}.csv'.format(self.time_gap, id)
            
            # Read both files
            res = pd.read_csv(result_file, header=0)
            par = pd.read_csv(params_file, header=0)
            par.columns = ['param', 'value']

            self.mtgp_mae = np.append(self.mtgp_mae, res['MAE'][0])
            self.mtgp_rmse = np.append(self.mtgp_rmse, res['RMSE'][0])
            self.mtgp_mad = np.append(self.mtgp_mad, res['MAD'][0])
            self.mtgp_mape = np.append(self.mtgp_mape, res['MAPE'][0])
            self.mtgp_maxape = np.append(self.mtgp_maxape, res['MAXAPE'][0])
            self.mtgp_tape = np.append(self.mtgp_tape, res['TAPE'][0])
            self.mtgp_df = np.append(self.mtgp_df, res['DF'][0])
            self.mtgp_area = np.append(self.mtgp_area, res['AREA'][0])
            self.mtgp_dtw = np.append(self.mtgp_dtw, res['DTW'][0])
            self.rbf_mae = np.append(self.rbf_mae, res['MAE'][1])
            self.rbf_rmse = np.append(self.rbf_rmse, res['RMSE'][1])
            self.rbf_mad = np.append(self.rbf_mad, res['MAD'][1])
            self.rbf_mape = np.append(self.rbf_mape, res['MAPE'][1])  
            self.rbf_maxape = np.append(self.rbf_maxape, res['MAXAPE'][1])
            self.rbf_tape = np.append(self.rbf_tape, res['TAPE'][1])
            self.rbf_df = np.append(self.rbf_df, res['DF'][1])
            self.rbf_area = np.append(self.rbf_area, res['AREA'][1])
            self.rbf_dtw = np.append(self.rbf_dtw, res['DTW'][1])
            self.ses_mae = np.append(self.ses_mae, res['MAE'][2])
            self.ses_rmse = np.append(self.ses_rmse, res['RMSE'][2])
            self.ses_mad = np.append(self.ses_mad, res['MAD'][2])
            self.ses_mape = np.append(self.ses_mape, res['MAPE'][2])
            self.ses_maxape = np.append(self.ses_maxape, res['MAXAPE'][2])
            self.ses_tape = np.append(self.ses_tape, res['TAPE'][2])
            self.ses_df = np.append(self.ses_df, res['DF'][2])
            self.ses_area = np.append(self.ses_area, res['AREA'][2])
            self.ses_dtw = np.append(self.ses_dtw, res['DTW'][2])
            self.holt_mae = np.append(self.holt_mae, res['MAE'][3])
            self.holt_rmse = np.append(self.holt_rmse, res['RMSE'][3])
            self.holt_mad = np.append(self.holt_mad, res['MAD'][3])
            self.holt_mape = np.append(self.holt_mape, res['MAPE'][3])
            self.holt_maxape = np.append(self.holt_maxape, res['MAXAPE'][3])
            self.holt_tape = np.append(self.holt_tape, res['TAPE'][3])
            self.holt_df = np.append(self.holt_df, res['DF'][3])
            self.holt_area = np.append(self.holt_area, res['AREA'][3])
            self.holt_dtw = np.append(self.holt_dtw, res['DTW'][3])
            self.es_mae = np.append(self.es_mae, res['MAE'][4])
            self.es_rmse = np.append(self.es_rmse, res['RMSE'][4])
            self.es_mad = np.append(self.es_mad, res['MAD'][4])
            self.es_mape = np.append(self.es_mape, res['MAPE'][4])
            self.es_maxape = np.append(self.es_maxape, res['MAXAPE'][4])
            self.es_tape = np.append(self.es_tape, res['TAPE'][4])
            self.es_df = np.append(self.es_df, res['DF'][4])
            self.es_area = np.append(self.es_area, res['AREA'][4])
            self.es_dtw = np.append(self.es_dtw, res['DTW'][4])
            self.arima_mae = np.append(self.arima_mae, res['MAE'][5])
            self.arima_rmse = np.append(self.arima_rmse, res['RMSE'][5])
            self.arima_mad = np.append(self.arima_mad, res['MAD'][5])
            self.arima_mape = np.append(self.arima_mape, res['MAPE'][5])
            self.arima_maxape = np.append(self.arima_maxape, res['MAXAPE'][5])
            self.arima_tape = np.append(self.arima_tape, res['TAPE'][5])
            self.arima_df = np.append(self.arima_df, res['DF'][5])
            self.arima_area = np.append(self.arima_area, res['AREA'][5])
            self.arima_dtw = np.append(self.arima_dtw, res['DTW'][5])
            self.sarima_mae = np.append(self.sarima_mae, res['MAE'][6])
            self.sarima_rmse = np.append(self.sarima_rmse, res['RMSE'][6])
            self.sarima_mad = np.append(self.sarima_mad, res['MAD'][6])
            self.sarima_mape = np.append(self.sarima_mape, res['MAPE'][6])
            self.sarima_maxape = np.append(self.sarima_maxape, res['MAXAPE'][6])
            self.sarima_tape = np.append(self.sarima_tape, res['TAPE'][6])
            self.sarima_df = np.append(self.sarima_df, res['DF'][6])
            self.sarima_area = np.append(self.sarima_area, res['AREA'][6])
            self.sarima_dtw = np.append(self.sarima_dtw, res['DTW'][6])
            self.new_ocp = np.append(self.new_ocp, par[par['param'] == 'new_ocp'].value.astype(float))
            self.init_lengthscale = np.append(self.init_lengthscale, par[par['param'] == 'init_lengthscale'].value.astype(float))
            self.bic = np.append(self.bic, par[par['param'] == 'bic'].value.astype(float))
        
    def skmobMetrics(self):  
        
        os.chdir(self.skmob_files_path)

        self.mtgp_rec = np.array([])
        self.mtgp_freq = np.array([])
        self.mtgp_no_loc = np.array([])
        self.mtgp_k_rg = np.array([])
        self.mtgp_spat_burst = np.array([])
        self.mtgp_dist_straight = np.array([])
        self.mtgp_rand_entr = np.array([])
        self.mtgp_real_entr = np.array([])
        self.mtgp_uncorr_entr = np.array([])
        self.mtgp_rbf_rec = np.array([])
        self.mtgp_rbf_freq = np.array([])
        self.mtgp_rbf_no_loc = np.array([])
        self.mtgp_rbf_k_rg = np.array([])
        self.mtgp_rbf_spat_burst = np.array([])
        self.mtgp_rbf_dist_straight = np.array([])
        self.mtgp_rbf_rand_entr = np.array([])
        self.mtgp_rbf_real_entr = np.array([])
        self.mtgp_rbf_uncorr_entr = np.array([])
        self.ses_rec = np.array([])
        self.ses_freq = np.array([])
        self.ses_no_loc = np.array([])
        self.ses_k_rg = np.array([])
        self.ses_spat_burst = np.array([])
        self.ses_dist_straight = np.array([])
        self.ses_rand_entr = np.array([])
        self.ses_real_entr = np.array([])
        self.ses_uncorr_entr = np.array([])
        self.holt_rec = np.array([])
        self.holt_freq = np.array([])
        self.holt_no_loc = np.array([])
        self.holt_k_rg = np.array([])
        self.holt_spat_burst = np.array([])
        self.holt_dist_straight = np.array([])
        self.holt_rand_entr = np.array([])
        self.holt_real_entr = np.array([])
        self.holt_uncorr_entr = np.array([])
        self.es_rec = np.array([])
        self.es_freq = np.array([])
        self.es_no_loc = np.array([])
        self.es_k_rg = np.array([])
        self.es_spat_burst = np.array([])
        self.es_dist_straight = np.array([])
        self.es_rand_entr = np.array([])
        self.es_real_entr = np.array([])
        self.es_uncorr_entr = np.array([])
        self.arima_rec = np.array([])
        self.arima_freq = np.array([])
        self.arima_no_loc = np.array([])
        self.arima_k_rg = np.array([])
        self.arima_spat_burst = np.array([])
        self.arima_dist_straight = np.array([])
        self.arima_rand_entr = np.array([])
        self.arima_real_entr = np.array([])
        self.arima_uncorr_entr = np.array([])
        self.sarima_rec = np.array([])
        self.sarima_freq = np.array([])
        self.sarima_no_loc = np.array([])
        self.sarima_k_rg = np.array([])
        self.sarima_spat_burst = np.array([])
        self.sarima_dist_straight = np.array([])
        self.sarima_rand_entr = np.array([])
        self.sarima_real_entr = np.array([])
        self.sarima_uncorr_entr = np.array([])
        self.li_rec = np.array([])
        self.li_freq = np.array([])
        self.li_no_loc = np.array([])
        self.li_k_rg = np.array([])
        self.li_spat_burst = np.array([])
        self.li_dist_straight = np.array([])
        self.li_rand_entr = np.array([])
        self.li_real_entr = np.array([])
        self.li_uncorr_entr = np.array([])


        for file in glob.glob("*.csv"):
            res = pd.read_csv(file, header=0)
            # Store each metric in a numpy array
            self.mtgp_rec = np.append(self.mtgp_rec, res['recency'][0])
            self.mtgp_freq = np.append(self.mtgp_freq, res['freq_rank'][0])
            self.mtgp_no_loc = np.append(self.mtgp_no_loc, res['no_loc_error'][0])
            self.mtgp_k_rg = np.append(self.mtgp_k_rg, res['k_rg_error'][0])
            self.mtgp_spat_burst = np.append(self.mtgp_spat_burst, res['spat_burst_error'][0])
            self.mtgp_dist_straight = np.append(self.mtgp_dist_straight, res['dist_straight_error'][0])
            self.mtgp_rand_entr = np.append(self.mtgp_rand_entr, res['rand_entr_error'][0])
            self.mtgp_real_entr = np.append(self.mtgp_real_entr, res['real_entr_error'][0])
            self.mtgp_uncorr_entr = np.append(self.mtgp_uncorr_entr, res['uncorr_entr_error'][0])
            self.mtgp_rbf_rec = np.append(self.mtgp_rbf_rec, res['recency'][1])
            self.mtgp_rbf_freq = np.append(self.mtgp_rbf_freq, res['freq_rank'][1])
            self.mtgp_rbf_no_loc = np.append(self.mtgp_rbf_no_loc, res['no_loc_error'][1])
            self.mtgp_rbf_k_rg = np.append(self.mtgp_rbf_k_rg, res['k_rg_error'][1])
            self.mtgp_rbf_spat_burst = np.append(self.mtgp_rbf_spat_burst, res['spat_burst_error'][1])
            self.mtgp_rbf_dist_straight = np.append(self.mtgp_rbf_dist_straight, res['dist_straight_error'][1])
            self.mtgp_rbf_rand_entr = np.append(self.mtgp_rbf_rand_entr, res['rand_entr_error'][1])
            self.mtgp_rbf_real_entr = np.append(self.mtgp_rbf_real_entr, res['real_entr_error'][1])
            self.mtgp_rbf_uncorr_entr = np.append(self.mtgp_rbf_uncorr_entr, res['uncorr_entr_error'][1])
            self.ses_rec = np.append(self.ses_rec, res['recency'][2])
            self.ses_freq = np.append(self.ses_freq, res['freq_rank'][2])
            self.ses_no_loc = np.append(self.ses_no_loc, res['no_loc_error'][2])
            self.ses_k_rg = np.append(self.ses_k_rg, res['k_rg_error'][2])
            self.ses_spat_burst = np.append(self.ses_spat_burst, res['spat_burst_error'][2])
            self.ses_dist_straight = np.append(self.ses_dist_straight, res['dist_straight_error'][2])
            self.ses_rand_entr = np.append(self.ses_rand_entr, res['rand_entr_error'][2])
            self.ses_real_entr = np.append(self.ses_real_entr, res['real_entr_error'][2])
            self.ses_uncorr_entr = np.append(self.ses_uncorr_entr, res['uncorr_entr_error'][2])
            self.holt_rec = np.append(self.holt_rec, res['recency'][3])
            self.holt_freq = np.append(self.holt_freq, res['freq_rank'][3])
            self.holt_no_loc = np.append(self.holt_no_loc, res['no_loc_error'][3])
            self.holt_k_rg = np.append(self.holt_k_rg, res['k_rg_error'][3])
            self.holt_spat_burst = np.append(self.holt_spat_burst, res['spat_burst_error'][3])
            self.holt_dist_straight = np.append(self.holt_dist_straight, res['dist_straight_error'][3])
            self.holt_rand_entr = np.append(self.holt_rand_entr, res['rand_entr_error'][3])
            self.holt_real_entr = np.append(self.holt_real_entr, res['real_entr_error'][3])
            self.holt_uncorr_entr = np.append(self.holt_uncorr_entr, res['uncorr_entr_error'][3])
            self.es_rec = np.append(self.es_rec, res['recency'][4])
            self.es_freq = np.append(self.es_freq, res['freq_rank'][4])
            self.es_no_loc = np.append(self.es_no_loc, res['no_loc_error'][4])
            self.es_k_rg = np.append(self.es_k_rg, res['k_rg_error'][4])
            self.es_spat_burst = np.append(self.es_spat_burst, res['spat_burst_error'][4])
            self.es_dist_straight = np.append(self.es_dist_straight, res['dist_straight_error'][4])
            self.es_rand_entr = np.append(self.es_rand_entr, res['rand_entr_error'][4])
            self.es_real_entr = np.append(self.es_real_entr, res['real_entr_error'][4])
            self.es_uncorr_entr = np.append(self.es_uncorr_entr, res['uncorr_entr_error'][4])
            self.arima_rec = np.append(self.arima_rec, res['recency'][5])
            self.arima_freq = np.append(self.arima_freq, res['freq_rank'][5])
            self.arima_no_loc = np.append(self.arima_no_loc, res['no_loc_error'][5])
            self.arima_k_rg = np.append(self.arima_k_rg, res['k_rg_error'][5])
            self.arima_spat_burst = np.append(self.arima_spat_burst, res['spat_burst_error'][5])
            self.arima_dist_straight = np.append(self.arima_dist_straight, res['dist_straight_error'][5])
            self.arima_rand_entr = np.append(self.arima_rand_entr, res['rand_entr_error'][5])
            self.arima_real_entr = np.append(self.arima_real_entr, res['real_entr_error'][5])
            self.arima_uncorr_entr = np.append(self.arima_uncorr_entr, res['uncorr_entr_error'][5])
            self.sarima_rec = np.append(self.sarima_rec, res['recency'][6])
            self.sarima_freq = np.append(self.sarima_freq, res['freq_rank'][6])
            self.sarima_no_loc = np.append(self.sarima_no_loc, res['no_loc_error'][6])
            self.sarima_k_rg = np.append(self.sarima_k_rg, res['k_rg_error'][6])
            self.sarima_spat_burst = np.append(self.sarima_spat_burst, res['spat_burst_error'][6])
            self.sarima_dist_straight = np.append(self.sarima_dist_straight, res['dist_straight_error'][6])
            self.sarima_rand_entr = np.append(self.sarima_rand_entr, res['rand_entr_error'][6])
            self.sarima_real_entr = np.append(self.sarima_real_entr, res['real_entr_error'][6])
            self.sarima_uncorr_entr = np.append(self.sarima_uncorr_entr, res['uncorr_entr_error'][6])
            self.li_rec = np.append(self.li_rec, res['recency'][7])
            self.li_freq = np.append(self.li_freq, res['freq_rank'][7])
            self.li_no_loc = np.append(self.li_no_loc, res['no_loc_error'][7])
            self.li_k_rg = np.append(self.li_k_rg, res['k_rg_error'][7])
            self.li_spat_burst = np.append(self.li_spat_burst, res['spat_burst_error'][7])
            self.li_dist_straight = np.append(self.li_dist_straight, res['dist_straight_error'][7])
            self.li_rand_entr = np.append(self.li_rand_entr, res['rand_entr_error'][7])
            self.li_real_entr = np.append(self.li_real_entr, res['real_entr_error'][7])
            self.li_uncorr_entr = np.append(self.li_uncorr_entr, res['uncorr_entr_error'][7])


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