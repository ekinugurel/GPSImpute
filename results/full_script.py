import skmob
import pandas as pd
import skmob.measures.individual as ind_measure
import torch
import gpytorch
from gpytorch.kernels import RQKernel as RQ, RBFKernel as SE, \
PeriodicKernel as PER, ScaleKernel, LinearKernel as LIN, MaternKernel as MAT, \
SpectralMixtureKernel as SMK, PiecewisePolynomialKernel as PPK, CylindricalKernel as CYL
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from gpytorch.constraints import Interval
import time
import json
import os
import shutil
import statistics as stats

# Import intra-package scripts
import utils.helper_func as helper_func
import utils.GP as GP
from utils.helper_func import dec_floor
import mobileDataToolkit.analysis as analysis
import mobileDataToolkit.preprocessing_v2 as preprocessing
import mobileDataToolkit.methods as methods
import mobileDataToolkit.metrics as metrics

# Import benchmarks
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings('ignore')

def full_script():
    '''
    This script will run the full pipeline for the GPR project.
    '''
    file_path = "C:\\Users\\ekino\\OneDrive - UW\\GPR\\Data\\seattle_2000_all_obs_sampled.csv"
    df = pd.read_csv(file_path, header=0)

    # Add month column
    df['month'] = pd.DatetimeIndex(df['datetime']).month

    # Group by user ID, find month with third most observations (average)
    df_m = df.groupby('UID').apply(lambda x: x[x['month'] == x['month'].value_counts().index[2]])

    df_m = df_m.reset_index(drop=True)

    max_speed_kmh = 400 # for filtering out unrealistic speeds
    spatial_radius_km = 0.3 # for compressing similar points using Douglas-Peucker algorithm
    bin_len_ls = [10080, 1440, 360, 60, 30, 15] # Bin lengths to test: 1 week, 1 day, 6 hours, 1 hour, 30 min, 15 min
    init_period_len_1 = 60*8 # 8 hours
    init_period_len_2 = 60*24 # 24 hours

    runtimes = []


    for j in bin_len_ls:
        bin_len = j
        # Create a directory for each bin length
        if not os.path.exists('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len)):
            os.makedirs('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len))
        print("Starting tests on bin length = {}".format(bin_len))
        # Main loop that will go through each user ID, create a directory for each user, etc.
        for i in df_m.UID.unique():
            try:
                
                df_curr = df_m[df_m.UID == i]

                tdf = skmob.TrajDataFrame(df_curr, latitude='orig_lat', longitude='orig_long', datetime='datetime')
                f_tdf = skmob.preprocessing.filtering.filter(tdf, max_speed_kmh=max_speed_kmh, include_loops=False)
                # Print the difference in number of rows
                print("Number of rows before filtering: {}".format(tdf.shape[0]))
                print("Number of rows after filtering: {}".format(f_tdf.shape[0]))
                fc_tdf = skmob.preprocessing.compression.compress(f_tdf, spatial_radius_km=spatial_radius_km)
                # Print the difference in number of rows
                print("Number of rows after compression: {}".format(fc_tdf.shape[0]))
                # Remove data points with uncertainty > 100m
                fcu_tdf = fc_tdf[fc_tdf['orig_unc'] <= 100]
                # Print the difference in number of rows
                print("Number of rows after uncertainty filtering: {}".format(fcu_tdf.shape[0]))
                df_curr = fcu_tdf

                # Calculate sci-kit mobility metrics
                df_curr_metrics = helper_func.skmob_metric_calcs(df_curr, method='GT', lat='lat', long='lng', datetime='datetime')

                # Remove duplicates in the unix column
                df_curr = df_curr.drop_duplicates(subset=['unix_min'])

                curr_ocp = analysis.tempOcp(df_curr, 'unix_min', bin_len=bin_len)

                upper_bound = dec_floor(curr_ocp)
                
                # See current temporal occupancy
                print("Current temporal occupancy: {}".format(curr_ocp))
                while True:
                    try:
                        if curr_ocp <= 0.1:
                            target_ocp = np.random.uniform(0, curr_ocp)
                        else:
                            # Choose random decimal between 0 and upper bound
                            target_ocp = dec_floor(np.random.uniform(0.1, upper_bound))
                        print("Target temporal occupancy: {}".format(target_ocp))
                        # Simulate gaps in the user's data to match the target level
                        gapped_user_data, train_index, new_ocp = analysis.simulate_gaps(df_curr, target_ocp, unix_col='unix_min', bin_len= bin_len)
                    except:
                        continue
                    break

                # Change name of 'lat' and 'lon' columns to 'orig_lat' and 'orig_long'
                df_curr = df_curr.rename(columns={'lat': 'orig_lat', 'lng': 'orig_long'})

                # Create MultiTrip object
                curr_mt = preprocessing.dp_MultiTrip(data=df_curr)
                curr_mt.Multi_Trip_Preprocess(lat='orig_lat', long='orig_long', datetime='datetime')

                # Move 'unix_start_t' to before 'SaM'
                cols = list(curr_mt.data.columns)
                cols.insert(16, cols.pop(cols.index('unix_min')))
                curr_mt.data = curr_mt.data.loc[:, cols] 
                # Print data columns
                print(curr_mt.data.columns)

                curr_mt.Multi_Trip_TrainTestSplit(test_start_date=None, test_end_date=None, 
                                            training_index = set(gapped_user_data['unix_min']), lat='orig_lat', 
                                            long='orig_long', datetime='datetime', unix='unix_min', inputstart='unix_min', 
                                            inputend=curr_mt.data.columns[-1])

                n_train = len(curr_mt.X_train[:,0])
                n_test = len(curr_mt.X_test[:,0])
                n_dims = curr_mt.X_train.shape[1]

                # See number of points in training and test sets
                print("Number of points in training set: {}".format(n_train))
                print("Number of points in test set: {}".format(n_test))
                print("Number of input dimensions: {}".format(n_dims))

                # If there are no points in the test set, skip to the next user
                if n_test == 0:
                    print("No points in test set. Skipping to next user.")
                    continue

                # Visualize the training and test data in two subplots, one lat vs time and one long vs time
                plt.rcParams.update({'font.size': 9})
                plt.rcParams.update({'font.family': 'serif'})
                fig1, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                ax[0].scatter(curr_mt.data[curr_mt.data['unix_min'].isin(set(gapped_user_data['unix_min']))]['unix_min'],
                                curr_mt.data[curr_mt.data['unix_min'].isin(set(gapped_user_data['unix_min']))]['orig_lat'],
                                color='blue', label='Training data', s=1)
                ax[0].scatter(curr_mt.data[~curr_mt.data['unix_min'].isin(set(gapped_user_data['unix_min']))]['unix_min'],
                                curr_mt.data[~curr_mt.data['unix_min'].isin(set(gapped_user_data['unix_min']))]['orig_lat'],
                                color='red', label='Test data', s=1)
                ax[0].set_ylabel('Latitude')
                ax[1].scatter(curr_mt.data[curr_mt.data['unix_min'].isin(set(gapped_user_data['unix_min']))]['unix_min'],
                                curr_mt.data[curr_mt.data['unix_min'].isin(set(gapped_user_data['unix_min']))]['orig_long'],
                                color='blue', label='Training data', s=1)
                ax[1].scatter(curr_mt.data[~curr_mt.data['unix_min'].isin(set(gapped_user_data['unix_min']))]['unix_min'],
                                curr_mt.data[~curr_mt.data['unix_min'].isin(set(gapped_user_data['unix_min']))]['orig_long'],
                                color='red', label='Test data', s=1)
                ax[1].set_xlabel('Time')
                ax[1].set_ylabel('Longitude')
                ax[1].legend()


                mean_lat = curr_mt.y_train[:,0].mean()
                mean_long = curr_mt.y_train[:,1].mean()
                std_lat = curr_mt.y_train[:,0].std()
                std_long = curr_mt.y_train[:,1].std()

                scaler = StandardScaler()
                y_train_scaled = torch.tensor(np.float64(scaler.fit_transform(curr_mt.y_train)))
                y_test_scaled = torch.tensor(np.float64(scaler.transform(curr_mt.y_test)))

                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

                model = GP.MTGPRegressor(curr_mt.X_train, y_train_scaled, 
                                        ScaleKernel(RQ(ard_num_dims=n_dims) * PER(active_dims=[0])) + 
                                        ScaleKernel(RQ(ard_num_dims=n_dims) * PER(active_dims=[0])))

                # Set initial lengthscale guess as half the average length of gap in training set
                init_lengthscale = curr_mt.data[curr_mt.data['unix_min'].isin(set(gapped_user_data['unix_min']))]['unix_min'].diff().mean() / 2 
                initializations = np.ones(n_dims - 1)
                initializations = np.insert(initializations, 0, init_lengthscale)
                model.covar_module.data_covar_module.kernels[0].base_kernel.kernels[0].lengthscale = initializations
                model.covar_module.data_covar_module.kernels[1].base_kernel.kernels[0].lengthscale = initializations

                # Set initial period lengths
                model.covar_module.data_covar_module.kernels[0].base_kernel.kernels[1].period_length = init_period_len_1
                model.covar_module.data_covar_module.kernels[1].base_kernel.kernels[1].period_length = init_period_len_2

                # Train model
                start = time.time()
                ls, mll = GP.training(model, curr_mt.X_train, y_train_scaled, lr=0.3, n_epochs=150)
                end = time.time()
                runtime = end - start
                runtimes.append(runtime)

                iters = range(0, len(ls))
                fig2, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax.plot(iters, ls, 'g')
                ax.set_title('Training Loss')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                ax.legend()
                
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

                with torch.no_grad():
                    log_ll = mll(model(curr_mt.X_train), y_train_scaled) * curr_mt.X_train.shape[0]
                            
                N = curr_mt.X_train.shape[0]
                m = sum(p.numel() for p in model.hyperparameters())
                bic = -2 * log_ll + m * np.log(N)

                predictions, mean = model.predict(curr_mt.X_test)

                # Use smaller font
                plt.rcParams.update({'font.size': 9})
                # Make the font nicer
                plt.rcParams.update({'font.family': 'serif'})
                fig3, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax.set_title('Predictions')
                pd.DataFrame(mean.detach().numpy()).plot(x=1, y=0, kind='scatter',ax=ax, color='red', alpha=0.5, s=0.4, label='Predictions')
                pd.DataFrame(y_test_scaled.detach().numpy()).plot(x=1, y=0, kind='scatter',ax=ax, color='blue', alpha=0.5, s=0.4, label='Actual')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')

                # Unix time for benchmarks
                unix_min_tr = np.array(curr_mt.X_train[:,0]).astype(int)
                unix_min_te = np.array(curr_mt.X_test[:,0]).astype(int)

                # Model results
                mtgp_res = metrics.average_eval(pd.Series(y_test_scaled[:,0]), pd.Series(y_test_scaled[:,1]), pd.Series(mean[:,0]), pd.Series(mean[:,1]))

                # Convert mean predictions back to original scale in lat/long
                orig_preds = scaler.inverse_transform(mean.reshape(-1,2))

                GP_full_preds_df = helper_func.preds_to_full_df(preds_lat=orig_preds[:,0], preds_long=orig_preds[:,1], 
                                                            test_df = curr_mt.test, train_df=curr_mt.train)

                mtgp_tdf = helper_func.skmob_metric_calcs(GP_full_preds_df, method='GP', lat='lat', long='long', datetime='datetime')

                mtgp_rec_acc = helper_func.matrix_acc(mtgp_tdf.recency_gp_pred, df_curr_metrics.recency_gt_pred, metric_name='recency', tolerance=1e-04)
                mtgp_freq_rank_acc = helper_func.matrix_acc(mtgp_tdf.freq_rank_gp_pred, df_curr_metrics.freq_rank_gt_pred, metric_name='freq_rank', tolerance=1e-01)

                try:
                    # Linear Interpolation
                    print("Running Linear Interpolation...")
                    LI_preds_lat, LI_preds_long = methods.LI(curr_mt.X_train[:,0], curr_mt.X_test[:,0], y_train_scaled, y_test_scaled)

                    LI_preds_df = pd.DataFrame(LI_preds_lat, columns=['lat'])
                    LI_preds_df['long'] = LI_preds_long

                    LI_preds_origs = scaler.inverse_transform(LI_preds_df)
                    
                    LI_full_preds_df = helper_func.preds_to_full_df(preds_lat=LI_preds_origs[:,0], preds_long=LI_preds_origs[:,1], 
                                                                test_df = curr_mt.test, train_df=curr_mt.train)
                    
                    LI_tdf = helper_func.skmob_metric_calcs(LI_full_preds_df, method='LI', lat='lat', long='long', datetime='datetime')
                    LI_res = metrics.average_eval(np.array(y_test_scaled[:,0]), np.array(y_test_scaled[:,1]), LI_preds_lat, LI_preds_long)

                    LI_rec_acc = helper_func.matrix_acc(LI_tdf.recency_li_pred, df_curr_metrics.recency_gt_pred, metric_name='recency', tolerance=1e-04)
                    LI_freq_rank_acc = helper_func.matrix_acc(LI_tdf.freq_rank_li_pred, df_curr_metrics.freq_rank_gt_pred, metric_name='freq_rank', tolerance=1e-01)
                except:
                    print("Error in LI")
                    LI_res = None
                    LI_rec_acc = None
                    LI_freq_rank_acc = None
                
                lat = pd.Series(y_train_scaled[:,0].tolist(), unix_min_tr)
                lat_t = pd.Series(y_test_scaled[:,0].tolist(), unix_min_te)
                # Replace duplicates (in time) with the mean of the two values
                lat = lat.groupby(lat.index).mean().reset_index()
                lat = pd.Series(lat[0].tolist(), lat['index'].tolist())
                lat_tc = lat_t.groupby(lat_t.index).mean().reset_index()
                lat_tc = pd.Series(lat_tc[0].tolist(), lat_tc['index'].tolist())
                # Replace zeroes with positives close to zero
                lat.replace(0, 0.000000001, inplace=True)

                lon = pd.Series(y_train_scaled[:,1].tolist(), unix_min_tr)
                lon_t = pd.Series(y_test_scaled[:,1].tolist(),unix_min_te)
                # Replace duplicates (in time) with the mean of the two values
                lon = lon.groupby(lon.index).mean().reset_index()
                lon = pd.Series(lon[0].tolist(), lon['index'].tolist())
                lon_tc = lon_t.groupby(lon_t.index).mean().reset_index()
                lon_tc = pd.Series(lon_tc[0].tolist(), lon_tc['index'].tolist())
                # Replace zeroes with positives close to zero
                lon.replace(0, 0.000000001, inplace=True)

                # SES model
                ses_smoothing_level = 0.1
                print("Running Simple Exponential Smoothing...")
                ses_lat = SimpleExpSmoothing(lat, initialization_method="heuristic").fit(smoothing_level=ses_smoothing_level, optimized=True)
                pred_lat_ses = ses_lat.predict(start=lat_tc.index[0], end=lat_tc.index[-1])
                pred_lat_comp_ses = pred_lat_ses[pred_lat_ses.index.isin(unix_min_te)]

                ses_lon = SimpleExpSmoothing(lon, initialization_method="heuristic").fit(smoothing_level=ses_smoothing_level, optimized=True)
                pred_lon_ses = ses_lon.predict(start=lon_tc.index[0], end=lon_tc.index[-1])
                pred_lon_comp_ses = pred_lon_ses[pred_lon_ses.index.isin(unix_min_te)]

                ses_preds_df = pd.DataFrame(pred_lat_comp_ses, columns=['lat'])
                ses_preds_df['long'] = pred_lon_comp_ses

                ses_preds_origs = scaler.inverse_transform(ses_preds_df)

                ses_full_preds_df = helper_func.preds_to_full_df(preds_lat=ses_preds_origs[:,0], preds_long=ses_preds_origs[:,1],
                                                            test_df = curr_mt.test, train_df=curr_mt.train)
                ses_tdf = helper_func.skmob_metric_calcs(ses_full_preds_df, method='ses', lat='lat', long='long', datetime='datetime')
                ses_res = metrics.average_eval(lat_tc, lon_tc, pred_lat_comp_ses, pred_lon_comp_ses)

                ses_rec_acc = helper_func.matrix_acc(ses_tdf.recency_ses_pred, df_curr_metrics.recency_gt_pred, metric_name='recency', tolerance=1e-04)
                ses_freq_rank_acc = helper_func.matrix_acc(ses_tdf.freq_rank_ses_pred, df_curr_metrics.freq_rank_gt_pred, metric_name='freq_rank', tolerance=1e-01)

                # Holt model
                holt_smoothing_level_lat=0.2
                holt_smoothing_slope_lat=0.045

                print("Running Holt-Winters model...")
                holt_lat = Holt(lat, damped_trend=True, initialization_method="estimated").fit(smoothing_level=holt_smoothing_level_lat, smoothing_slope=holt_smoothing_slope_lat)
                pred_lat_holt = holt_lat.predict(start=lat_tc.index[0], end=lat_tc.index[-1])
                pred_lat_comp_holt = pred_lat_holt[pred_lat_holt.index.isin(unix_min_te)]

                holt_smoothing_level_lon=0.1
                holt_smoothing_slope_lon=0.0307

                holt_lon = Holt(lon, damped_trend=True, initialization_method="estimated").fit(smoothing_level=holt_smoothing_level_lon, smoothing_slope=holt_smoothing_slope_lon)
                pred_lon_holt = holt_lon.predict(start=lat_tc.index[0], end=lat_tc.index[-1])
                pred_lon_comp_holt = pred_lon_holt[pred_lon_holt.index.isin(unix_min_te)]

                holt_preds_df = pd.DataFrame(pred_lat_comp_holt, columns=['lat'])
                holt_preds_df['long'] = pred_lon_comp_holt

                holt_preds_origs = scaler.inverse_transform(holt_preds_df)

                holt_full_preds_df = helper_func.preds_to_full_df(preds_lat=holt_preds_origs[:,0], preds_long=holt_preds_origs[:,1],
                                                            test_df = curr_mt.test, train_df=curr_mt.train)
                holt_tdf = helper_func.skmob_metric_calcs(holt_full_preds_df, method='holt', lat='lat', long='long', datetime='datetime')

                holt_res = metrics.average_eval(lat_tc, lon_tc, pred_lat_comp_holt, pred_lon_comp_holt)

                holt_rec_acc = helper_func.matrix_acc(holt_tdf.recency_holt_pred, df_curr_metrics.recency_gt_pred, metric_name='recency', tolerance=1e-04)
                holt_freq_rank_acc = helper_func.matrix_acc(holt_tdf.freq_rank_holt_pred, df_curr_metrics.freq_rank_gt_pred, metric_name='freq_rank', tolerance=1e-01)

                try:
                    # Exponential Smoothing
                    es_seasonal_periods=24

                    print("Running Exponential Smoothing...")
                    es = ExponentialSmoothing(lat, seasonal_periods=es_seasonal_periods, trend='add', seasonal='add', damped_trend=True, use_boxcox=False, initialization_method='estimated').fit()
                    pred_lat_es = es.predict(start=lat_tc.index[0], end=lat_tc.index[-1])
                    pred_lat_comp_es = pred_lat_es[pred_lat_es.index.isin(unix_min_te)]

                    es = ExponentialSmoothing(lon, seasonal_periods=es_seasonal_periods, trend='add', seasonal='add', damped_trend=True, use_boxcox=False, initialization_method='estimated').fit()
                    pred_lon_es = es.predict(start=lon_tc.index[0], end=lon_tc.index[-1])
                    pred_lon_comp_es = pred_lon_es[pred_lon_es.index.isin(unix_min_te)]

                    es_preds_df = pd.DataFrame(pred_lat_comp_es, columns=['lat'])
                    es_preds_df['long'] = pred_lon_comp_es

                    es_preds_origs = scaler.inverse_transform(es_preds_df)

                    es_full_preds_df = helper_func.preds_to_full_df(preds_lat=es_preds_origs[:,0], preds_long=es_preds_origs[:,1],
                                                                test_df = curr_mt.test, train_df=curr_mt.train)
                    es_tdf = helper_func.skmob_metric_calcs(es_full_preds_df, method='es', lat='lat', long='long', datetime='datetime')
                    es_res = metrics.average_eval(lat_tc, lon_tc, pred_lat_comp_es, pred_lon_comp_es)

                    es_rec_acc = helper_func.matrix_acc(es_tdf.recency_es_pred, df_curr_metrics.recency_gt_pred, metric_name='recency', tolerance=1e-04)
                    es_freq_rank_acc = helper_func.matrix_acc(es_tdf.freq_rank_es_pred, df_curr_metrics.freq_rank_gt_pred, metric_name='freq_rank', tolerance=1e-01)
                except:
                    print("Exponential Smoothing failed")
                    es_res = None
                    es_rec_acc = None
                    es_freq_rank_acc = None
                
                try:
                    # ARIMA
                    arima_order = (1,1,0)
                    print("Running ARIMA...")
                    arima = ARIMA(lat, order=arima_order).fit()
                    pred_lat_arima = arima.predict(start=lat_tc.index[0], end=lat_tc.index[-1])
                    pred_lat_comp_arima = pred_lat_arima[pred_lat_arima.index.isin(unix_min_te)]

                    arima = ARIMA(lon, order=arima_order).fit()
                    pred_lon_arima = arima.predict(start=lon_tc.index[0], end=lon_tc.index[-1])
                    pred_lon_comp_arima = pred_lon_arima[pred_lon_arima.index.isin(unix_min_te)]

                    arima_preds_df = pd.DataFrame(pred_lat_comp_arima, columns=['lat'])
                    arima_preds_df['long'] = pred_lon_comp_arima

                    arima_preds_origs = scaler.inverse_transform(arima_preds_df)
                    
                    arima_full_preds_df = helper_func.preds_to_full_df(preds_lat=arima_preds_origs[:,0], preds_long=arima_preds_origs[:,1],
                                                                test_df = curr_mt.test, train_df=curr_mt.train)
                    arima_tdf = helper_func.skmob_metric_calcs(arima_full_preds_df, method='arima', lat='lat', long='long', datetime='datetime')
                    arima_res = metrics.average_eval(lat_tc, lon_tc, pred_lat_comp_arima, pred_lon_comp_arima)

                    arima_rec_acc = helper_func.matrix_acc(arima_tdf.recency_arima_pred, df_curr_metrics.recency_gt_pred, metric_name='recency', tolerance=1e-04)
                    arima_freq_rank_acc = helper_func.matrix_acc(arima_tdf.freq_rank_arima_pred, df_curr_metrics.freq_rank_gt_pred, metric_name='freq_rank', tolerance=1e-01)
                except:
                    print("ARIMA failed")
                    arima_res = None
                    arima_rec_acc = None
                    arima_freq_rank_acc = None

                try:
                    # SARIMAX
                    sarimax_order = (1,0,0)
                    print("Running SARIMAX...")
                    sarimax_seasonal_order = (1, 1, 1, 24)
                    sarimax_lat = SARIMAX(lat, order=sarimax_order, seasonal_order=sarimax_seasonal_order).fit(disp=False)
                    pred_lat_sar = sarimax_lat.predict(start=lat_tc.index[0], end=lat_tc.index[-1])
                    pred_lat_comp_sar = pred_lat_sar[pred_lat_sar.index.isin(unix_min_te)]

                    sarimax_lon = SARIMAX(lon, order=sarimax_order, seasonal_order=sarimax_seasonal_order).fit(disp=False)
                    pred_lon_sar = sarimax_lon.predict(start=lon_tc.index[0], end=lon_tc.index[-1])
                    pred_lon_comp_sar = pred_lon_sar[pred_lon_sar.index.isin(unix_min_te)]

                    sarimax_preds_df = pd.DataFrame(pred_lat_comp_sar, columns=['lat'])
                    sarimax_preds_df['long'] = pred_lon_comp_sar

                    sarimax_preds_origs = scaler.inverse_transform(sarimax_preds_df)

                    sarimax_full_preds_df = helper_func.preds_to_full_df(preds_lat=sarimax_preds_origs[:,0], preds_long=sarimax_preds_origs[:,1],
                                                                test_df = curr_mt.test, train_df=curr_mt.train)
                    sarimax_tdf = helper_func.skmob_metric_calcs(sarimax_full_preds_df, method='sarimax', lat='lat', long='long', datetime='datetime')
                    sarimax_res = metrics.average_eval(lat_tc, lon_tc, pred_lat_comp_sar, pred_lon_comp_sar)

                    sarimax_rec_acc = helper_func.matrix_acc(sarimax_tdf.recency_sarimax_pred, df_curr_metrics.recency_gt_pred, metric_name='recency', tolerance=1e-04)
                    sarimax_freq_rank_acc = helper_func.matrix_acc(sarimax_tdf.freq_rank_sarimax_pred, df_curr_metrics.freq_rank_gt_pred, metric_name='freq_rank', tolerance=1e-01)
                except:
                    print("SARIMAX failed")
                    sarimax_res = None
                    sarimax_rec_acc = None
                    sarimax_freq_rank_acc = None

                # Create a directory for each user
                if not os.path.exists('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\' + str(i)):
                    os.makedirs('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\' + str(i))
                else:
                    # If directory already exists, then prediction has already been done for this user, so skip
                    print("User {} already exists".format(i))
                    continue
                # Navigate to the directory
                os.chdir('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\' + str(i))

                # Save figure to file
                fig1.savefig('train_test_sets_plot.png', dpi=300)
                fig2.savefig('training_loss_plot.png', dpi=300)
                fig3.savefig('predictions_plot.png', dpi=300)

                # Create dictionary to store parameters
                params = {
                    'max_speed_kmh': max_speed_kmh,
                    'spatial_radius_km': spatial_radius_km,
                    'bin_len': bin_len,
                    'tdf.shape[0]': tdf.shape[0],
                    'f_tdf.shape[0]': f_tdf.shape[0],
                    'fc_tdf.shape[0]': fc_tdf.shape[0],
                    'fcu_tdf.shape[0]': fcu_tdf.shape[0],
                    'curr_ocp': curr_ocp,
                    'target_ocp': target_ocp,
                    'new_ocp': new_ocp,
                    'n_train': n_train,
                    'n_test': n_test,
                    'n_dims': n_dims,
                    'mean_lat': mean_lat,
                    'mean_long': mean_long,
                    'std_lat': std_lat,
                    'std_long': std_long,
                    'init_lengthscale': init_lengthscale,
                    'init_period_len_1': init_period_len_1,
                    'init_period_len_2': init_period_len_2,
                    'log_ll': log_ll,
                    'm': m,
                    'bic': bic,
                    'gp_runtime': runtime,
                    'ses_smoothing_level': ses_smoothing_level,
                    'holt_smoothing_level_lat': holt_smoothing_level_lat,
                    'holt_smoothing_slope_lat': holt_smoothing_slope_lat,
                    'holt_smoothing_level_lon': holt_smoothing_level_lon,
                    'holt_smoothing_slope_lon': holt_smoothing_slope_lon,
                    'es_seasonal_periods': es_seasonal_periods,
                    'arima_order': arima_order,
                    'sarimax_order': sarimax_order,
                    'sarimax_seasonal_order': sarimax_seasonal_order
                }

                print("Saving parameters...")

                # See the differences in metric results
                
                # Convert all values to float, except for tuples
                for k, v in params.items():
                    try:
                        params[k] = float(v)
                    except TypeError:
                        pass
                # Write params to a file
                with open('params_' + str(i) + '.json', 'w') as fp:
                    json.dump(params, fp)

                # Create dataframe to store parameters
                params_df = pd.DataFrame.from_dict(params, orient='index')
                params_df.columns = ['value']
                params_df.to_csv('params_' + str(i) + '.csv')

                # Create dataframe to store results
                results = pd.DataFrame.from_dict([mtgp_res, ses_res, holt_res, es_res, arima_res, sarimax_res])
                results['model'] = ['MTGP', 'SES', 'Holt', 'ES', 'ARIMA', 'SARIMAX']
                results = results.set_index('model')

                # Create dataframe to store scalar scikit-mobility metrics
                skmob_metrics_df = pd.DataFrame(columns=['no_loc', 'rg', 'k_rg',    
                                                    'spat_burst', 'rand_entr', 
                                                    'real_entr', 'uncorr_entr',
                                                    'max_dist', 'dist_straight', 'max_dist_home', 
                                                    'recency', 'freq_rank'])

                skmob_metrics_df['methods'] = ['MTGP', 'SES', 'Holt', 'ES', 'ARIMA', 'SARIMAX','LI', 'Ground Truth']
                # Make methods the index
                skmob_metrics_df = skmob_metrics_df.set_index('methods')

                skmob_metrics_df.iloc[0] = mtgp_tdf.no_loc_gp_pred, mtgp_tdf.rg_gp_pred, mtgp_tdf.k_rg_gp_pred, mtgp_tdf.spat_burst_gp_pred, mtgp_tdf.rand_entr_gp_pred, mtgp_tdf.real_entr_gp_pred, mtgp_tdf.uncorr_entr_gp_pred, mtgp_tdf.max_dist_gp_pred, mtgp_tdf.dist_straight_gp_pred, mtgp_tdf.max_dist_home_gp_pred, mtgp_rec_acc, mtgp_freq_rank_acc
                skmob_metrics_df.iloc[1] = ses_tdf.no_loc_ses_pred, ses_tdf.rg_ses_pred, ses_tdf.k_rg_ses_pred, ses_tdf.spat_burst_ses_pred, ses_tdf.rand_entr_ses_pred, ses_tdf.real_entr_ses_pred, ses_tdf.uncorr_entr_ses_pred, ses_tdf.max_dist_ses_pred, ses_tdf.dist_straight_ses_pred, ses_tdf.max_dist_home_ses_pred, ses_rec_acc, ses_freq_rank_acc
                skmob_metrics_df.iloc[2] = holt_tdf.no_loc_holt_pred, holt_tdf.rg_holt_pred, holt_tdf.k_rg_holt_pred, holt_tdf.spat_burst_holt_pred, holt_tdf.rand_entr_holt_pred, holt_tdf.real_entr_holt_pred, holt_tdf.uncorr_entr_holt_pred, holt_tdf.max_dist_holt_pred, holt_tdf.dist_straight_holt_pred, holt_tdf.max_dist_home_holt_pred, holt_rec_acc, holt_freq_rank_acc
                skmob_metrics_df.iloc[3] = es_tdf.no_loc_es_pred, es_tdf.rg_es_pred, es_tdf.k_rg_es_pred, es_tdf.spat_burst_es_pred, es_tdf.rand_entr_es_pred, es_tdf.real_entr_es_pred, es_tdf.uncorr_entr_es_pred, es_tdf.max_dist_es_pred, es_tdf.dist_straight_es_pred, es_tdf.max_dist_home_es_pred, es_rec_acc, es_freq_rank_acc
                skmob_metrics_df.iloc[4] = arima_tdf.no_loc_arima_pred, arima_tdf.rg_arima_pred, arima_tdf.k_rg_arima_pred, arima_tdf.spat_burst_arima_pred, arima_tdf.rand_entr_arima_pred, arima_tdf.real_entr_arima_pred, arima_tdf.uncorr_entr_arima_pred, arima_tdf.max_dist_arima_pred, arima_tdf.dist_straight_arima_pred, arima_tdf.max_dist_home_arima_pred, arima_rec_acc, arima_freq_rank_acc
                skmob_metrics_df.iloc[5] = sarimax_tdf.no_loc_sarimax_pred, sarimax_tdf.rg_sarimax_pred, sarimax_tdf.k_rg_sarimax_pred, sarimax_tdf.spat_burst_sarimax_pred, sarimax_tdf.rand_entr_sarimax_pred, sarimax_tdf.real_entr_sarimax_pred, sarimax_tdf.uncorr_entr_sarimax_pred, sarimax_tdf.max_dist_sarimax_pred, sarimax_tdf.dist_straight_sarimax_pred, sarimax_tdf.max_dist_home_sarimax_pred, sarimax_rec_acc, sarimax_freq_rank_acc
                skmob_metrics_df.iloc[6] = LI_tdf.no_loc_li_pred, LI_tdf.rg_li_pred, LI_tdf.k_rg_li_pred, LI_tdf.spat_burst_li_pred, LI_tdf.rand_entr_li_pred, LI_tdf.real_entr_li_pred, LI_tdf.uncorr_entr_li_pred, LI_tdf.max_dist_li_pred, LI_tdf.dist_straight_li_pred, LI_tdf.max_dist_home_li_pred, LI_rec_acc, LI_freq_rank_acc
                skmob_metrics_df.iloc[7] = df_curr_metrics.no_loc_gt_pred, df_curr_metrics.rg_gt_pred, df_curr_metrics.k_rg_gt_pred, df_curr_metrics.spat_burst_gt_pred, df_curr_metrics.rand_entr_gt_pred, df_curr_metrics.real_entr_gt_pred, df_curr_metrics.uncorr_entr_gt_pred, df_curr_metrics.max_dist_gt_pred, df_curr_metrics.dist_straight_gt_pred, df_curr_metrics.max_dist_home_gt_pred, -1, -1,
                
                # Find absolute difference between predicted and ground truth
                skmob_metrics_df['no_loc_error'] = skmob_metrics_df['no_loc'] - skmob_metrics_df.iloc[7]['no_loc']
                skmob_metrics_df['rg_error'] = skmob_metrics_df['rg'] - skmob_metrics_df.iloc[7]['rg']
                skmob_metrics_df['k_rg_error'] = skmob_metrics_df['k_rg'] - skmob_metrics_df.iloc[7]['k_rg']
                skmob_metrics_df['spat_burst_error'] = skmob_metrics_df['spat_burst'] - skmob_metrics_df.iloc[7]['spat_burst']
                skmob_metrics_df['rand_entr_error'] = skmob_metrics_df['rand_entr'] - skmob_metrics_df.iloc[7]['rand_entr']
                skmob_metrics_df['real_entr_error'] = skmob_metrics_df['real_entr'] - skmob_metrics_df.iloc[7]['real_entr']
                skmob_metrics_df['uncorr_entr_error'] = skmob_metrics_df['uncorr_entr'] - skmob_metrics_df.iloc[7]['uncorr_entr']
                skmob_metrics_df['max_dist_error'] = skmob_metrics_df['max_dist'] - skmob_metrics_df.iloc[7]['max_dist']
                skmob_metrics_df['dist_straight_error'] = skmob_metrics_df['dist_straight'] - skmob_metrics_df.iloc[7]['dist_straight']
                skmob_metrics_df['max_dist_home_error'] = skmob_metrics_df['max_dist_home'] - skmob_metrics_df.iloc[7]['max_dist_home']

                # Find mean absolute error (MAE) and median absolute error for each method from the absolute differences in each metric
                skmob_metrics_df['mae'] = (1/10) * (abs(skmob_metrics_df['no_loc_error']) + abs(skmob_metrics_df['rg_error']) + abs(skmob_metrics_df['k_rg_error']) + abs(skmob_metrics_df['spat_burst_error']) + abs(skmob_metrics_df['rand_entr_error']) + abs(skmob_metrics_df['real_entr_error']) + abs(skmob_metrics_df['uncorr_entr_error']) + abs(skmob_metrics_df['max_dist_error']) + abs(skmob_metrics_df['dist_straight_error']) + abs(skmob_metrics_df['max_dist_home_error']) )
                skmob_metrics_df['mad'] = np.median(0 - np.median(np.array([abs(skmob_metrics_df['no_loc_error']), abs(skmob_metrics_df['rg_error']), abs(skmob_metrics_df['k_rg_error']), abs(skmob_metrics_df['spat_burst_error']), abs(skmob_metrics_df['rand_entr_error']), abs(skmob_metrics_df['real_entr_error']), abs(skmob_metrics_df['uncorr_entr_error']), abs(skmob_metrics_df['max_dist_error']), abs(skmob_metrics_df['dist_straight_error']), abs(skmob_metrics_df['max_dist_home_error']) ])))
                # Write skmob metrics to a file
                skmob_metrics_df.to_csv('skmob_metrics_' + str(i) + '.csv')

                # Write results to a file
                results.to_csv('results_' + str(i) + '.csv')

                # Create a directory for all results if it doesn't exist
                if not os.path.exists('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\all_results'):
                    os.makedirs('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\all_results')
                # Navigate to the directory
                os.chdir('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\all_results')

                # Write results there as well
                results.to_csv('results_' + str(i) + '.csv')

                # Create a directory for all skmob metrics if it doesn't exist
                if not os.path.exists('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\all_skmob_metrics'):
                    os.makedirs('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\all_skmob_metrics')
                # Navigate to the directory
                os.chdir('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\all_skmob_metrics')

                # Write skmob metrics there as well
                skmob_metrics_df.to_csv('skmob_metrics_' + str(i) + '.csv')

                # Create a directory for all parameters if it doesn't exist
                if not os.path.exists('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\all_parameters'):
                    os.makedirs('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\all_parameters')
                # Navigate to the directory
                os.chdir('C:\\Users\\ekino\\OneDrive - UW\\GPR\\Results\\' + str(bin_len) + '\\all_parameters')

                # Write parameters there as well
                params_df.to_csv('params_' + str(i) + '.csv')

            except:
                continue

if __name__ == '__main__':
    full_script()