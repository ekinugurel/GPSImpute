import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from math import radians, cos, sin, asin, sqrt, pi, atan2, degrees
import geopandas as gpd
import skmob


def init():
    plt.rcdefaults()
    
def plot_2d_func(func, n_rows=1, n_cols=1, title=None):
    grid_size = 100
    x_grid = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    x_grid = np.hstack((x_grid[0].reshape(-1, 1), x_grid[1].reshape(-1, 1)))
    y = func(x_grid)
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 6))
    ax = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    #ax.plot_surface(x_grid[:, 0].reshape(grid_size, grid_size), x_grid[:, 1].reshape(grid_size, grid_size),
    #                y.reshape(grid_size, grid_size),
    #                cmap=cm.jet, rstride=1, cstride=1)
    if title is not None:
        ax.set_title(title)
    return fig

def plotTempOcpvsError(error, ocp, fitline=True, title=None, xlabel='Temporal occupancy', ylabel='Error', figsize=(8,6), **kwargs):
    '''
    Plot temporal occupancy vs error
    Args:
        error: float numpy array of error values
        ocp: float numpy array of temporal occupancy values
        fitline: boolean, whether to fit a line to the data
        title: string, title of the plot
        xlabel: string, label of the x-axis
        ylabel: string, label of the y-axis
        figsize: tuple, size of the figure
        **kwargs: keyword arguments for plt.scatter
    Returns:
        None
    '''
    plt.figure(figsize=figsize)
    plt.scatter(ocp, error, **kwargs)
    if fitline:
        m, b = np.polyfit(ocp, error, 1)
        x = np.linspace(min(ocp), max(ocp), 100)
        plt.plot(x, m*x + b, label='MTGP')
    if title is not None:
        plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # X-tick font size
    plt.xticks(fontsize=14)
    # Y-tick font size
    plt.yticks(fontsize=14)
    plt.show()

def haversine(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees); in meters
    """
    # convert decimal degrees to radians 
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # haversine formula 
    dlong = long2 - long1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
    c = 2 * asin(sqrt(a)) 
    R = 6371  # radius of the earth in km
    m = R * c * 1000
    return m

def haversine_np(lon1, lat1, lon2, lat2):
    # Convert latitude and longitude to radians
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Calculate the difference between latitudes and longitudes
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Apply the haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r # Distance in kilometers

def geodesic(lat1, lon1, lat2, lon2):
    """
    geodesic distance; in meters

    Parameters
    ----------
    lat1 : float
        Latitude of the first point.
    lon1 : float
        Longitude of the first point.
    lat2 : float
        Latitude of the second point.
    lon2 : float
        Longitude of the second point.
    """
    
    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))
    R = 6371  # radius of the earth in km
    x = (lon2 - lon1) * cos( 0.5*(lat2+lat1) )
    y = lat2 - lat1
    d = R * sqrt( x*x + y*y ) * 1000
    return d

def newCoords(lat, lon, dy, dx):
    """
    Calculates a new lat/lon from an old lat/lon + displacement in x and y.

    Parameters
    ----------
    lat : float
        Latitude of the first point.
    lon : float
        Longitude of the first point.
    dy : float
        Displacement in y.
    dx : float
        Displacement in x.
    """
    r = 6371
    new_lat  = lat  + (dy*0.001 / r) * (180 / pi)
    new_lon = lon + (dx*0.001 / r) * (180 / pi) / cos(lat * pi/180)
    return new_lat, new_lon

def newCoordsAlt(lat1, lon1, d, brng, R = 6378.1):
    """
    Calculates a new lat/lon from an old lat/lon + distance and bearing

    Parameters
    ----------
    lat1 : float
        Latitude of the first point.
    lon1 : float
        Longitude of the first point.
    d : float
        Distance between the two points.
    brng : float
        Bearing between the two points.
    R : float
        Radius of the earth. Default is 6378.1 km.
    """
    d = d/1000 # convert distance to km
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    
    lat2 = asin( sin(lat1)*cos(d/R) +
         cos(lat1)*sin(d/R)*cos(brng))
    
    lon2 = lon1 + atan2(sin(brng)*sin(d/R)*cos(lat1),
                 cos(d/R)-sin(lat1)*sin(lat2))
    
    lat2 = degrees(lat2)
    lon2 = degrees(lon2)
    return lat2, lon2

def addDist(data, type=haversine_np, lat='orig_lat', lon='orig_long'):
    """
    Add distance column to a dataframe with latitudes and longitudes. 
    Type specifies whether to use the Haversine distance (default) or the geodesic distance.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe with latitudes and longitudes.
    type : function
        Function to calculate distance. Default is haversine_np.
    lat : string
        Name of the column with latitudes. Default is 'orig_lat'.
    lon : string
        Name of the column with longitudes. Default is 'orig_long'.
    """
    print("Adding distance column to dataframe...")
    if type == haversine_np:
        lat1, lon1 = data[lat], data[lon]
        lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
        dist = type(lat1, lon1, lat2, lon2).fillna(0)
        data['dist'] = dist
    elif type == geodesic:
        lat1, lon1 = data[lat], data[lon]
        lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
        dist = type(lat1, lon1, lat2, lon2).miles.fillna(0)
        data['dist'] = dist
        
def addVel(data, unix='unix_min', lat='orig_lat', lon='orig_long'):
    """
    Add velocity column to a dataframe with latitudes and longitudes.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe with latitudes and longitudes.
    unix : string
        Name of the column containing unix timestamps.
    lat : string
        Name of the column containing latitudes.
    lon : string
        Name of the column containing longitudes.
    """
    print("Adding velocity column to dataframe...")
    if 'dist' in data.columns:
        lat1, lon1 = data[lat], data[lon]
        lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
        dist = data['dist'].fillna(0)
        time_diff = (data[unix] - data[unix].shift(1)).fillna(0)
        vel = dist / time_diff
        vel.iloc[0] = 0
        vel.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace infinite values with NaN
        data['vel'] = vel
    else:
        print("Please run addDist method to calculate distances between points first.")

def skmob_metric_calcs(df, method = 'GP', lat='lat', long = 'lng', datetime = 'datetime'):
    """
    Calculates scikit-mobility metrics for a dataframe with latitudes and longitudes.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with latitudes and longitudes.
    method : string
        Method used to generate the dataframe. Default is 'GP'.
    lat : string
        Name of the column containing latitudes. Default is 'lat'.
    long : string
        Name of the column containing longitudes. Default is 'lng'.
    datetime : string
        Name of the column containing datetimes. Default is 'datetime'.
    """
     # Make into TrajDataFrame
    tdf = skmob.TrajDataFrame(df, latitude=lat, longitude=long, datetime=datetime)

    # Calculate scikit-mobility metrics, name parameters using method name
    no_loc_pred = skmob.measures.individual._number_of_locations_individual(tdf)
    rg_pred = skmob.measures.individual._radius_of_gyration_individual(tdf).squeeze()
    k_rg_pred = skmob.measures.individual._k_radius_of_gyration_individual(tdf).squeeze()
    jumps_pred = skmob.measures.individual._jump_lengths_individual(tdf).squeeze()
    spat_burst_pred = burstiness(jumps_pred)
    loc_freq_pred = skmob.measures.individual._location_frequency_individual(tdf, normalize=True) # matrix
    rand_entr_pred = skmob.measures.individual._random_entropy_individual(tdf).squeeze()
    real_entr_pred = skmob.measures.individual._real_entropy_individual(tdf).squeeze()
    recency_pred = skmob.measures.individual._recency_rank_individual(tdf).squeeze()  # matrix
    freq_rank_pred = skmob.measures.individual._frequency_rank_individual(tdf).squeeze() # matrix
    uncorr_entr_pred = skmob.measures.individual._uncorrelated_entropy_individual(tdf).squeeze()
    max_dist_pred = skmob.measures.individual._maximum_distance_individual(tdf).squeeze()
    dist_straight_pred = skmob.measures.individual._distance_straight_line_individual(tdf).squeeze()
    waiting_time_pred = skmob.measures.individual._waiting_times_individual(tdf).squeeze() # array
    home_loc_pred = skmob.measures.individual._home_location_individual(tdf) # tuple
    max_dist_home_pred = skmob.measures.individual._max_distance_from_home_individual(tdf).squeeze()
    mob_network_pred = skmob.measures.individual._individual_mobility_network_individual(tdf) # big matrix
    
    setattr(tdf, f"no_loc_{method.lower()}_pred", no_loc_pred)
    setattr(tdf, f"rg_{method.lower()}_pred", rg_pred)
    setattr(tdf, f"k_rg_{method.lower()}_pred", k_rg_pred)
    setattr(tdf, f"jumps_{method.lower()}_pred", jumps_pred)
    setattr(tdf, f"spat_burst_{method.lower()}_pred", spat_burst_pred)
    setattr(tdf, f"loc_freq_{method.lower()}_pred", loc_freq_pred)
    setattr(tdf, f"rand_entr_{method.lower()}_pred", rand_entr_pred)
    setattr(tdf, f"real_entr_{method.lower()}_pred", real_entr_pred)
    setattr(tdf, f"recency_{method.lower()}_pred", recency_pred)
    setattr(tdf, f"freq_rank_{method.lower()}_pred", freq_rank_pred)
    setattr(tdf, f"uncorr_entr_{method.lower()}_pred", uncorr_entr_pred)
    setattr(tdf, f"max_dist_{method.lower()}_pred", max_dist_pred)
    setattr(tdf, f"dist_straight_{method.lower()}_pred", dist_straight_pred)
    setattr(tdf, f"waiting_time_{method.lower()}_pred", waiting_time_pred)
    setattr(tdf, f"home_loc_{method.lower()}_pred", home_loc_pred)
    setattr(tdf, f"max_dist_home_{method.lower()}_pred", max_dist_home_pred)
    setattr(tdf, f"mob_network_{method.lower()}_pred", mob_network_pred)

    return tdf

def preds_to_full_df(preds_lat, preds_long, test_df, train_df, 
                     unix='unix_min', datetime='date', lat='lat', long='long'):
    '''
    Function to merge the predictions with the original training set to create a full dataframe.
    
    '''
    # Create dataframe with GP predictions
    orig_preds_df = pd.DataFrame(test_df[unix], columns=[unix])
    orig_preds_df[datetime] = test_df[datetime]
    orig_preds_df[lat] = preds_lat
    orig_preds_df[long] = preds_long

    tdf = pd.concat([train_df, orig_preds_df], axis=0)

    # Sort by unix time
    tdf.sort_values(by=unix, inplace=True)

    # Rename datetime column to datetime
    tdf.rename(columns={datetime: 'datetime'}, inplace=True)

    return tdf

def matrix_acc(metric_pred, metric_test, metric_name, tolerance=1e-04):
    try:
        metric_tot = pd.concat([metric_pred.reset_index()[:len(metric_test)], metric_test.reset_index(drop=True)[:len(metric_pred)]], axis=1)
        metric_tot.columns = ['index', 'pred_lat', 'pred_lng', f'pred_{metric_name.lower()}', 
        'test_lat', 'test_lng', f'test_{metric_name.lower()}']
        metric_match = metric_tot.apply(lambda x: np.all(np.isclose([x.test_lat, x.test_lng, x[f'test_{metric_name.lower()}']], 
        [x.pred_lat, x.pred_lng, x[f'pred_{metric_name.lower()}']], atol=tolerance)), axis=1)
        metric_perc = np.count_nonzero(metric_match) / len(metric_match)
    except ValueError:
        metric_perc = 0
    return metric_perc

def burstiness(series):
    avg=series.mean()
    std=series.std()
    if (std+avg)==0:
        B=np.nan
    else:
        B=(std-avg)/(std+avg) # if std+avg=0
    return B

def uniqueid():
    seed = random.getrandbits(32)
    while True:
        yield seed
        seed += 1
        
def GPR_id(seed = random.seed(10)):
    seed
    GPR_id = str(next(uniqueid()))
    return GPR_id

def dec_floor(a, precision=1):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

def loc_based_filter(data, trip, m_threshold=200, lat='orig_lat', lon='orig_long', trip_ID='trip_ID'):
    """
    Filters out points that are not within a certain distance of the start and end points of a trip. 
    Achieves this by creating a bounding box around the start and end points, and then filtering out 
    points that are not within the bounding box.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing all points.
    trip : pandas dataframe
        Dataframe containing points for a single trip.
    m_threshold : int
        Distance in meters from start and end points to create bounding box.
    lat : str
        Name of the column containing the latitude.
    lon : str
        Name of the column containing the longitude.
    trip_ID : str
        Name of the column containing the trip ID.
    """
    # Get bounding box coordinates for trip
    start_lat, start_long = trip[lat].iloc[0], trip[lon].iloc[0]
    end_lat, end_long = trip[lat].iloc[-1], trip[lon].iloc[-1]

    upper_lat_s, upper_long_s = newCoords(start_lat, start_long, m_threshold, 0)
    right_lat_s, right_long_s = newCoords(start_lat, start_long, 0, m_threshold)
    lower_lat_s, lower_long_s = newCoords(start_lat, start_long, -m_threshold, 0)
    left_lat_s, left_long_s = newCoords(start_lat, start_long, 0, -m_threshold)

    upper_lat_e, upper_long_e = newCoords(end_lat, end_long, m_threshold, 0)
    right_lat_e, right_long_e = newCoords(end_lat, end_long, 0, m_threshold)
    lower_lat_e, lower_long_e = newCoords(end_lat, end_long, -m_threshold, 0)
    left_lat_e, left_long_e = newCoords(end_lat, end_long, 0, -m_threshold)

    # Exclude trips that do not start and end within the bounding boxes; group by trip_ID
    filtered_trips = data.groupby(trip_ID).filter(lambda x: 
                (x[lat].iloc[0] <= upper_lat_s) & 
                (x[lat].iloc[0] >= lower_lat_s) & 
                (x[lon].iloc[0] <= right_long_s) & 
                (x[lon].iloc[0] >= left_long_s) & 
                (x[lat].iloc[-1] <= upper_lat_e) & 
                (x[lat].iloc[-1] >= lower_lat_e) & 
                (x[lon].iloc[-1] <= right_long_e) & 
                (x[lon].iloc[-1] >= left_long_e))
    
    return filtered_trips

def trajectory_points_to_linestring(trajectory, speed_column="speed_kmph"):
    """
    A helper function to create a GeoDataFrame with LineString geometry based on the points of a given trajectory.
    Also calculates the average and standard deviation of the speed of the trajectory.
    
    Parameters
    ----------
    
    trajectory : mpd.Trajectory
    
        A MovingPandas Trajectory object which will be used for creating the GeoDataFrame with LineString geometry 
        and some relevant trajectory-related attributes.
        
    speed_column : str
    
        A column containing the travel speed.
    
    """
    # Parse the GeoDataFrame
    df = trajectory.df
    
    # Remove observations that are very slow (< 1 kmph)
    df = df.loc[df[speed_column] > 1].copy()
    
    # If there were only less than 3 observations return empty GeoDataFrame
    # This will clean the data from trajectories 
    if len(df) < 3:
        return None
    
    # Sort the observations
    df = df.sort_index()
    
    # What is the average speed of the route
    avg_speed = df[speed_column].mean()
    std_speed = df[speed_column].std()
    
    # Parse other useful values
    vehicle_id = df["vehicle_id"].unique()[0]
    route_id = df["route_id"].unique()[0]
    direction_id = df["direction_id"].unique()[0]
    start_time = df["timestamp"].head(1).values[0]
    
    # Create LineString geometry
    geom = trajectory.to_linestring()
    
    # Store the relevant data
    data = {"geometry": [geom], "route_id": route_id, 
            "vehicle_id": vehicle_id, 
            "direction_id": direction_id,
            "avg_speed": avg_speed,
            "std_speed": std_speed,
           }
    return gpd.GeoDataFrame(data, crs="epsg:4326")