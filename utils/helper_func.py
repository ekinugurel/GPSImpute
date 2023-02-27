import holidays
import numpy as np
import matplotlib
import pandas as pd
import random
from math import radians, cos, sin, asin, sqrt, pi


def init():
    matplotlib.pyplot.rcdefaults()
    
def plot_2d_func(func, n_rows=1, n_cols=1, title=None):
    grid_size = 100
    x_grid = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    x_grid = np.hstack((x_grid[0].reshape(-1, 1), x_grid[1].reshape(-1, 1)))
    y = func(x_grid)
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 6))
    ax = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    ax.plot_surface(x_grid[:, 0].reshape(grid_size, grid_size), x_grid[:, 1].reshape(grid_size, grid_size),
                    y.reshape(grid_size, grid_size),
                    cmap=cm.jet, rstride=1, cstride=1)
    if title is not None:
        ax.set_title(title)
    return fig

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
    """
    r = 6371
    new_lat  = lat  + (dy*0.001 / r) * (180 / pi)
    new_lon = lon + (dx*0.001 / r) * (180 / pi) / cos(lat * pi/180)
    return new_lat, new_lon

def newCoordsAlt(lat1, lon1, d, brng, R = 6378.1):
    """
    Calculates a new lat/lon from an old lat/lon + distance and bearing
    """
    d = d/1000 # convert distance to km
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    
    lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
         math.cos(lat1)*math.sin(d/R)*math.cos(brng))
    
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
                 math.cos(d/R)-math.sin(lat1)*math.sin(lat2))
    
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return lat2, lon2

def addDist(data, type=haversine_np):
    """
    Add distance column to a dataframe with latitudes and longitudes. Type specifies whether to use the Haversine distance (default) or the geodesic distance.
    """
    print("Adding distance column to dataframe...")
    if type == haversine_np:
        lat1, lon1 = data['orig_lat'], data['orig_long']
        lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
        dist = type(lat1, lon1, lat2, lon2).fillna(0)
        data['dist'] = dist
    elif type == geodesic:
        lat1, lon1 = data['orig_lat'], data['orig_long']
        lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
        dist = type((lat1, lon1), (lat2, lon2)).miles.fillna(0)
        data['dist'] = dist
        
def addVel(data):
    print("Adding velocity column to dataframe...")
    if 'dist' in longitudinal.data.columns:
        lat1, lon1 = longitudinal.data['orig_lat'], longitudinal.data['orig_long']
        lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
        dist = longitudinal.data['dist'].fillna(0)
        time_diff = (longitudinal.data['unix_min'] - longitudinal.data['unix_min'].shift(1)).fillna(0)
        vel = dist / time_diff
        vel.iloc[0] = 0
        vel.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace infinite values with NaN
        longitudinal.data['vel'] = vel
    else:
        print("Please run addDist method to calculate distances between points first.")

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