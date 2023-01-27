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

def addDist(data, type=haversine):
    """
    Add distance column to a dataframe with latitudes and longitudes. type specifies whether to use the Haversine distance (default) or the geodesic distance.
    """
    if type == haversine:
        cnt = 1
        dist = list()
        for i, j in zip(data['orig_lat'], data['orig_long']):
            dist.append(haversine(i, j, data['orig_lat'][cnt], data['orig_long'][cnt]))
            cnt += 1
            if cnt == len(data):
                dist.insert(0, 0)
                break
        data['dist'] = dist
    elif type == geodesic:
        cnt = 1
        dist = list()
        for i, j in zip(data['orig_lat'], data['orig_long']):
            dist.append(geodesic(i, j, data['orig_lat'][cnt], data['orig_long'][cnt]))
            cnt += 1
            if cnt == len(data):
                dist.insert(0, 0)
                break
        data['dist'] = dist
        
def addVel(data):
    dt = list(np.diff(data['unix_start_t']))
    dt.insert(0, 0)
    if 'dist' in list(data.columns):
        data['vel'] = data['dist'] / dt
        data['vel'][0] = 0
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