import numpy as np
import random

def tempOcp(data, unix_col = 'unix_min', bin_len = 5):
    """
    Calculates the temporal occupancy of a given mobile data sequence.

    Parameters
    ----------
    data: pd.DataFrame
        Trajectory data that needs to be processed
    
    bin_len : INT, optional
        Number of minutes in a time interval (bin). The default is 5.

    Returns
    -------
    temp_ocp : FLOAT
    Temporal occupancy metric.

    """
    unix_col = unix_col
    bins = np.arange(min(data[unix_col]), max(data[unix_col])+1, bin_len )
    Nb = len(bins)
    obs = list()
    for i, j in enumerate(bins):
        if i == Nb:
            break
        hi = list()
        for k in range(int(bins[i]), int(bins[i+1])):
            condition = [k in data[unix_col].values]
            hi.append(any(condition))
        if any(hi):
            obs.append(1)
        if i == (Nb-2):
            break
    obs.append(1)
    temp_ocp = float(len(obs) / len(bins))
    return temp_ocp

def simulate_gaps(data, target_temp_ocp, user_id=None, unix_col = 'unix_min', bin_len=5):
    """
    Simulates gaps in the data by removing random time intervals (bins) until the temporal occupancy is below the target.

    Parameters
    ----------
    data: pd.DataFrame
        Trajectory data that needs to be processed
    
    target_temp_ocp : FLOAT
        Target temporal occupancy metric.
    
    user_id : STR, optional
        User ID column name. The default is None.

    bin_len : INT, optional
        Number of minutes in a time interval (bin). The default is 5.

    Returns
    -------
    sparse_data : pd.DataFrame
        Trajectory data with simulated gaps.
    
    sparse_data[unix_col] : pd.Series
        Unix timestamps of the sparse data.
    
    new_ocp : FLOAT
        New temporal occupancy metric.
    """
    bins = np.arange(min(data[unix_col]), max(data[unix_col])+1, bin_len )

    # Dictionary comprehension to map values to bins
    bins_dict = {b: [x for x in data[unix_col] if b <= x < b+bin_len] for b in bins}

    # New dictionary with non-empty bins
    non_empty_bins_dict = {k: v for k, v in bins_dict.items() if len(v) > 0}

    # Remove bins until the temporal occupancy is below the target
    sparse_data = data.copy()

    while tempOcp(data = sparse_data, unix_col = unix_col, bin_len = bin_len) > target_temp_ocp:
        # Randomly choose a bin to remove
        bin_to_remove = np.random.choice(list(non_empty_bins_dict.keys()))
        # Remove all values in this bin from original data
        sparse_data = sparse_data[~sparse_data[unix_col].isin(non_empty_bins_dict[bin_to_remove])]
        # Remove duplicates in the unix column
        sparse_data = sparse_data.drop_duplicates(subset=[unix_col])
        # Update the dictionary
        bins_dict = {b: [x for x in sparse_data[unix_col] if b <= x < b+bin_len] for b in bins}
        non_empty_bins_dict = {k: v for k, v in bins_dict.items() if len(v) > 0}
    new_ocp = tempOcp(data = sparse_data, unix_col = unix_col, bin_len = bin_len)
    print(f"New temporal occupancy is {new_ocp}.")
    return sparse_data.reset_index(drop=True), sparse_data[unix_col], new_ocp