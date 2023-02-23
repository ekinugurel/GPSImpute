import numpy as np

def tempOcp(data, bin_len = 5):
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
    bins = np.arange(min(data['unix_start_t_min']), max(data['unix_start_t_min'])+1, bin_len )
    Nb = len(bins)
    obs = list()
    for i, j in enumerate(bins):
        if i == Nb:
            break
        hi = list()
        for k in range(int(bins[i]), int(bins[i+1])):
            condition = [k in data['unix_start_t_min'].values]
            hi.append(any(condition))
        if any(hi):
            obs.append(1)
        if i == (Nb-2):
            break
    obs.append(1)
    temp_ocp = float(len(obs) / len(bins))
    return temp_ocp

def simulate_gaps(data, user_id, target_temp_ocp, bin_len=5):
    user_data = data[data['user_id'] == user_id]
    current_temp_ocp = tempOcp(user_data, bin_len)
    if current_temp_ocp <= target_temp_ocp:
        print(f"User {user_id} already has temporal occupancy of {current_temp_ocp}.")
        return None
    else:
        target_bins = int(len(user_data) * target_temp_ocp)
        current_bins = len(np.unique(user_data['unix_start_t_min'])) - 1
        if target_bins >= current_bins:
            print(f"Cannot decrease temporal occupancy for user {user_id} to {target_temp_ocp}.")
            return None
        else:
            gap_size = int((current_bins - target_bins) / (target_bins + 1))
            gap_starts = np.arange(gap_size, current_bins, gap_size+1)
            gap_ends = gap_starts + int(gap_size/2)
            for start, end in zip(gap_starts, gap_ends):
                start_time = user_data.iloc[start]['unix_start_t_min']
                end_time = user_data.iloc[end]['unix_start_t_min']
                user_data = user_data[user_data['unix_start_t_min'] < start_time]\
                            .append(user_data[user_data['unix_start_t_min'] > end_time])
            return user_data.reset_index(drop=True)