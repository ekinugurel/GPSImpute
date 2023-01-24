import numpy as np

def tempOcp(data, bin_len = 5):
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