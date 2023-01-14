import numpy as np

# TODO: Fix input

def Temp_Ocp(data):
        bins = np.arange(min(data['time']), max(data['time'])+1, 60) # divide total time period into one-minute bins
        Nb = len(bins)
        obs = list()
        for i, j in enumerate(bins):
            if i == Nb:
                break
            hi = list()
            for k in range(bins[i], bins[i+1]):
                condition = [k in data['time'].values]
                hi.append(any(condition))
            if any(hi):
                obs.append(1)
            if i == (Nb-2):
                break
        obs.append(1)
        temp_ocp = len(obs) / len(bins)
        return temp_ocp