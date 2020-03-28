
import itertools
from functools import partial
import copy
import pandas as pd
import numpy as np
from scipy import signal
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA

def load_data_from_mat(path):
    data = loadmat(path)['bearing']
    return_dict = {
        "gs": data['gs'][0][0],
        "sr": data['sr'][0][0][0][0],
        "load": data['load'][0][0][0][0],
        "rate": data['rate'][0][0][0][0],
    }
    return return_dict

def load_data(files):
    data_dict = {}
    for f in files:
        data = load_data_from_mat(f)
        data_name = f.split('/')[-1].split('.')[0]
        data_dict[data_name] = data
    return data_dict

def get_segments_list(data_dict):
    all_segments_list = []
    for k,v in data_dict.items():
        for i in v['segment'].keys():
            all_segments_list.append((k,i))
    return all_segments_list


def init_data_dicts(data_dict, n_splits=1, inplace=True):
    if not inplace:
        data_dict = copy.deepcopy(data_dict)
    for key,value in data_dict.items():
        segments_dict = {}
        segments_fft_dict = {}
        for i,seg in enumerate(np.split(value['gs'], n_splits)):
            segments_dict[i] = seg
            freq, spectrum = signal.welch(seg[:,0], fs=value['sr'], nperseg=1000)
            segments_fft_dict[i] = spectrum
        value['segment'] = segments_dict
        value['segment_fft'] = segments_fft_dict
        value['fft_freq'] = freq
    return data_dict


def plot_distances_bar(ref,distances,ax=None):

    def prepare_crosstab(df,aggfunc=np.nanmean):
        df = pd.crosstab(df['set1'],df['set2'],values=df['dist'],aggfunc=aggfunc)
        idx = df.columns.union(df.index)
        df = df.reindex(index=idx, columns=idx)
        df = df.mask(df.isna(),df.T)
        return df

    df = pd.DataFrame(distances,columns=['set1','seg1','set2','seg2','dist'])
    df_mean = prepare_crosstab(df,aggfunc=np.nanmean)
    df_std = prepare_crosstab(df,aggfunc=np.nanstd)

    ax_ = df_mean.loc[ref].T.plot.bar(yerr=df_std,ax=ax)
    ax_.set_ylabel("Distance")
    ax_.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax_.grid(linewidth=0.3)

    return df_mean
