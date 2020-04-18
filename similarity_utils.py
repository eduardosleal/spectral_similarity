import itertools
import copy
import pandas as pd
import numpy as np
from scipy import signal
from scipy.io import loadmat
from sklearn.covariance import EmpiricalCovariance


def load_data_from_mat(path):
    data = loadmat(path)["bearing"]
    return_dict = {
        "gs": data["gs"][0][0],
        "sr": data["sr"][0][0][0][0],
        "load": data["load"][0][0][0][0],
        "rate": data["rate"][0][0][0][0],
    }
    return return_dict


def load_data(files):
    data_dict = {}
    for f in files:
        data = load_data_from_mat(f)
        data_name = f.split("/")[-1].split(".")[0]
        data_dict[data_name] = data
    return data_dict


def get_segments_list(data_dict):
    all_segments_list = []
    for k, v in data_dict.items():
        for i in v["segment"].keys():
            all_segments_list.append((k, i))
    return all_segments_list


def init_data_dicts(data_dict, n_splits=1, inplace=True):
    if not inplace:
        data_dict = copy.deepcopy(data_dict)
    for key, value in data_dict.items():
        segments_dict = {}
        segments_fft_dict = {}
        for i, seg in enumerate(np.split(value["gs"], n_splits)):
            segments_dict[i] = seg
            freq, spectrum = signal.welch(seg[:, 0], fs=value["sr"], nperseg=2000)
            segments_fft_dict[i] = spectrum
        value["segment"] = segments_dict
        value["segment_fft"] = segments_fft_dict
        value["fft_freq"] = freq
    return data_dict


def plot_similarities_bar(ref, similarities, ax=None):
    def prepare_crosstab(df, aggfunc=np.nanmean):
        df = pd.crosstab(df["set1"], df["set2"], values=df["sim"], aggfunc=aggfunc)
        idx = df.columns.union(df.index)
        df = df.reindex(index=idx, columns=idx)
        df = df.mask(df.isna(), df.T)
        return df

    df = pd.DataFrame(similarities, columns=["set1", "seg1", "set2", "seg2", "sim"])
    df_mean = prepare_crosstab(df, aggfunc=np.nanmean)
    df_std = prepare_crosstab(df, aggfunc=np.nanstd)

    if df_std.shape != df_mean.shape:
        df_std = None

    ax_ = df_mean.loc[ref].T.plot.bar(yerr=df_std, ax=ax)
    ax_.set_ylabel("Similarity")
    ax_.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax_.grid(linewidth=0.3)

    return df_mean


def compute_similarity(func, data_dict, apply_before=lambda x: x):
    all_segments_list = get_segments_list(data_dict)
    similarities = []
    all_segments_iter = itertools.combinations(all_segments_list, 2)
    for (key1, split1), (key2, split2) in all_segments_iter:
        vec1 = data_dict[key1]["segment_fft"][split1]
        vec2 = data_dict[key2]["segment_fft"][split2]
        vec1 = apply_before(vec1)
        vec2 = apply_before(vec2)
        sim = func(vec1, vec2)
        similarities.append((key1, split1, key2, split2, sim))
    return similarities


def init_covariance(data_dict, n_splits=6):
    data_dict_tmp = init_data_dicts(data_dict, n_splits=n_splits, inplace=False)
    for k, v in data_dict_tmp.items():
        spectra = np.vstack(np.array(list(data_dict_tmp[k]["segment_fft"].values())))
        min_cov = EmpiricalCovariance()
        min_cov.fit(spectra)
        data_dict[k]["cov"] = min_cov


def compute_mahalanobis_similarity(data_dict):
    init_covariance(data_dict, n_splits=6)
    all_segments_list = get_segments_list(data_dict)
    similarities = []
    all_segments_iter = itertools.combinations(all_segments_list, 2)
    for (key1, split1), (key2, split2) in all_segments_iter:
        vec2 = data_dict[key2]["segment_fft"][split2]
        min_cov = data_dict[key1]["cov"]
        sim = min_cov.mahalanobis(vec2[None, :])[0]
        similarities.append((key1, split1, key2, split2, sim))
    return similarities
