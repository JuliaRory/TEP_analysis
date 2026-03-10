from h5py import File 
import os 
import json
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from src.analysis.find_GFP_peaks import find_gfp_peaks, extract_gfp_windows

DATASET = "pilot/M1_SMA_differentPhases/clean_epochs"
Fs = 5000
ms_to_samples = lambda t: int(t / 1000 * Fs)

def find_peaks(filename):
    with File(filename, "r") as h5f:
        epochs = h5f['data/Epruned'][:]
        time = h5f['data/tvec'][:][0]
        conds = h5f['data/conds'][:]
    
    pre_mask = conds[:, 0].astype(bool)
    imag_mask = conds[:, 1].astype(bool)
    post_mask = conds[:, 2].astype(bool)

    TEP_pre = np.mean(epochs[:, pre_mask, :], axis=1)#[:, idxs_good]
    TEP_im = np.mean(epochs[:, imag_mask, :], axis=1)#[:, idxs_good]
    TEP_post = np.mean(epochs[:, post_mask, :], axis=1)#[:, idxs_good]

    shift = 200
    start = ms_to_samples(10 + shift)

    peaks_all = []
    
    for title, TEPs in zip(["pre", "imagery",  "post"], [TEP_pre, TEP_im, TEP_post]):
        t_peaks, a_peaks, gfp_s, peaks, props, idxs = find_gfp_peaks(TEPs, time, start_sample=start)
        peaks_all.append({"cond": title, "t_peaks": t_peaks.tolist(), "amp": gfp_s[idxs].tolist()})

    return peaks_all


if __name__ == "__main__":
    
    data_folder = os.path.join(r"./data", DATASET)
    records = os.listdir(data_folder)

    peaks_all = {}
    for record in tqdm(records):
        # filename = os.path.join(data_folder, record)
        # peaks = find_peaks(filename)
        # for peak in peaks:
        #     peak["record"] = record
        #     peaks_all.append(peak)
        try:
            filename = os.path.join(data_folder, record)
            peaks = find_peaks(filename)
            peaks_all[record] = peaks
            # for peak in peaks:
            #     peak["record"] = record
            #     peaks_all.append(peak)
        except:
            print(record)
    
    # Сохраняем в файл data.json
    filename = r"./results/diff_phases_peaks.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(peaks_all, f, ensure_ascii=False, indent=4)