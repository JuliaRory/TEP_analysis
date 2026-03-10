from h5py import File 
import os 
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from src.analysis.find_GFP_peaks import find_gfp_peaks, extract_gfp_windows

DATASET = "pilot/M1_SMA_differentPhases/clean_epochs"
Fs = 5000
ms_to_samples = lambda t: int(t / 1000 * Fs)

TEP_WINDOWS = {
    "P30": (20, 40),
    "N45": (35, 60),
    "P60": (50, 80),
    "N100": (80, 140),
    "P180": (150, 230),
}

def process_file(filename):
    with File(filename, "r") as h5f:
        epochs = h5f['data/Epruned'][:]
        time = h5f['data/tvec'][:][0]
        conds = h5f['data/conds'][:]
    
    conds = np.vstack([[False, False, False], conds[:-1]])
    
    pre_mask = conds[:, 0].astype(bool)
    imag_mask = conds[:, 1].astype(bool)
    post_mask = conds[:, 2].astype(bool)

    TEP_pre = np.mean(epochs[:, pre_mask, :], axis=1)#[:, idxs_good]
    TEP_im = np.mean(epochs[:, imag_mask, :], axis=1)#[:, idxs_good]
    TEP_post = np.mean(epochs[:, post_mask, :], axis=1)#[:, idxs_good]

    all_conditions = {"pre": TEP_pre-TEP_post, "imagery": TEP_im-TEP_post, "post": TEP_post-TEP_post}

    rows = []
    Fs = 5000

    for cond_name, TEPs in all_conditions.items():
        results, _ = extract_gfp_windows(TEPs, time, TEP_WINDOWS, mode="peak", sfreq=Fs, smooth_ms=10)
        for comp_name, comp_res in results.items():
            rows.append({
                "subject": os.path.basename(filename),
                "condition": cond_name,
                "component": comp_name,
                "amplitude": comp_res["amplitude"],
                "latency": comp_res["latency"]
            })

    return rows


if __name__ == "__main__":
    
    data_folder = os.path.join(r"./data", DATASET)
    records = os.listdir(data_folder)

    df_all = []
    for record in tqdm(records):
        filename = os.path.join(data_folder, record)
        try:
            rows = process_file(filename)
            df_all.extend(rows)
        except Exception as e:
            print(f"Ошибка в {record}: {e}")

    # превращаем в pandas DataFrame
    df_all = pd.DataFrame(df_all)

    # сохраняем в CSV (или JSON)
    df_all.to_csv("./results/diff_phases_meanDiffGFP.csv", index=False)
    # df_all.to_json("./results/diff_phases_meanGFP.json", orient="records", indent=4)

    print("Готово! DataFrame shape:", df_all.shape)