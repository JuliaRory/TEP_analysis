from h5py import File 
import os 
import numpy as np
from tqdm import tqdm

from src.visualization.TEPs_different_phases import plot_TEPs_per_phases

DATASET = "pilot/M1_SMA_differentPhases/clean_epochs"
Fs = 5000
ms_to_samples = lambda t: int(t / 1000 * Fs)

def plot_teps(filename):
    with File(filename, "r") as h5f:
        epochs = h5f['data/Epruned'][:]
        time = h5f['data/tvec'][:][0]
        conds = h5f['data/conds'][:]

    #conds = np.vstack([[False, False, False], conds[:-1]])
    
    pre_mask = conds[:, 0].astype(bool)
    imag_mask = conds[:, 1].astype(bool)
    post_mask = conds[:, 2].astype(bool)

    TEP_pre = np.mean(epochs[:, pre_mask, :], axis=1)#[:, idxs_good]
    TEP_im = np.mean(epochs[:, imag_mask, :], axis=1)#[:, idxs_good]
    TEP_post = np.mean(epochs[:, post_mask, :], axis=1)#[:, idxs_good]

    fig = plot_TEPs_per_phases(TEP_pre, TEP_im, TEP_post, time, ms_to_samples)

    fl = os.path.basename(filename)
    fig.suptitle(fl)
    
    output_filename = os.path.join(r"./results/TEPs_differentPhases", os.path.splitext(fl)[0]+".png")
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    
    data_folder = os.path.join(r"./data", DATASET)
    records = os.listdir(data_folder)
    for record in tqdm(records):
        # filename = os.path.join(data_folder, record)
        # plot_teps(filename)
        try:
            filename = os.path.join(data_folder, record)
            plot_teps(filename)
        except:
            print(record)
