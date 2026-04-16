import matplotlib.pyplot as plt 
from mne.viz import plot_topomap
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from src.utils.montage_processing import get_topo_positions
from src.analysis.find_GFP_peaks import find_gfp_peaks

from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap
viridisBig = cm.get_cmap('jet')
newcmp = ListedColormap(viridisBig(np.linspace(0, 1, 15)))

REGION_COLORS = {
    "frontal": "#4C72B0",        # синий
    "central": "#DD8452",        # оранжевый (sensorimotor)
    "parietal": "#DDCC77",       # жёлтый
    "temporal": "#55A868",       # зелёный
    "occipital": "#C44E52",      # красный
    "prefrontal": "#8172B2",     # фиолетовый
}
EEG64_REGION_COLOR = {
    "dark": "#040308", 
    # --- Prefrontal ---
    "Fp1": "#8172B2",
    "Fp2": "#8172B2",
    "AF3": "#8172B2",
    "AF4": "#8172B2",
    "AF7": "#8172B2",
    "AF8": "#8172B2",
    'FT9': "#8172B2",
    'FT10': "#8172B2",
    'Fpz': "#8172B2",

    # --- Frontal ---
    "F1": "#4C72B0",
    "F2": "#4C72B0",
    "F3": "#4C72B0",
    "F4": "#4C72B0",
    "F5": "#4C72B0",
    "F6": "#4C72B0",
    "F7": "#4C72B0",
    "F8": "#4C72B0",
    "Fz": "#4C72B0",

    # --- Fronto-central / Premotor → central color ---
    "FC1": "#DD8452",
    "FC2": "#DD8452",
    "FC3": "#DD8452",
    "FC4": "#DD8452",
    "FC5": "#DD8452",
    "FC6": "#DD8452",
    "FCz": "#DD8452",

    # --- Central (Sensorimotor) ---
    "C1": "#DD8452",
    "C2": "#DD8452",
    "C3": "#DD8452",
    "C4": "#DD8452",
    "C5": "#DD8452",
    "C6": "#DD8452",
    "Cz": "#DD8452",

    # --- Centro-parietal ---
    "CP1": "#DDCC77",
    "CP2": "#DDCC77",
    "CP3": "#DDCC77",
    "CP4": "#DDCC77",
    "CP5": "#DDCC77",
    "CP6": "#DDCC77",
    "CPz": "#DDCC77",

    # --- Parietal ---
    "P1": "#DDCC77",
    "P2": "#DDCC77",
    "P3": "#DDCC77",
    "P4": "#DDCC77",
    "P5": "#DDCC77",
    "P6": "#DDCC77",
    "P7": "#DDCC77",
    "P8": "#DDCC77",
    "Pz": "#DDCC77",

    # --- Temporal ---
    "FT7": "#55A868",
    "FT8": "#55A868",
    "T7": "#55A868",
    "T8": "#55A868",
    "TP7": "#55A868",
    "TP8": "#55A868",
    "TP9": "#55A868",
    "TP10": "#55A868",

    # --- Parieto-occipital ---
    "PO3": "#C44E52",
    "PO4": "#C44E52",
    "PO7": "#C44E52",
    "PO8": "#C44E52",
    "POz": "#C44E52",

    # --- Occipital ---
    "O1": "#C44E52",
    "O2": "#C44E52",
    "Oz": "#C44E52",
}

CED_FILE = r"./resources/mks64_standard.ced"
df = pd.read_csv(CED_FILE, sep='\t')
df["colors"] = df["labels"].map(EEG64_REGION_COLOR)
colors = df["colors"].values

positions = get_topo_positions(CED_FILE)

def plot_TEPs_per_phases(TEP_pre, TEP_im, TEP_post, time, ms_to_samples):
    fig = plt.figure(figsize=(22, 7))
    gs = gridspec.GridSpec(3, 6,  height_ratios = [1, 1, 2], wspace=0.1)

    titles = ["pre-imagery", "imagery", "post-imagery"]

    time_limits = [15, 120]
    shift = 200
    time_range = np.arange(ms_to_samples(time_limits[0]+shift), ms_to_samples(time_limits[1]+shift))
    def find_max(X, range):
        return np.max(np.abs(X[range, :]))

    amp = np.max([find_max(TEP_pre, time_range), find_max(TEP_im, time_range), find_max(TEP_post, time_range)])
    h = 2
    axis = [plt.subplot(gs[h, 0:2]), plt.subplot(gs[h, 2:4]), plt.subplot(gs[h, 4:6])]
    def plot_butterfly(TEPs, ax, time, amp, time_range):
        for color, TEP in zip(colors, TEPs.T):
            ax.plot(time, TEP, color=color)
        ax.set_ylim(-amp, amp)
        ax.set_xlim(time_range)
        ax.set_title(titles[i])
    
    for i, TEPs in enumerate([TEP_pre, TEP_im, TEP_post]):
        plot_butterfly(TEPs,  axis[i], time=time, amp=amp, time_range=time_limits)

    axis[0].set_xlabel("Time [ms]")
    axis[0].set_ylabel("[uV]")

    axis[1].set_yticks([])
    axis[2].set_yticks([])

    time_moments = [25, 45, 60, 120]
    ranges = [8, 8, 10, 30]

    def plot_timemoments(ax, time_moments):
        for time_moment in time_moments:
            ax.axvline(time_moment, color="black", linestyle="--", linewidth=.3)

    for ax in axis:
        plot_timemoments(ax, time_moments)

    legend_elements = [Patch(facecolor=color, edgecolor='k', label=area)
                    for area, color in REGION_COLORS.items()]
    axis[2].legend(handles=legend_elements, title="Области мозга", loc=[1.03,.35])


    for i, TEPs in enumerate([TEP_pre, TEP_im, TEP_post]):
        axis = [plt.subplot(gs[0, 0+2*i]), plt.subplot(gs[0, 1+2*i]), plt.subplot(gs[1, 0+2*i]), plt.subplot(gs[1, 1+2*i])]
        for j in np.arange(len(time_moments)):
                time_moment = time_moments[j]+shift
                range = ranges[j]
                start, end = time_moment - range, time_moment + range         # ms
                time_moment = np.arange(ms_to_samples(start), ms_to_samples(end)+1)
                TEP_t = np.mean(TEPs[time_moment, :], 0)
                im, cn = plot_topomap(TEP_t, positions,  image_interp='cubic', ch_type='eeg', #names = ch_labels,
                        size=5, show=False, contours=4, sphere=0.5, cmap=newcmp, extrapolate='head', axes=axis[j], vlim=[-amp, amp])
                fig.colorbar(im)
                axis[j].set_title("{} ms".format(time_moments[j]))

    return fig 

def plot_TEPs_per_phases_GFP(TEP_pre, TEP_im, TEP_post, time, ms_to_samples):
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    colors = ["#7BBDE1", "#1F78B4", "#0404AF"]
    
    max_x = 200
    max_y = 25

    shift = 200
    start = ms_to_samples(10 + shift)
    
    for title, TEPs, color in zip(["pre imagery", "imagery",  "post imagery"], [TEP_pre, TEP_im, TEP_post], colors):
        t_peaks, a_peaks, gfp_s, peaks, props, idxs = find_gfp_peaks(TEPs, time, start_sample=start)
        ax.plot(time, gfp_s, label=title, color=color)
        # ax.scatter(t_peaks, a_peaks, color='red')
        # ax.scatter(time[idxs], gfp_s[idxs], color='red')
        for i, t in enumerate(t_peaks):
            ax.axvline(t, color=color, linewidth=1)
        # print(title, "Peaks at (ms):", t_peaks)
    
    for interval in [[15, 35], [45, 75], [85, 150], [170, 250]]:
        ax.fill_betweenx(np.arange(max_y), interval[0], interval[1], color="lightgrey")

    ax.set_ylim(0, max_y)
    ax.set_xlim(0, max_x)
    ax.set_xticks(np.arange(0, max_x, 25))

    ax.set_ylabel("Global Field Power [µV]")
    ax.set_xlabel("Time [ms]")
    ax.grid(alpha=.4, linestyle="--")
    ax.legend()

    
    return fig