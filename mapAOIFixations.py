# import packages
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as matplotlib
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import seaborn as sns
import scipy.io as io
import pandas as pd
from pandas import DataFrame, read_csv
from nivlink import Screen, Raw, Epochs, align_to_aoi, compute_fixations, plot_heatmaps
import cv2
import readline
from math import dist
from scipy.spatial.distance import squareform, pdist

import warnings
from scipy.stats import kde
import nivlink
import ipywidgets as wdg
from scipy.stats import iqr

# import custom functions
import analysisHelperFunctions as ahf

## Define global variables ##
et_data_dir = os.getcwd().strip('ldm-analysis') + 'ProcessedData/' # this expects ProcessedData to be one directory up from the analysis repo
image_dir = os.getcwd().strip('ldm-analysis') + 'FinalStimuli/ByNumber/' # this expects FinalStimuli to be one directory up from the analysis repo
raw_data_dir = os.getcwd().strip('ldm-analysis') + 'RawData/' # this expects RawData to be one directory up from the analysis repo

## Define screen metadata.
xdim, ydim, n_screens = 1280, 1024, 1 
aoisidelength = 162
n_aois = 9


def makeFixationPlot(trial_featmap, fixations_block_trial, block_centers, indices, labels, xdim, ydim, height=3, ticks=False, cmap=None):
    ## Initialize plot.
    ratio = float(xdim) / float(ydim)
    fig, ax = plt.subplots(1,1,figsize=(ratio*height, height))            
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    ## Initialize colormap.
    if cmap is None:
        
        # Collect hex values from standard colormap.
        cmap = cm.get_cmap('tab20', 20)
        
        colors = []
        for i in range(cmap.N):
            rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
            colors.append(matplotlib.colors.rgb2hex(rgb))

        colors = colors[:len(labels)]

        # Add black.
        if np.any(indices==0): colors = np.insert(colors, 0, 'k')

        # Construct new colormap.
        cmap = ListedColormap(colors)

    ## Plotting.
    cbar = ax.imshow(indices[:,:,-1].T, cmap=cmap, aspect='auto', vmin=0, vmax=len(labels))
    fig.colorbar(cbar, cax, ticks=np.arange(len(cmap.colors)))
    if not ticks: ax.set(xticks=[], yticks=[])        

    for a in range(0, n_aois):
        col_names = ['aoi' + str(a) + '_x', 'aoi' + str(a) + '_y']
        this_aoi_centers = block_centers[col_names].iloc[0].values

        xy = (this_aoi_centers[0], this_aoi_centers[1])
        
        # grab fixations for this aoi
        aoi_fixations = fixations_block_trial[fixations_block_trial.AoI.astype(int) == a]
        circles = {}
        for index, row in aoi_fixations.iterrows():
            circles[index] = plt.Circle(xy, row.Duration*100, alpha=0.7)
            ax.add_patch(circles[index])

        feat_num = int(trial_featmap[a])
        ax.annotate(str(feat_num), # this is the text
                    xy, # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,0), # distance from text to points (x,y)
                    ha='center')
    return fig, ax


def loadData(subj_id, block, trial):
    ## Load data.
    data, raw_pos_data, messages, sfreq = ahf.load_subj_data(subj_id)

    ## Mark run onsets. 
    run_onsets = ahf.get_run_onsets(messages)
    n_blocks, d = run_onsets.shape

    # Load feature map
    featmap_file_path = et_data_dir + str(subj_id) + 'featmap.mat'
    featmap_mat = io.loadmat(featmap_file_path, struct_as_record=False, squeeze_me=True)
    featmap = np.array(featmap_mat["features_aoi_map"])

    # Re-format events dataframe. 
    events_df = ahf.rereference_events(subj_id, n_blocks, run_onsets, sfreq)

    # Subselect this block's events.
    this_block = events_df.loc[events_df['block'] == float(block)]
    this_block_idx = events_df.index[events_df['block'] == float(block)].values 
    n_trials_block = this_block.shape[0]

    # Subselect featmap.
    this_block_featmap = featmap[this_block_idx,:]
    # Add "null" AoI.
    this_block_featmap = np.hstack((this_block_featmap, np.ones((this_block_featmap.shape[0],1)) * 10))
    n_trials_block = this_block.shape[0]
    # Subselect trial
    trial_idx = trial-1
    this_trial_featmap = this_block_featmap[trial_idx]

    # Load fixations for sub
    fixations = pd.read_csv(et_data_dir + str(subj_id) + 'fixations.csv')
    fixations = ahf.loadFixations(fixations, subj_id)

    # grab fixations for block and trial
    fixations_block = fixations[fixations.BlockNumber == block]
    fixations_block_trial = fixations_block[fixations_block.Trial == (block-1)*n_trials_block + trial]

    # Load subject log
    log_file_path = raw_data_dir + 'Subj' + str(subj_id) + 'Log.mat'
    log_mat = io.loadmat(log_file_path)
    data = log_mat["Data"]
    instructions = log_mat["Instructions"]
    params = log_mat["Parms"]
    print(params)
    print(data)
    print(instructions)
    return
# driver function
if __name__=="__main__":
    if len(sys.argv) < 2:
        sub_id = input("Enter the Subject ID: \n")
        block_num = int(input("Enter the Block Number: \n"))
        trial_num = int(input("Enter the Trial Number: \n"))
    else:
        sub_id = sys.argv[1]
        block_num = int(sys.argv[2])
        trial_num = int(sys.argv[3])
    fig, ax = loadData(sub_id, block_num, trial_num)
    plt.draw()
    plt.show()