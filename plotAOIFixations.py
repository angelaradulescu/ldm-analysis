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
import plotAOIFeatures as paf

## Define global variables ##
et_data_dir = os.getcwd().strip('ldm-analysis') + 'ProcessedData/'# this expects ProcessedData to be one directory up from the analysis repo
image_dir = os.getcwd().strip('ldm-analysis') + 'FinalStimuli/ByNumber/'

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


def plotAOIFixations(subj_id, block, trial):
    ## Load data.
    data, raw_pos_data, messages, sfreq = paf.load_subj_data(subj_id)

    ## Mark run onsets. 
    run_onsets = paf.get_run_onsets(messages)
    n_blocks, d = run_onsets.shape

    # Load feature map
    featmap_file_path = et_data_dir + str(subj_id) + 'featmap.mat'
    featmap_mat = io.loadmat(featmap_file_path, struct_as_record=False, squeeze_me=True)
    featmap = np.array(featmap_mat["features_aoi_map"])

    # Re-format events dataframe. 
    events_df = paf.rereference_events(subj_id, n_blocks, run_onsets, sfreq)

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
    fixations = fixations.rename(columns={'Unnamed: 0': 'WithinBlockCount'})
    # split the fixations df based on block to get a list of indices when the block number changes
    grouped = fixations.groupby(fixations.WithinBlockCount)
    df_getindexes = grouped.get_group(0)
    index_start_list = df_getindexes.index.values.tolist()
    index_start_list.append(fixations.shape[0])
    if index_start_list[0] != 0:
        index_start_list.insert(0, 0)
    # use list of indices to make a new column
    new_col = np.ones((fixations.shape[0])).astype(int)
    # subj 39, block 4 is empty
    if subj_id == 39:
        for block_num in np.arange(3):
            new_col[index_start_list[block_num]:index_start_list[block_num+1]] = new_col[index_start_list[block_num]:index_start_list[block_num+1]]*(block_num+1)
        for block_num in np.arange(4,n_blocks):
            new_col[index_start_list[block_num-1]:index_start_list[block_num]] = new_col[index_start_list[block_num-1]:index_start_list[block_num]]*(block_num+1)
    else: 
        for block_num in np.arange(n_blocks):
            new_col[index_start_list[block_num]:index_start_list[block_num+1]] = new_col[index_start_list[block_num]:index_start_list[block_num+1]]*(block_num+1)
    fixations['BlockNumber'] = new_col

    # grab fixations for block and trial
    fixations_block = fixations[fixations.BlockNumber == block]
    fixations_block_trial = fixations_block[fixations_block.Trial == (block-1)*n_trials_block + trial]

    ## Initialize indices and labels of plot.
    indices = np.zeros((xdim,ydim,n_screens))
    labels = ()

    ## Load all centers.
    all_centers = pd.read_csv(os.getcwd() + '/allCenters.csv')

    ## Subset this participant's centers. 
    sub = 'Sub' + str(subj_id) + '_'
    centers = all_centers[all_centers['Unnamed: 0'].str.contains(sub)]
    centers['Block'] = [int(s.replace(sub + 'block' + '_', "")) for s in list(centers['Unnamed: 0'].values)]

    # Grab block centers. 
    block_centers = centers[centers['Block'] == int(block)]
    for a in range(0, n_aois):
        col_names = ['aoi' + str(a) + '_x', 'aoi' + str(a) + '_y']
        this_aoi_centers = block_centers[col_names].iloc[0].values

        isfrac = lambda v: True if v < 1 and v > 0 else False
        xmin, xmax = [int(xdim * x) if isfrac(x) else int(x) for x in [this_aoi_centers[0]-aoisidelength//2, this_aoi_centers[0]+aoisidelength//2]]
        ymin, ymax = [int(ydim * y) if isfrac(y) else int(y) for y in [this_aoi_centers[1]-aoisidelength//2, this_aoi_centers[1]+aoisidelength//2]]
        
        indices[xmin:xmax,ymin:ymax,0] = indices.max() + 1
        
        values, curr_indices = np.unique(indices, return_inverse=True)

        if np.all(values): curr_indices += 1
        indices = curr_indices.reshape(xdim, ydim, n_screens)
        labels = tuple(range(1,int(indices.max())+1))

    return makeFixationPlot(this_trial_featmap, fixations_block_trial, block_centers, indices, labels, xdim, ydim)

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
    fig, ax = plotAOIFixations(sub_id, block_num, trial_num)
    plt.draw()
    plt.show()