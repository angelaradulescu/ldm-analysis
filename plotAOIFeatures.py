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

## Define global variables ##
et_data_dir = os.getcwd().strip('ldm-analysis') + 'ProcessedData/'# this expects ProcessedData to be one directory up from the analysis repo
image_dir = os.getcwd().strip('ldm-analysis') + 'FinalStimuli/ByNumber/'

## Define screen metadata.
xdim, ydim, n_screens = 1280, 1024, 1 
aoisidelength = 162
n_aois = 9


# import rereference events, get run onsets, and load subj data from preprocessing script
def rereference_events(subj_id, n_blocks, run_onsets, sfreq): 
    """The events file is a `n_trials ` x 3 array which defines: 
    (1) the run/block number 
    (2) the trial onset relative to the first stimulus onset in the run (in seconds) 
    (3) the trial offset (in seconds) 
    Using the sample frequency, it gets re-referenced below to the NivLink 0.2 format: 
    (1) the event onset (in sample indices)
    (2) start time before the event 
    (3) end time after the event """

    # Load file.
    events_file_path = et_data_dir + str(subj_id) + 'events.mat'
    events_mat = io.loadmat(events_file_path)
    events = np.array(events_mat["events_array"])
    
    # Record run onsets in events array. 
    new_col = events.sum(1)[...,None]*0
    events = np.append(events, new_col, 1)
    for b in np.arange(n_blocks):
        block = b+1
        this_block = events[:,0] == block
        events[this_block,3] = run_onsets[block-1,1].astype(int) 
        
    # Convert to dataframe.
    events_df = pd.DataFrame(events)
    events_df.columns = ['block','1', '2', '3']
    
    # Round events to sampling frequency.   
    events_df['1'] = np.floor(events_df['1'] * sfreq) / sfreq
    events_df['2'] = np.ceil(events_df['2'] * sfreq) / sfreq

    # Re-reference events.
    events_df['3'] += events_df['1'] * sfreq
    events_df['2'] -= events_df['1']
    events_df['1'] -= events_df['1']
    
    return events_df

def load_subj_data(subj_id):
    raw_data_dir = os.getcwd().strip('ldm-analysis') + 'RawData/' # this expects RawData to be one directory up from the analysis repo
    edf_path = raw_data_dir + 'Sub' + str(subj_id) + 'ET.edf'
    # Read subject's data from edf file.
    data = Raw(edf_path)
    # Filter out only eye position.
    raw_pos_data = data.data[:,0,(0,1)]
    # Grab messages for epoching step.
    messages = data.messages 
 
    return data, raw_pos_data, messages, data.info['sfreq']

def get_run_onsets(messages): 
    """ Returns run onsets. 
        
        This function is specific to how the experiment code handles
        messages to the EDF file during the task.

    Parameters
    ----------
    messages: array, shape (n_times, 1) 
        Array containing messages from the NivLink Raw object.

    Returns
    -------
    run_onsets : array, shape (n_runs, 2)
        Run IDs and start indices.

    """

    n_messages = len(messages)
    run_onsets = np.empty((0,3), dtype = int)

    for m in np.arange(n_messages):
        
        this_message_index = messages[m][0]
        this_message = messages[m][1]
        
        ## We encountered a new run.
        if 'Run' in this_message:
             
            ## Get index of first XDAT 2 in this run. 
            ## This is specific to the LDM dataset
            ## because the first stim onset is not align with the Start Run message.   
            first_xdat2_message_index = messages[m+12][0]
            first_xdat2_message = messages[m+12][1]
            
            ## Initialize onset array for this run.
            this_run_onsets = np.empty((1,3), dtype = int); 
            
            this_run_onsets[:,0] = int(this_message.strip('Run '))
            # this_run_onsets[:,1] = int(this_message_index) # actual run start message
            this_run_onsets[:,1] = int(first_xdat2_message_index)  # first XDAT2 in run
              
        ## Re-construct End Run message index by looking at the last trial in each run.
        ## Assumes we consistently had 40 trial per run as per LDM_Run4.m
        if 'Trial 40' in this_message:
            
            last_xdat1_message_index = messages[m+6][0]
            last_xdat1_message = messages[m+6][1]
            
            this_run_onsets[:,2] = int(last_xdat1_message_index)
            run_onsets = np.vstack((run_onsets,this_run_onsets))
            
    return run_onsets

def plotAOIFeatures(subj_id, block, trial):
    ## Load data.
    data, raw_pos_data, messages, sfreq = load_subj_data(subj_id)

    ## Mark run onsets. 
    run_onsets = get_run_onsets(messages)
    n_blocks, d = run_onsets.shape

    # Load feature map
    featmap_file_path = et_data_dir + str(subj_id) + 'featmap.mat'
    featmap_mat = io.loadmat(featmap_file_path, struct_as_record=False, squeeze_me=True)
    featmap = np.array(featmap_mat["features_aoi_map"])

    # Re-format events dataframe. 
    events_df = rereference_events(subj_id, n_blocks, run_onsets, sfreq)

    # Load fixations for sub
    fixations = pd.read_csv(et_data_dir + str(subj_id) + 'fixations.csv')

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

    return makeFeaturePlot(this_trial_featmap, block_centers, indices, labels, xdim, ydim)

def makeFeaturePlot(trial_featmap, block_centers, indices, labels, xdim, ydim, height=3, ticks=False, cmap=None):
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

    for a in range(0, n_aois):
        feat_num = int(trial_featmap[a])
        arr_a = mpimg.imread(image_dir + str(feat_num) + '.jpg')
        imagebox = OffsetImage(arr_a, zoom=0.1)
        col_names = ['aoi' + str(a) + '_x', 'aoi' + str(a) + '_y']
        this_aoi_centers = block_centers[col_names].iloc[0].values

        ab = AnnotationBbox(imagebox, (this_aoi_centers[0], this_aoi_centers[1]))
        ax.add_artist(ab)

    ## Plotting.
    cbar = ax.imshow(indices[:,:,-1].T, cmap=cmap, aspect='auto', vmin=0, vmax=len(labels))
    fig.colorbar(cbar, cax, ticks=np.arange(len(cmap.colors)))
    if not ticks: ax.set(xticks=[], yticks=[])        

    return fig, ax

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
    fig, ax = plotAOIFeatures(sub_id, block_num, trial_num)
    plt.draw()
    plt.show()