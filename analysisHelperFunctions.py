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
et_data_dir = os.getcwd() + '/ProcessedData/' # this expects ProcessedData to be one directory up from the analysis repo
image_dir = os.getcwd().strip('ldm-analysis') + 'FinalStimuli/ByNumber/' # this expects FinalStimuli to be one directory up from the analysis repo
raw_data_dir = os.getcwd().strip('ldm-analysis') + 'RawData/' # this expects RawData to be one directory up from the analysis repo

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

def loadFixations(fixations, subj_id, n_blocks=10):
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

    # make new column containing block number
    # subj 39, block 4 is empty -- handle this
    if subj_id == 39:
        for block_num in np.arange(3):
            new_col[index_start_list[block_num]:index_start_list[block_num+1]] = new_col[index_start_list[block_num]:index_start_list[block_num+1]]*(block_num+1)
        for block_num in np.arange(4,n_blocks):
            new_col[index_start_list[block_num-1]:index_start_list[block_num]] = new_col[index_start_list[block_num-1]:index_start_list[block_num]]*(block_num+1)
    else: 
        for block_num in np.arange(n_blocks):
            new_col[index_start_list[block_num]:index_start_list[block_num+1]] = new_col[index_start_list[block_num]:index_start_list[block_num+1]]*(block_num+1)
    
    # add block number to fixations dataframe
    fixations['BlockNumber'] = new_col
    return fixations

def getAgeMap(csvfile):
    return pd.read_csv(csvfile, index_col='Subj_id').Age