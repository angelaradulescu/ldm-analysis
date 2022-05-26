# Library of functions for handling data from participant behavior when learning
# in a multidimensional environment with discrete features. 

import numpy as np

class Data(object):
    """ Container for data and methods.

    Parameters
    ----------
    behav_data: Pandas dataframe 
        Behavioral data for one participant.
    ----------
    """

    def __init__(self, behav_data):
    
        ## Define data.
        self.behav_data = behav_data
    
        ## Get other variables 
        self.n_trials = max(behav_data['Trial'])
        self.n_games = max(behav_data['Game'])
        self.game_length = len(behav_data.loc[(behav_data['Game'] == 1)])

        ## Add trial-within-game variable.
        self.behav_data['Trial_2'] = self.behav_data['Trial'] - (self.behav_data['Game']-1)*self.game_length

    def split_data(self, test_game):
        """ Splits behavioral data into training data (n-1 games) and test data (1 game).
        """

        ## Behavioral data.
        behav_training_data = self.behav_data.loc[self.behav_data['Game'] != test_game]
        behav_test_data = self.behav_data.loc[self.behav_data['Game'] == test_game]
        
        return behav_training_data, behav_test_data

def extract_vars(behav_data, trials):
    """ Helper function that extracts variables from one game given trial indices. 
    """

    ## Get observations for this game (available stimuli, choices, outcomes, center dimension and feature).
    stimuli_1 = behav_data.loc[behav_data['Trial'].isin(trials)][['Stim11','Stim12','Stim13']].values
    stimuli_2 = behav_data.loc[behav_data['Trial'].isin(trials)][['Stim21','Stim22','Stim23']].values
    stimuli_3 = behav_data.loc[behav_data['Trial'].isin(trials)][['Stim31','Stim32','Stim33']].values  
    choices = behav_data.loc[behav_data['Trial'].isin(trials)][['Chosen1','Chosen2','Chosen3']].values
    outcomes = behav_data.loc[behav_data['Trial'].isin(trials)]['Outcome'].values
    center_dim = behav_data.loc[behav_data['Trial'].isin(trials)]['CenterDim'].values
    center_feat = behav_data.loc[behav_data['Trial'].isin(trials)]['CenterFeat'].values
    center = np.vstack((center_dim,center_feat)).T
    missed_trials = np.isnan(outcomes)

    ## Mark target.
    target = behav_data['Feat'].iloc[0]

    ## Mark whether game was learned. 
    point_of_learning = behav_data.loc[behav_data['Trial'].isin(trials)]['PoL'].values[0]
    if point_of_learning < 16: learned = 1
    else: learned = 0 
 
    ## Mark chosen action. 
    chose_1 = np.prod(choices == stimuli_1, axis=1)
    chose_2 = np.prod(choices == stimuli_2, axis=1)
    chose_3 = np.prod(choices == stimuli_3, axis=1)
    # actions = np.vstack((chose_1, chose_2, chose_3)).T.nonzero()[1]+1
    actions = np.ones(len(trials))*np.nan
    actions[np.vstack((chose_1, chose_2, chose_3)).T.nonzero()[0]] = np.vstack((chose_1, chose_2, chose_3)).T.nonzero()[1] + 1

    ## Remove missed trials.
    if np.sum(np.isnan(outcomes)):
        nan_idx = np.argwhere(np.isnan(outcomes)).flatten()
    else:
        nan_idx = []
   
    stimuli_1 = np.delete(stimuli_1, nan_idx, axis=0)
    stimuli_2 = np.delete(stimuli_2, nan_idx, axis=0)
    stimuli_3 = np.delete(stimuli_3, nan_idx, axis=0)
    choices = np.delete(choices, nan_idx, axis=0)
    outcomes = np.delete(outcomes, nan_idx, axis=0)   
    center = np.delete(center, nan_idx, axis=0)
    actions = np.delete(actions, nan_idx, axis=0)
    
    ## Create dictionary.
    extracted_data = {
        "stimuli_1": stimuli_1,
        "stimuli_2": stimuli_2,   
        "stimuli_3": stimuli_3,
        "choices": choices,
        "actions": actions,
        "outcomes": outcomes,
        "center": center,
        "learned_game": learned,
        "target": target
    }

    return extracted_data

