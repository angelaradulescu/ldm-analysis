# Feature reinforcement learning class. 

# Instantiates a feature reinforcement learning agent that is learing
# in a multidimensional environment with discrete features.

# Available methods: 
# Likelihood computation for: 
#       o Choice data

import numpy as np
from scipy.special import logsumexp
from scipy.stats import dirichlet
import warnings

# Custom dependencies
import sys
import os
sys.path.append(os.getcwd()) 
from World import World
from Data import extract_vars

# Start Agent class.
class Agent(object):
    """ Container for agent properties and methods.

    Parameters.
    ----------
    world: 
        instance of World.
    eta: float
        Learning rate.
    eta_k: float
        Decay rate.
    beta: float
        Softmax temperature for choice linking function.
    w_init: float
        Initial feature values.
    decay_target: float
        Value to decay feature weights towards. 
    precision: float
        Precision for Dirichlet attention linking function. 
    ----------
    """

    ###############################
    ## Initialize agent properties.
    ###############################
    def __init__(self, world, params):
        """ Sets agent parameters.
        """

        self.eta = params['learning_rate']
        self.eta_k = params['decay_rate']
        self.beta = params['softmax_temperature']
        self.w_init = params['w_init']
        self.dt = params['decay_target']
        self.precision = params['precision']

    def softmax(self, v):
        """ Softmax action selection for an arbitrary number of actions with values v.
            Ref. on logsumexp: https://blog.feedly.com/tricks-of-the-trade-logsumexp/

            Parameters
            ----------
            v: array, float
                Array of action values.

            Returns
            -------
            p_c: array, float bounded between 0 and 1
                Probability distribution over actions
            a: int 
                Chosen action.
        """

        v_b = self.beta * v;
        p_c = np.exp(v_b - logsumexp(v_b));

        ## Uniformly sample from cumulative distribution over p_c.
        a = np.nonzero(np.random.random((1,)) <= np.cumsum(p_c))[0][0] + 1

        return p_c, a

    ######################
    ## Simulation
    ######################


    ########################
    ## Likelihood function
    ########################

    def choice_likelihood(self, world, extracted_data):
   
        """ Returns the log likelihood of a sequence of choices. 
            
            Parameters
            ----------
            world: instance of World.

            extracted_data: dictionary of extracted variables. 
            
            Contains: 

            stimuli_1, stimuli_2, stimuli_3: int, shape(n_trials, n_dims)
                Each available stimulus, expanded coding as defined in World.make_stimuli. 

            choices: int, shape(n_trials, n_dims)
                Sequence of chosen stimuli, expanded feature coding as defined in World.make_stimuli 

            actions: int, shape(n_trials, 1)
                Sequence of chosen actions.

            outcomes: int, shape(n_trials, 1)
                Sequence of outcomes. 

            center: int, shape(n_trials,2)
                Center dimension and center feature.

            Returns
            -------
            w_all: float, array(n_trials, n_feats)
                Learned feature weights.

            log_lik: float
                Log-likelihood of choices.
        """

        ## Remap dictionary to necessary local variables. 
        outcomes = extracted_data["outcomes"]
        stimuli_1 = extracted_data["stimuli_1"]
        stimuli_2 = extracted_data["stimuli_2"]
        stimuli_3 = extracted_data["stimuli_3"]
        choices = extracted_data["choices"]
        actions = extracted_data["actions"]
       
        ## Get number of trials.
        n_trials = len(outcomes)

        ## Preallocate value array.
        w_all = np.ones((n_trials, world.n_feats)) * np.nan
        
        ## Initialize feature weights.
        W = self.w_init * np.ones(world.n_feats)

        ## Initialize likelihood.
        log_lik = 0

        ## Loop through trials. 
        for t in np.arange(n_trials):

            ## Store current W.
            w_all[t,:] = W

            ## Grab stimuli. 
            stimulus_1 = stimuli_1[t,:].astype(int)
            stimulus_2 = stimuli_2[t,:].astype(int)
            stimulus_3 = stimuli_3[t,:].astype(int)

            ## Compute current value. 
            V = np.full(world.n_dims, np.nan)
            V[0] = np.sum(W[stimulus_1-1])
            V[1] = np.sum(W[stimulus_2-1])
            V[2] = np.sum(W[stimulus_3-1])
      
            ## Compute action likelihood.
            p_c, a = self.softmax(V)
            log_p_c = np.log(p_c)
            trial_lik = log_p_c[actions[t].astype(int)-1]
            log_lik = log_lik + trial_lik

            ## Observe outcome. 
            outcome = outcomes[t].astype(int)

            ## Grab current choice. 
            choice = choices[t].astype(int)

            ## Update chosen weights.
            pe = outcome-np.sum(W[choice-1])
            W[choice-1] = W[choice-1] + self.eta * pe

            ## Decay unchosen weights.
            all_feats = np.arange(world.n_feats)+1
            unchosen_feats = all_feats[~np.isin(all_feats, choice)]

            W[unchosen_feats-1] = (1-self.eta_k) * W[unchosen_feats-1]

        return w_all, log_lik

######################
## Training functions.
######################

def train_frl_choice(training_params, behav_training_data):
    
    """ Trains model on choice data. 
    """

    ## Set world properties. 
    world = World(3, 3, 0, 0.75, 0.25, 1)

    ## Initialize likelihood.
    Lik = 0
    
    ## Get indices of training games.
    training_games_idxs = behav_training_data.Game.unique()

    ## Get number of training games.
    n_training_games = len(behav_training_data.Game.unique())

    ## Set parameters.
    # Default values set to 0.
    params = {'learning_rate': training_params[0],
              'decay_rate': training_params[1],
              'softmax_temperature': training_params[2],
              'w_init': 0,
              'decay_target': 0,
              'precision': 0}

    ## Instantiate agent.
    frl = Agent(world, params)
    
    ## Loop over training games.
    for g in np.arange(n_training_games-1):
        
        ## Subselect game trials and format data.
        trials = behav_training_data.loc[behav_training_data['Game'] == training_games_idxs[g]]['Trial'].values   
        extracted_data = extract_vars(behav_training_data, trials)

        ## Run model to obtain likelihood.
        W, lik = frl.choice_likelihood(world, extracted_data)
        
        Lik = Lik + lik
    
    print("total training set log likelihood:", Lik)
    
    return -Lik

# End Agent class.
