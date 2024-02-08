#hmmUtils.py

"""
@author: celiaberon, jbwallace123

"""

import os
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import random

def list_to_str(seq):
    
    '''take list of ints/floats and convert to string'''
    
    seq = [str(el) for el in seq] # convert element of sequence to string
    
    return ''.join(seq) # flatten list to single string

def encode_as_ab(row, symm):
    
    '''
    converts choice/outcome history to character code where where letter represents choice and case outcome
    INPUTS:
        - row: row from pandas DataFrame containing named variables 'decision_seq' and 'reward_seq' (previous N decisions/rewards) 
        - symm (boolean): if True, symmetrical encoding with A/B for direction (A=first choice in sequence)
                          if False, R/L encoding right/left choice
    OUTPUTS:
        - (string): string of len(decision_seq) trials encoding each choice/outcome combination per trial
    
    '''
    
    if int(row.decision_seq[0]) & symm: # symmetrical mapping based on first choice in sequence 1 --> A
        mapping = {('0','0'): 'b', ('0','1'): 'B', ('1','0'): 'a', ('1','1'): 'A'} 
    elif (int(row.decision_seq[0])==0) & symm: # symmetrical mapping for first choice 0 --> A    
        mapping = {('0','0'): 'a', ('0','1'): 'A', ('1','0'): 'b', ('1','1'): 'B'} 
    else: # raw right/left mapping (not symmetrical)
        mapping = {('0','0'): 'r', ('0','1'): 'R', ('1','0'): 'l', ('1','1'): 'L'} 

    return ''.join([mapping[(c,r)] for c,r in zip(row.decision_seq, row.reward_seq)])

def add_history_cols(df, N):
    
    '''
    INPUTS:
        - df (pandas DataFrame): behavior dataset
        - N (int): number trials prior to to previous trial to sequence (history_length)
        
    OUTPUTS:
        - df (pandas DataFrame): add columns:
            - 'decision_seq': each row contains string of previous decisions t-N, t-N+1,..., t-1
            - 'reward_seq': as in decision_seq, for reward history
            - 'history': encoded choice/outcome combination (symmetrical)
            - 'RL_history': encoded choice/outcome combination (raw right/left directionality)
       
    '''
    from numpy.lib.stride_tricks import sliding_window_view
    
    df['decision_seq']=np.nan # initialize column for decision history (current trial excluded)
    df['reward_seq']=np.nan # initialize column for laser stim history (current trial excluded)

    df = df.reset_index(drop=True) # need unique row indices (likely no change)

    for session in df.Session.unique(): # go by session to keep boundaries clean

        d = df.loc[df.Session == session] # temporary subset of dataset for session
        df.loc[d.index.values[N:], 'decision_seq'] = \
                                    list(map(list_to_str, sliding_window_view(d.Decision.astype('int'), N)))[:-1]

        df.loc[d.index.values[N:], 'reward_seq'] = \
                                    list(map(list_to_str, sliding_window_view(d.Reward.astype('int'), N)))[:-1]

        df.loc[d.index.values[N:], 'history'] = \
                                    df.loc[d.index.values[N:]].apply(encode_as_ab, args=([True]), axis=1)

        df.loc[d.index.values[N:], 'RL_history'] = \
                                    df.loc[d.index.values[N:]].apply(encode_as_ab, args=([False]), axis=1)
        
    return df

def calc_conditional_probs(df, symm, action=['Switch'], run=0):

    '''
    calculate probabilities of behavior conditional on unique history combinations
    
    Inputs:
        df (pandas DataFrame): behavior dataset
        symm (boolean): use symmetrical history (True) or raw right/left history (False)
        action (string): behavior for which to compute conditional probabilities (should be column name in df)
        
    OUTPUTS:
        conditional_probs (pandas DataFrame): P(action | history) and binomial error, each row for given history sequence
    '''

    group = 'history' if symm else 'RL_history' # define columns for groupby function

    max_runs = len(action) - 1 # run recursively to build df that contains summary for all actions listed

    conditional_probs = df.groupby(group).agg(
        paction=pd.NamedAgg(action[run], np.mean),
        n = pd.NamedAgg(action[run], len),
    ).reset_index()
    conditional_probs[f'p{action[run].lower()}_err'] = np.sqrt((conditional_probs.paction * (1 - conditional_probs.paction))
                                                  / conditional_probs.n) # binomial error
    conditional_probs.rename(columns={'paction': f'p{action[run].lower()}'}, inplace=True) # specific column name
    
    if not symm:
        conditional_probs.rename(columns={'RL_history':'history'}, inplace=True) # consistent naming for history
    
    if max_runs == run:
    
        return conditional_probs
    
    else:
        run += 1
        return pd.merge(calc_conditional_probs(df, symm, action, run), conditional_probs.drop(columns='n'), on='history')

def sort_cprobs(conditional_probs, sorted_histories):
    
    '''
    sort conditional probs by reference order for history sequences to use for plotting/comparison
    
    INPUTS:
        - conditional_probs (pandas DataFrame): from calc_conditional_probs
        - sorted_histories (list): ordered history sequences from reference conditional_probs dataframe
    OUTPUTS:
        - (pandas DataFrame): conditional_probs sorted by reference history order
    '''
    
    from pandas.api.types import CategoricalDtype
    
    cat_history_order = CategoricalDtype(sorted_histories, ordered=True) # make reference history ordinal
    
    conditional_probs['history'] = conditional_probs['history'].astype(cat_history_order) # apply reference ordinal values to new df
    
    return conditional_probs.sort_values('history') # sort by reference ordinal values for history

def create_animal_datalist(train_data, test_data, x_cols):
        
        '''
        INPUTS:
            - train_data: list of training data for each animal
            - test_data: list of testing data for each animal
            - x_cols: list of column names for input data
        OUTPUTS:
            - train_data_x: list of training data for each animal
            - test_data_x: list of testing data for each animal
            - train_choices: list of training choices for each animal
            - test_choices: list of testing choices for each animal
            - train_data_trials: list of training data trials for each animal
            - test_data_trials: list of testing data trials for each animal
        '''

        train_data_x = []
        test_data_x = []
        train_choices = []
        test_choices = []
        train_data_trials = []
        test_data_trials = []

        for mouse_index, mouse_data in enumerate(train_data):
            data_x = mouse_data['mouse']['data'] 
            data_x = data_x[x_cols].values
            choices = mouse_data['mouse']['data']['Decision'].to_numpy().reshape(-1, 1).astype(int)
            train_data_trials.append({'mouse': mouse_data['mouse']['mouse'], 'sessions': len(data_x)})
            train_data_x.append({'mouse': mouse_data['mouse']['mouse'], 'data': data_x})
            train_choices.append({'mouse': mouse_data['mouse']['mouse'], 'choices': choices})


        for mouse_index, mouse_data in enumerate(test_data):
            data_x = (mouse_data['mouse']['data'])
            data_x = data_x[x_cols].values
            choices = mouse_data['mouse']['data']['Decision'].to_numpy().reshape(-1, 1).astype(int)
            test_data_trials.append({'mouse': mouse_data['mouse']['mouse'], 'sessions': len(data_x)})
            test_data_x.append({'mouse': mouse_data['mouse']['mouse'], 'data': data_x})
            test_choices.append({'mouse': mouse_data['mouse']['mouse'], 'choices': test_choices})
            
            return train_data_x, test_data_x, train_choices, test_choices, train_data_trials, test_data_trials


def animal_fit(train_x, y, num_states, obs_dim, observations, num_categories, prior_sigma, transitions,
               prior_alpha, iters):
    
    '''
    Fit GLM HMM to each animals' behavior
    - train_x = list of x data for each animal. Expected to be [{'mouse': MOUSEID, 'data': train_x}, ...]
    - y = list of y data for each animal. Expected to be [{'mouse': MOUSEID, 'choices': y}, ...]
    - num_states = number of hidden states
    - obs_dim = number of observation dimensions
    - observations = expected to be a string "input_driven_obs", "bernoulli", etc.
    - num_categories = number of categories for the observations
    - prior_sigma = prior sigma for the observations
    - transitions = expected to be a string "standard", "sticky", etc.
    - prior_alpha = prior alpha for the transitions
    - iters = number of iterations for the fit

    Returns:
    - list of fitted models for each animal
    - list of test log likelihoods for each animal
    - list of train scores for CV for each animal
    - list of test scores for CV for each animal

    '''
    model_list = []
    ll_list = []
    train_scores_list = []
    test_scores_list = []

    import ssm
    for i, mouse in enumerate(train_x):
        print(f'Fitting model for mouse {i+1}/{len(train_x)}...')

        if type(train_x) == list:
            inpts = train_x[i]['data']
            input_dim = inpts.shape[1]
        else: 
            inpts = train_x.to_numpy()
            input_dim = inpts.shape[1]

        choices = y[i]['choices']
        for k in range(len(num_states)):
            from ssm import model_selection
            glmhmm = ssm.HMM(num_states[k], obs_dim, input_dim, observations=observations,
                             observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma), 
                             transitions=transitions, prior_alpha=prior_alpha)
            train_scores, test_scores = ssm.model_selection.cross_val_scores(glmhmm, choices, inpts, heldout_frac=0.1, n_repeats=5, verbose=True)
            N_iters = iters
            ll = glmhmm.fit(choices, inputs=inpts, method="em", num_iters=N_iters, initialize=False)
            ll_list.append({'mouse': train_x[i]['mouse'], 'll': ll})
            train_scores_list.append({'mouse': train_x[i]['mouse'], 'scores': train_scores})
            test_scores_list.append({'mouse': train_x[i]['mouse'], 'scores': test_scores})
            model_list.append({'mouse': train_x[i]['mouse'], 'glmhmm': glmhmm})

    return model_list, ll_list, train_scores_list, test_scores_list

def global_fit(train_x, y, num_states, obs_dim, observations, num_categories, prior_sigma, transitions,
               prior_alpha, iters):
    
    '''
    Fit GLM HMM to all animals' behavior
    - train_x = list of x data for all animals. 
    - y = list of y data for each animal. 
    - num_states = number of hidden states
    - obs_dim = number of observation dimensions
    - observations = expected to be a string "input_driven_obs", "bernoulli", etc.
    - num_categories = number of categories for the observations
    - prior_sigma = prior sigma for the observations
    - transitions = expected to be a string "standard", "sticky", etc.
    - prior_alpha = prior alpha for the transitions
    - iters = number of iterations for the fit

    Returns:
    - fitted model
    - test log likelihood
    - train scores for CV
    - test scores for CV

    '''

    global_model_list = []
    global_ll_list = []
    global_train_scores = []
    global_test_scores = []

    if type(train_x) == list:
        inpts = train_x
        input_dim = inpts[0].shape[1]
    else: 
        inpts = train_x.to_numpy()
        input_dim = inpts.shape[1]

    import ssm
    for i in range(len(num_states)):
        from ssm import model_selection
        glmhmm = ssm.HMM(num_states[i], obs_dim, input_dim, observations=observations,
                         observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma), 
                         transitions=transitions, prior_alpha=prior_alpha)
        train_scores, test_scores = ssm.model_selection.cross_val_scores(glmhmm, y, inpts, heldout_frac=0.1, n_repeats=5, verbose=True)
        N_iters = iters
        ll = glmhmm.fit(y, inputs=inpts, method="em", num_iters=N_iters, initialize=False)
        global_ll_list.append(ll)
        global_train_scores.append(train_scores)
        global_test_scores.append(test_scores)
        global_model_list.append(glmhmm)

    return global_model_list, global_ll_list, global_train_scores, global_test_scores
 

            