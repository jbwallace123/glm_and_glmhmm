#qUtils.py

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
import ssm

def calc_sem(x):

    return np.std(x) / np.sqrt(len(x))


def calc_ci(x, confidence: float = 0.95):

    from scipy.stats import norm

    # determine critical value for set confidence interval.
    alpha_2 = (1 - confidence) / 2
    critical_value = norm.ppf(1 - alpha_2)

    sem = calc_sem(x)
    err_width = critical_value * sem
    ci_low = np.mean(x) - err_width
    ci_high = np.mean(x) + err_width
    return (ci_low, ci_high, err_width)
 
def make_onehot_array(x):

    print(len(np.unique(x)))
    if len(np.unique(x)) == 1:
        return x

    onehot = np.zeros((x.size, x.max() + 1))
    onehot[np.arange(x.size), x] = 1
    return onehot

def compare_k_states(scores, num_states, multi_animal = False):
    #plot train and test scores for each model and plot confidence intervals
    datasets =['train', 'test']

    if multi_animal == False:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
        for key in datasets:
            plt.scatter(num_states, [np.mean(s) for s in scores[key]],
                        label=key)
            plt.errorbar(num_states, [np.mean(s) for s in scores[key]],
                         [calc_ci(s)[2] for s in scores[key]], alpha=0.3)
            plt.legend(bbox_to_anchor=(1, 1))

        ax.set(xlabel="State number", ylabel="Log Probability",
            title="Cross Validation Scores with 95% CIs")
            
    elif multi_animal == True: 
        grouped_data = {}

        for dataset in datasets:
            for item in scores[dataset]:
                mouse = item['mouse']
                if mouse not in grouped_data:
                    grouped_data[mouse] = {'train': [], 'test': []}
                grouped_data[mouse][dataset].append({'mouse': mouse, 'score':item['score']})
                
        for mouse in grouped_data:
            fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
            for i, key in enumerate(datasets):
                plt.scatter(num_states, [np.mean(s['score']) for s in grouped_data[mouse][key]],
                            label=key)
                plt.errorbar(num_states, [np.mean(s['score']) for s in grouped_data[mouse][key]],
                                [calc_ci(s['score'])[2] for s in grouped_data[mouse][key]], alpha=0.3)
            plt.legend(bbox_to_anchor=(1, 1))
            ax.set(xlabel="State number", ylabel="Log Probability",
                title="Cross Validation Scores with 95% CIs for  {}".format(mouse))
            

def predict_state(train_x, test_x, train_y, test_y, model_list, multi_animal = False):
    train_states = []
    test_states = []
    models = []

    if multi_animal == False:
        #Grab models from model list
        for i in range(len(model_list)):
            hmm = model_list[i]['model_list'][0]
            models.append(hmm)

        for i in range(len(models)):
            train_states.append({'num_states': i+1 , 'expected_states': ([models[i].expected_states(y, X)[0]
                                for y, X in zip([train_y[i]], [train_x[i]])])})
            test_states.append({'num_states': i+1, 'expected_states': ([models[i].expected_states(y, X)[0]
                                for y, X in zip([test_y[i]], [test_x[i]])])})
    elif multi_animal == True:
        animal_hmm = []
        for i in range(len(model_list)):
            for item in model_list[i]['model_list']:
                num_states = model_list[i]['model_params']['num_states'][0]
                animal_hmm.append({'mouse': item['mouse'], 'num_states': num_states, 'hmm': item['glmhmm']})
                
        for i in range(len(animal_hmm)):
            expected_train_states = [animal_hmm[i]['hmm'].expected_states(y, X)[0]
                                for y, X in zip([train_y[i]['data']], [train_x[i]['data']])]
            train_states.append({'mouse': animal_hmm[i]['mouse'], 'num_states': animal_hmm[i]['num_states'],
                                 'expected_states': expected_train_states})
        for i in range(len(animal_hmm)):
            expected_test_states = [animal_hmm[i]['hmm'].expected_states(y, X)[0]
                                for y, X in zip([test_y[i]['data']], [test_x[i]['data']])]
            test_states.append({'mouse': animal_hmm[i]['mouse'], 'num_states': animal_hmm[i]['num_states'],
                                'expected_states': expected_test_states})          
    return train_states, test_states

def predict_choice(models, test_states, test_x, test_y, accuracy=True, verbose=False,
                    multi_animal = False):

    pchoice = []
    acc = []
    if multi_animal == False:
        for i in range(len(models)):
            model = models[i]['model_list'][0]
            expectations = test_states[i]['expected_states']
            glm_weights = -model.observations.params
            permutation = np.argsort(glm_weights[:, 0, 0])
            masks = [np.ones_like(data, dtype=bool) for data in [test_y[i]]]
            # Convert this now to one array:
            posterior_probs = np.concatenate(expectations, axis=0)
            posterior_probs = posterior_probs[:, permutation]
            prob_right = [
                np.exp(model.observations.calculate_logits(input=test_x[i]))
                for data, input, train_mask in zip(test_y, test_x, masks)
            ]
            prob_right = np.concatenate(prob_right, axis=0)
            # Now multiply posterior probs and prob_right:
            prob_right = prob_right[:, :, 1] ## taking logits running an exponential then arranging in a 3d array
            # Now multiply posterior probs and prob_right and sum over latent axis:
            final_prob_right = np.sum(np.multiply(posterior_probs, prob_right), axis=1)
            # Get the predicted label for each time step:
            pred_choice = np.around(final_prob_right, decimals=0).astype('int')
            pchoice.append(pred_choice) 
            if accuracy:
                pred_accuracy = np.mean(test_y[i][:, 0] == pred_choice)
                if verbose:
                    print(f'Model with {i} state(s) has a test predictive accuracy of {pred_accuracy}')
                acc.append(pred_accuracy)
        if accuracy:
            return acc
    elif multi_animal == True:
        animal_hmm = []
        animal_hmm = []
        for i in range(len(models)):
            for item in models[i]['model_list']:
                num_states = models[i]['model_params']['num_states'][0]
                animal_hmm.append({'mouse': item['mouse'], 'num_states': num_states, 'hmm': item['glmhmm']})

        for i in range(len(animal_hmm)):
            mouse = animal_hmm[i]['mouse']
            num_states = animal_hmm[i]['num_states']
            model = animal_hmm[i]['hmm']
            expectations = test_states[i]['expected_states']
            glm_weights = -model.observations.params
            permutation = np.argsort(glm_weights[:, 0, 0])
            masks = [np.ones_like(data['data'], dtype=bool) for data in [test_y[i]]]
            # Convert this now to one array:
            posterior_probs = np.concatenate(expectations, axis=0)
            posterior_probs = posterior_probs[:, permutation]
            prob_right = [
                np.exp(model.observations.calculate_logits(input=test_x[i]['data']))
                for data, input, train_mask in zip(test_y[i], test_x, masks)
            ]
            prob_right = np.concatenate(prob_right, axis=0)
            # Now multiply posterior probs and prob_right:
            prob_right = prob_right[:, :, 1]
            # Now multiply posterior probs and prob_right and sum over latent axis:
            final_prob_right = np.sum(np.multiply(posterior_probs, prob_right), axis=1)
            # Get the predicted label for each time step:
            pred_choice = np.around(final_prob_right, decimals=0).astype('int')
            pchoice.append(pred_choice)
            if accuracy:
                pred_accuracy = np.mean(test_y[i]['data'][:, 0] == pred_choice)
                if verbose:
                    print(f'Animal Model {mouse} with {num_states} state(s) has a test predictive accuracy of {pred_accuracy}')
                acc.append(pred_accuracy)
        if accuracy:
            return acc

def pred_occupancy(self):

        train_max_prob_state = []
        test_max_prob_state = []
        train_occupancy = []
        test_occupancy = []
        train_occupancy_rates = []
        test_occupancy_rates = []
        for i in model:
            state_max_posterior = [np.argmax(posterior, axis=1) for posterior in self.train_states[i]]

            state_occupancies = np.zeros((i+1, len(self.train_states[i])))
            for idx_sess, max_post in enumerate(state_max_posterior):
                idx, count = np.unique(max_post, return_counts=True)
                state_occupancies[idx, idx_sess] = count.astype('float')

            state_occupancies = state_occupancies.sum(axis=1) / state_occupancies.sum()
            self.train_max_prob_state.append(state_max_posterior)
            self.train_occupancy.append([make_onehot_array(max_post) for max_post in state_max_posterior])
            self.train_occupancy_rates.append(state_occupancies)
    
            state_max_posterior = [np.argmax(posterior, axis=1) for posterior in self.test_states[i]]
            state_occupancies = np.zeros((i+1, len(self.test_states[i])))
            for idx_sess, max_post in enumerate(state_max_posterior):
                idx, count = np.unique(max_post, return_counts=True)
                state_occupancies[idx, idx_sess] = count.astype('float')

            state_occupancies = state_occupancies.sum(axis=1) / state_occupancies.sum()
            self.test_max_prob_state.append(state_max_posterior)
            self.test_occupancy.append([make_onehot_array(max_post) for max_post in state_max_posterior])
            self.test_occupancy_rates.append(state_occupancies)


def plot_state_probs(self, model_idx, sess_idx: int = 0,
                        as_occupancy: bool = False):

    if as_occupancy:
        samples = self.test_occupancy[model_idx][sess_idx]
    else:
        samples = self.test_states[model_idx][sess_idx]

    fig, ax = plt.subplots(figsize=(6, 3))
    for i in range(model_idx + 1):
        plt.plot(samples[:, i], label=i, alpha=0.8)
    ax.set(xlabel='trial', ylabel='prob')
    plt.legend(bbox_to_anchor=(1, 1), title='latent state')
    sns.despine()


def plot_tmat(self, model_idx, ax=None):

    tmat = np.exp(self.model[model_idx].transitions.params)[0]
    plot_states = self.num_states[:model_idx+1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=80)
    ax.imshow(tmat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(tmat.shape[0]):
        for j in range(tmat.shape[1]):
            _ = plt.text(j, i, str(np.around(tmat[i, j], decimals=2)),
                            ha="center", va="center", color="k", fontsize=10)
    ax.set(xlabel='state t+1', xlim=(-0.5, len(plot_states) - 0.5),
            ylabel='state t', ylim=(len(plot_states) - 0.5, -0.5),
            title='Transition matrix')
    plt.xticks(plot_states - 1, plot_states, fontsize=10)
    plt.yticks(plot_states - 1, plot_states, fontsize=10)
    plt.tight_layout()
