#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:42:38 2020

@author: viniciussaurin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_gaussian_at(support, sd=1.0, height=1.0, 
        xpos=0.0, ypos=0.0, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    gaussian = (1/(np.sqrt(2*np.pi*sd**2)))*np.exp((-1/2)*((support-support.mean())/sd)**2)
    return ax.plot(gaussian + xpos, support, **kwargs)

def choosing_explore(epsilon, est_action_value):
    """
    The function that let the agent explore sometimes accordingly to the chosen epsilon
    """
    
    if np.random.uniform(0,1,1) >= epsilon:
        maximum = max(est_action_value)
        qtd_max = np.sum(est_action_value==maximum)
        if qtd_max == 1:
            choice = np.argmax(est_action_value)
        else:
            maxs = np.array([index for index, value in enumerate(est_action_value) if value == maximum])
            choice = np.random.choice(maxs)
    else:
        choice = np.random.choice(range(len(est_action_value)))
    return choice

def sumprod(const, var, n):
    soma = 0
    for i in range(1,n+1):
        soma += const*((1-const)**(n-i)) * var[i-1]
    return soma


def wa_nonstationary(alpha, initial_value, ret, n):
    """
    Computes the weighted average of past rewards and the initial estimate Q1
    """
    update = ((1-alpha)**n)*initial_value + sumprod(alpha, ret, n)
    return update

def stationary_func(action_values, mean=0.0, sd=0.1, stationary=True, **kwargs):
    if stationary == True:
        return action_values
    else:
        action_values += np.random.normal(mean,sd,len(action_values))
        return action_values
    
def greedy_bothmethods(iv, t, action_values, k=10, qt = 1, epsilon=0.0, alpha = None, stationary = True, **kwargs):
    """
    Function that replicates the epsilon-greedy method with option to modify for nonstationary problems
    """
    if iv.shape != action_values.shape:
        raise ValueError('iv and action_values must have the same shape')
        
    ret_total = pd.DataFrame()
    opt_act_tot = pd.DataFrame()
    ret_total_nstat = pd.DataFrame()
    opt_act_tot_nstat = pd.DataFrame()
    
    for i in range(qt):
        print(f'Amostra {i+1} de {qt}')
        est_action_value = list(iv.copy().astype(float))
        initial_value = list(iv.copy().astype(float))
        qt_chosen = list(np.zeros_like(est_action_value))        
        ret = []
        optimal_action = []
        av = action_values.copy()
        
        iv_nstat = list(np.repeat(0.0,k))
        est_av_nstat = list(np.repeat(0.0,k))
        qt_chosen_nstat = list(np.zeros(k).astype(int))
        ret_nstat = [[] for _ in range(k)]
        ret_t_nstat = []
        optimal_action_nstat = []
        
        for n in range(1, t+1):
            # print(f'Timestep {n} de {t}')
            # Epsilon-greedy function
            if np.random.uniform(0,1,1) >= epsilon:
                maximum = max(est_action_value)
                max_nstat = max(est_av_nstat)
                qtd_max = np.sum(est_action_value==maximum)
                qtd_max_nstat = np.sum(est_av_nstat==max_nstat)
                
                if qtd_max == 1:
                    choice = np.argmax(est_action_value)
                else:
                    maxs = np.array([index for index, value in enumerate(est_action_value) if value == maximum])
                    choice = np.random.choice(maxs)
                
                if qtd_max_nstat == 1:
                    choice_nstat = np.argmax(est_av_nstat)
                else:
                    maxs_nstat = np.array([index for index, value in enumerate(est_av_nstat) if value == max_nstat])
                    choice_nstat = np.random.choice(maxs_nstat)
            else:
                choice = np.random.choice(range(len(est_action_value)))
                choice_nstat = choice
            
            
            
            qt_chosen[choice] +=1 
            r_t = float(np.random.normal(loc=av[choice],scale=1.0,size=1))
            ret.append(r_t)
            
            qt_chosen_nstat[choice_nstat] +=1
            r_t_nstat = float(np.random.normal(loc=av[choice_nstat],scale=1.0,size=1))
            ret_nstat[choice_nstat].append(r_t_nstat)
            ret_t_nstat.append(r_t_nstat)
            
            # If alpha is None, computes the simple average, otherwise computes the weighted average 
            # for nonstationary problems
            update = est_action_value[choice] + (1/qt_chosen[choice]) * (r_t - est_action_value[choice])
            update_nstat = wa_nonstationary(alpha, iv_nstat[choice_nstat], ret_nstat[choice_nstat], qt_chosen_nstat[choice_nstat])
                
            est_action_value[choice] = update
            est_av_nstat[choice_nstat] = update_nstat

            if av[choice] == av[np.argmax(av)]:
                optimal_action.append(1)
            else:
                optimal_action.append(0)
            
            if av[choice_nstat] == av[np.argmax(av)]:
                optimal_action_nstat.append(1)
            else:
                optimal_action_nstat.append(0)
            
            av = stationary_func(av, stationary=stationary, **kwargs)
            
    
        ret_total = pd.concat([ret_total, pd.DataFrame(ret, columns = [i+1], index=range(1,t+1))], axis=1, sort=False, )
        opt_act_tot = pd.concat([opt_act_tot, pd.DataFrame(optimal_action, columns=[i+1], index=range(1,t+1))], axis=1, sort=False)

        ret_total_nstat = pd.concat([ret_total_nstat, pd.DataFrame(ret_t_nstat, columns = [i+1], index=range(1,t+1))], axis=1, sort=False, )
        opt_act_tot_nstat = pd.concat([opt_act_tot_nstat, pd.DataFrame(optimal_action_nstat, columns=[i+1], index=range(1,t+1))], axis=1, sort=False)
        
    return ret_total, opt_act_tot, ret_total_nstat, opt_act_tot_nstat