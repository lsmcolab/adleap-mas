import numpy as np
import scipy.stats
from scipy.stats import ttest_ind
from utils import *

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)-1
    m, se = np.mean(a), scipy.stats.sem(a)
    h = scipy.stats.t.interval(alpha=confidence, df=n,loc=m,scale=se)
    return [m, h[1], h[0]]

def mean_performance(results):
    performance = []
    for exp in range(len(results)):
        performance.append(results[exp]['it'][-1])
    return mean_confidence_interval(performance)
    
def mean_reward(results,return_=False):
    reward = []
    for exp in range(len(results)):
        reward.append(np.mean(results[exp]['reward']))
    if return_:
        return mean_confidence_interval(reward), reward
    else:
        return mean_confidence_interval(reward)
    
def get_cummulative_reward(results):
    reward = []
    max_range = 200
    for exp in range(len(results)):
        reward.append(list(np.cumsum(results[exp]['reward'])))
        while len(reward[-1]) < max_range:
            reward[-1].append(reward[-1][-1])

    cum_vector = np.zeros(max_range)
    for i in range(len(reward)):
        cum_vector += reward[i]
    return cum_vector/len(reward)

def mean_cummulative_reward(results):
    reward = []
    for exp in range(len(results)):
        reward.append(np.cumsum(results[exp]['reward'])[-1])
    return mean_confidence_interval(reward)

def mean_time(results,return_=False):
    time = []
    for exp in range(len(results)):
        time.append(np.mean(results[exp]['time']))
    if return_:
        return mean_confidence_interval(time), time
    else:
        return mean_confidence_interval(time)

def mean_total_time(results):
    time = []
    for exp in range(len(results)):
        time.append(np.sum(results[exp]['time']))
    return mean_confidence_interval(time)

def get_pvalues(results,methods_names,print_=False):
    pvalues = {}

    # 1. Get rewards
    rewards_result, rewards = {}, {}
    for i in range(len(methods_names)):
        rewards[methods_names[i]], rewards_result[methods_names[i]] = \
                            mean_reward(results[methods_names[i]],return_=True)

    # 2. Calculating the pvalues
    for i in range(len(methods_names)):
        for j in range(len(methods_names)):
            method1, method2 = methods_names[i],methods_names[j]
            pvalues[(method1,method2)] = ttest_ind(rewards[method1],rewards[method2],equal_var=False)[1]

    # 3. Printing the pvalues
    if print_:
        print(methods_names)
        for i in range(len(methods_names)):
            for j in range(len(methods_names)):
                method1, method2 = methods_names[i],methods_names[j]
                print('%.2f' % (pvalues[(method1,method2)]) + '\t',end='')
            print()
    
    # 4. Returning pvalues
    return pvalues
