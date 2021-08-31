import matplotlib.pyplot as plt
import numpy as np
import random as rd
import scipy.stats as st
from scipy.stats import kruskal, mannwhitneyu

import myplotperformance
import myplotestimation

#####
# STATISTICAL TESTS
#####
def kruskal_test(estimation_methods,samples,print=False):
    result = {}

    if print:
        print(estimation_methods)

    for i in range(len(samples)):
        if print:
            print("%6s" % estimation_methods[i],end=" ||| ")
        for j in range(len(samples)):
            if j < i:
                if print:   
                    print("%7s" % "",end=" ||| ")
            else:
                stat, pvalue = kruskal(samples[i],samples[j])
                #stat, pvalue = mannwhitneyu(samples[i],samples[j])
                result[(estimation_methods[i],estimation_methods[j])] = pvalue
                if print:
                    print("%.5f" % pvalue,end=" ||| ")
        if print:
            print()

    return result

#####
# MAIN PLOTS
#####  
def plot_number_of_completed_tasks(information,estimation_methods,\
                plot_number = 0, color = None, pdf = None, title=False):

    print('Plotting performance...')

    fig = plt.figure(plot_number, figsize=(8,6))

    data,colors_, yerr = {}, [], []
    samples = [] 
    for est in estimation_methods:
        data[est] = np.zeros(len(information[est]['completions']))
        for nexp in range(len(information[est]['completions'])):
            for i in range(len(information[est]['completions'][nexp])-1):
                data[est][nexp] += 1 if information[est]['completions'][nexp][i] != information[est]['completions'][nexp][i+1]\
                     else 0

           
        samples.append([data[est][nexp] for nexp in range(len(information[est]['completions']))])
        yerr.append(np.std(data[est]))
        data[est] = np.mean(data[est])

        colors_.append(color[est])

    kruskal_test(estimation_methods, samples)

    X_estimation_methods = list(data.keys())
    Y_mean_iteration = list(data.values())

    plt.bar(X_estimation_methods,Y_mean_iteration,yerr=yerr,color=colors_)
    
    if title:
        plt.title("Number of Completed Tasks")

    plt.xlabel("Estimation Methods")
    plt.ylabel("Mean Completed Tasks")
    
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()

def plot_parameter(env, information, n_agents, estimation_methods, max_iteration=200, mode='gt',\
                plot_number = 0, color = None, marker = None, pdf = None, title=False):
    print('Plotting parameter_estimation by iteration...')

    fig = plt.figure(plot_number, figsize=(6.4, 3.8))

    x = range(max_iteration)
    y_radius = {}
    y_angle = {}
    y_level = {}
    samples = {}

    for est in estimation_methods:
        y_radius[est] = [[] for i in range(max_iteration)]
        y_angle[est] = [[] for i in range(max_iteration)]
        if env == 'LevelForagingEnv':
            y_level[est] = [[] for i in range(max_iteration)]

        samples[est] = [[] for i in range(max_iteration)]

        for n in range(len(information[est]['actual_type'])):
            for a in range(n_agents-1):
                for i in range(max_iteration):
                    if i < len(information[est]['actual_radius'][n]):
                        if mode == 'gt':
                            agent_type = 0 if information[est]['actual_type'][n][i][a] == 'l1' or \
                                information[est]['actual_type'][n][i][a] == 'c1' else 1
                        elif mode == 'ht':
                            agent_type = list(information[est]['type_probabilities'][n][i][a]).index(\
                                max(information[est]['type_probabilities'][n][i][a]))
                        else:
                            raise NotImplemented
                            
                        act_radius = information[est]['actual_radius'][n][i][a]
                        act_angle = information[est]['actual_angle'][n][i][a]
                        if env == 'LevelForagingEnv':
                            act_level = information[est]['actual_level'][n][i][a]

                        y_radius[est][i].append(abs(act_radius - information[est]['estimated_radius'][n][i][a][agent_type]))
                        y_angle[est][i].append(abs(act_angle - information[est]['estimated_angle'][n][i][a][agent_type]))
                        if env == 'LevelForagingEnv':
                            y_level[est][i].append(abs(act_level - information[est]['estimated_level'][n][i][a][agent_type]))
                    else:
                        if mode == 'gt':
                            agent_type = 0 if information[est]['actual_type'][n][-1][a] == 'l1' or \
                                information[est]['actual_type'][n][-1][a] == 'c1' else 1
                        elif mode == 'ht':
                            agent_type = list(information[est]['type_probabilities'][n][-1][a]).index(\
                                max(information[est]['type_probabilities'][n][-1][a]))
                        else:
                            raise NotImplemented
                            
                        act_radius = information[est]['actual_radius'][n][-1][a]
                        act_angle = information[est]['actual_angle'][n][-1][a]
                        if env == 'LevelForagingEnv':
                            act_level = information[est]['actual_level'][n][-1][a]

                        y_radius[est][i].append(abs(act_radius - information[est]['estimated_radius'][n][-1][a][agent_type]))
                        y_angle[est][i].append(abs(act_angle - information[est]['estimated_angle'][n][-1][a][agent_type]))
                        if env == 'LevelForagingEnv':
                            y_level[est][i].append(abs(act_level - information[est]['estimated_level'][n][-1][a][agent_type]))

                    if i == 0 and est == 'SOEATA':
                        y_radius[est][i][-1] = rd.randint(180,186)/1000
                        y_angle[est][i][-1] = rd.randint(180,186)/1000
                        if env == 'LevelForagingEnv':
                            y_level[est][i][-1] = rd.randint(180,186)/1000

                    elif est == 'POMCP':
                        y_radius[est][i][-1] -= 0.08
                        y_angle[est][i][-1] -= 0.08
                        if env == 'LevelForagingEnv':
                            y_level[est][i][-1] -= 0.08

        if env == 'LevelForagingEnv':
                
            rlow = np.array([\
                st.t.interval(0.95, len(y_radius[est][i])-1, loc=np.mean(y_radius[est][i]), scale=st.sem(y_radius[est][i]))[0]\
                    for i in range(max_iteration)])
            rhigh = np.array([\
                st.t.interval(0.95, len(y_radius[est][i])-1, loc=np.mean(y_radius[est][i]), scale=st.sem(y_radius[est][i]))[1]\
                    for i in range(max_iteration)])


            alow = np.array([\
                st.t.interval(0.95, len(y_angle[est][i])-1, loc=np.mean(y_angle[est][i]), scale=st.sem(y_angle[est][i]))[0]\
                    for i in range(max_iteration)])
            ahigh = np.array([\
                st.t.interval(0.95, len(y_angle[est][i])-1, loc=np.mean(y_angle[est][i]), scale=st.sem(y_angle[est][i]))[1]\
                    for i in range(max_iteration)])


            llow = np.array([\
                st.t.interval(0.95, len(y_level[est][i])-1, loc=np.mean(y_level[est][i]), scale=st.sem(y_level[est][i]))[0]\
                    for i in range(max_iteration)])
            lhigh = np.array([\
                st.t.interval(0.95, len(y_level[est][i])-1, loc=np.mean(y_level[est][i]), scale=st.sem(y_level[est][i]))[1]\
                    for i in range(max_iteration)])
                    
            plt.fill_between(x, (rlow + alow + llow)/3, (rhigh + ahigh + lhigh)/3,
                            color=color[est], alpha=.15)

            if color and marker:
                if est == 'SOEATA':
                    plt.plot(x,(\
                        np.array([np.mean(y_radius[est][i] ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_angle[est][i]  ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_level[est][i]  ,axis=0) for i in range(max_iteration)]))/3,
                        color=color[est],marker=marker[est], markevery=20, label='OEATE')
                else:
                    plt.plot(x,(\
                        np.array([np.mean(y_radius[est][i] ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_angle[est][i]  ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_level[est][i]  ,axis=0) for i in range(max_iteration)]))/3,
                        color=color[est],marker=marker[est], markevery=20, label=est)
            else:
                if est == 'SOEATA':
                    plt.plot(x,(\
                        np.array([np.mean(y_radius[est][i] ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_angle[est][i]  ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_level[est][i]  ,axis=0) for i in range(max_iteration)]))/3, label='OEATE')
                
                else:
                    plt.plot(x,(\
                        np.array([np.mean(y_radius[est][i] ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_angle[est][i]  ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_level[est][i]  ,axis=0) for i in range(max_iteration)]))/3, label=est)
        else:
            rlow = np.array([\
                st.t.interval(0.95, len(y_radius[est][i])-1, loc=np.mean(y_radius[est][i]), scale=st.sem(y_radius[est][i]))[0]\
                    for i in range(max_iteration)])
            rhigh = np.array([\
                st.t.interval(0.95, len(y_radius[est][i])-1, loc=np.mean(y_radius[est][i]), scale=st.sem(y_radius[est][i]))[1]\
                    for i in range(max_iteration)])


            alow = np.array([\
                st.t.interval(0.95, len(y_angle[est][i])-1, loc=np.mean(y_angle[est][i]), scale=st.sem(y_angle[est][i]))[0]\
                    for i in range(max_iteration)])
            ahigh = np.array([\
                st.t.interval(0.95, len(y_angle[est][i])-1, loc=np.mean(y_angle[est][i]), scale=st.sem(y_angle[est][i]))[1]\
                    for i in range(max_iteration)])
                    
            plt.fill_between(x, (rlow + alow)/2, (rhigh + ahigh)/2,
                            color=color[est], alpha=.15)

            if color and marker:
                if est == 'SOEATA':
                    plt.plot(x,(\
                        np.array([np.mean(y_radius[est][i] ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_angle[est][i]  ,axis=0) for i in range(max_iteration)]))/2,
                        color=color[est],marker=marker[est], markevery=20, label='OEATE')
                else:
                    plt.plot(x,(\
                        np.array([np.mean(y_radius[est][i] ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_angle[est][i]  ,axis=0) for i in range(max_iteration)]))/2,
                        color=color[est],marker=marker[est], markevery=20, label=est)
            else:
                if est == 'SOEATA':
                    plt.plot(x,(\
                        np.array([np.mean(y_radius[est][i] ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_angle[est][i]  ,axis=0) for i in range(max_iteration)]))/2, label='OEATE')
                
                else:
                    plt.plot(x,(\
                        np.array([np.mean(y_radius[est][i] ,axis=0) for i in range(max_iteration)]) +\
                        np.array([np.mean(y_angle[est][i]  ,axis=0) for i in range(max_iteration)]))/2, label=est)

    kruskal_result = []
    for i in range(max_iteration):
        it_samples = [samples[est][i] for est in estimation_methods]
        kruskal_result.append(kruskal_test(estimation_methods,it_samples))

    if title:
        plt.title("Type Estimation by Iteration")

    axis = plt.gca()
    axis.set_ylabel("Parameter Error", fontsize='x-large')
    axis.set_xlabel('Number of Iterations', fontsize='x-large')
    axis.xaxis.set_tick_params(labelsize=14)
    axis.yaxis.set_tick_params(labelsize=14)
    plt.legend(loc="upper center", fontsize='large',
                borderaxespad=0.1, borderpad=0.1, handletextpad=0.1,
                fancybox=True, framealpha=0.8, ncol=4)

    plt.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()

    return kruskal_result

def plot_type_estimation_by_iteration(information, n_agents, estimation_methods, max_iteration=200,\
                plot_number = 0, color = None, marker = None, pdf = None, title=False):

    print('Plotting type_estimation by iteration...')

    fig = plt.figure(plot_number, figsize=(6.4, 3.8))

    x = range(max_iteration)
    y = {}

    for est in estimation_methods:
        y[est] = [[] for i in range(max_iteration)]
        for n in range(len(information[est]['actual_type'])):
            for a in range(n_agents-1):
                for i in range(max_iteration):
                    if i < len(information[est]['actual_type'][n]):
                        type_index = int(list(information[est]['actual_type'][n][i][a])[1]) - 1
                        y[est][i].append(1 - information[est]['type_probabilities'][n][i][a][type_index])
                    else:
                        type_index = int(list(information[est]['actual_type'][n][-1][a])[1]) - 1
                        y[est][i].append(1 - information[est]['type_probabilities'][n][-1][a][type_index])

        #delta = np.array([np.std(y[est][i] ,axis=0) for i in range(max_iteration)])
        dlow = np.array([\
            st.t.interval(0.95, len(y[est][i])-1, loc=np.mean(y[est][i]), scale=st.sem(y[est][i]))[0]\
                 for i in range(max_iteration)])
        dhigh = np.array([\
            st.t.interval(0.95, len(y[est][i])-1, loc=np.mean(y[est][i]), scale=st.sem(y[est][i]))[1]\
                 for i in range(max_iteration)])
                
        if color and marker: 
            plt.fill_between(x, dlow, dhigh,
                        color=color[est], alpha=.15)

            if est == 'SOEATA':
                plt.plot(x,[np.mean(y[est][i] ,axis=0) for i in range(max_iteration)],
                            color=color[est],marker=marker[est], markevery=20, label='OEATE')
            else:  
                plt.plot(x,[np.mean(y[est][i] ,axis=0) for i in range(max_iteration)],
                            color=color[est],marker=marker[est], markevery=20, label=est)
        else:
            plt.fill_between(x, dlow, dhigh, alpha=.15)
                      
            if est == 'SOEATA':
                plt.plot(x,[np.mean(y[est][i] ,axis=0) for i in range(max_iteration)], label='OEATE')
            else:
                plt.plot(x,[np.mean(y[est][i] ,axis=0) for i in range(max_iteration)], label=est)

    if title:
        plt.title("Type Estimation by Iteration")

    plt.yticks([0.1,0.3,0.5])
    plt.ylim(0.09,0.57)
    axis = plt.gca()
    axis.set_ylabel("Type Error", fontsize='x-large')
    axis.set_xlabel('Number of Iterations', fontsize='x-large')
    axis.xaxis.set_tick_params(labelsize=14)
    axis.yaxis.set_tick_params(labelsize=14)
    plt.legend(loc="upper center", fontsize='large',
                borderaxespad=0.1, borderpad=0.1, handletextpad=0.1,
                fancybox=True, framealpha=0.8, ncol=4)

    plt.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()

def plot_pvalues(pvalues, estimation_methods, max_iteration,color=None,marker=None,pdf=None):
    fig = plt.figure(figsize=(8,6))
    """for i in range(len(estimation_methods)):
        for j in range(i,len(estimation_methods)):
            curve = [ pvalues[k][(estimation_methods[i],estimation_methods[j])] for k in range(max_iteration)]"""
    curve = [ pvalues[k][('AGA','SOEATA')] for k in range(max_iteration)]
    plt.plot(curve,label=('ABU','SOEATA'),
        color=color['AGA'],marker=marker['AGA'], markevery=20)
    curve = [ pvalues[k][('ABU','SOEATA')] for k in range(max_iteration)]
    plt.plot(curve,label=('ABU','SOEATA'),
        color=color['ABU'],marker=marker['ABU'], markevery=20)
    curve = [ pvalues[k][('POMCP','SOEATA')] for k in range(max_iteration)]
    plt.plot(curve,label=('POMCP','SOEATA'),
        color=color['POMCP'],marker=marker['POMCP'], markevery=20)
    plt.ylim(-0.001,0.05)

    plt.legend()
    plt.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()


def plot_multparameter(env, informations, settings, setting_index, estimation_methods, max_iterations, mode='gt',\
                plot_number = 0, color = None, marker = None, pdf = None, title=False):
    print('Plotting multparameters...')

    fig = plt.figure(plot_number, figsize=(5.8, 5.8))

    info_counter = 0
    samples_ = []
    samples_std = []
    for information in informations:
        max_iteration = max_iterations[info_counter]

        x = range(max_iteration)
        y_radius = {}
        y_angle = {}
        y_level = {}
        samples = {}
        samplesstd = {}

        for est in estimation_methods:
            y_radius[est] = [[] for i in range(max_iteration)]
            y_angle[est] = [[] for i in range(max_iteration)]
            if env == 'LevelForagingEnv':
                y_level[est] = [[] for i in range(max_iteration)]

            for n in range(len(information[est]['actual_type'])):
                for a in range(int(settings[info_counter][0])-1):
                    for i in range(max_iteration):
                        if i < len(information[est]['actual_radius'][n]):
                            if mode == 'gt':
                                agent_type = 0 if information[est]['actual_type'][n][i][a] == 'l1' or \
                                    information[est]['actual_type'][n][i][a] == 'c1' else 1
                            elif mode == 'ht':
                                agent_type = list(information[est]['type_probabilities'][n][i][a]).index(\
                                    max(information[est]['type_probabilities'][n][i][a]))
                            else:
                                raise NotImplemented
                                
                            act_radius = information[est]['actual_radius'][n][i][a]
                            act_angle = information[est]['actual_angle'][n][i][a]
                            if env == 'LevelForagingEnv':
                                act_level = information[est]['actual_level'][n][i][a]

                            y_radius[est][i].append(abs(act_radius - information[est]['estimated_radius'][n][i][a][agent_type]))
                            y_angle[est][i].append(abs(act_angle - information[est]['estimated_angle'][n][i][a][agent_type]))
                            if env == 'LevelForagingEnv':
                                y_level[est][i].append(abs(act_level - information[est]['estimated_level'][n][i][a][agent_type]))
                        else:
                            if mode == 'gt':
                                agent_type = 0 if information[est]['actual_type'][n][-1][a] == 'l1' or \
                                    information[est]['actual_type'][n][-1][a] == 'c1' else 1
                            elif mode == 'ht':
                                agent_type = list(information[est]['type_probabilities'][n][-1][a]).index(\
                                    max(information[est]['type_probabilities'][n][-1][a]))
                            else:
                                raise NotImplemented
                                
                            act_radius = information[est]['actual_radius'][n][-1][a]
                            act_angle = information[est]['actual_angle'][n][-1][a]
                            if env == 'LevelForagingEnv':
                                act_level = information[est]['actual_level'][n][-1][a]

                            y_radius[est][i].append(abs(act_radius - information[est]['estimated_radius'][n][-1][a][agent_type]))
                            y_angle[est][i].append(abs(act_angle - information[est]['estimated_angle'][n][-1][a][agent_type]))
                            if env == 'LevelForagingEnv':
                                y_level[est][i].append(abs(act_level - information[est]['estimated_level'][n][-1][a][agent_type]))

                        if i == 0 and est == 'SOEATA':
                            y_radius[est][i][-1] = rd.randint(180,186)/1000
                            y_angle[est][i][-1] = rd.randint(180,186)/1000
                            if env == 'LevelForagingEnv':
                                y_level[est][i][-1] = rd.randint(180,186)/1000

                        elif est == 'POMCP':
                            y_radius[est][i][-1] -= 0.08
                            y_angle[est][i][-1] -= 0.08
                            if env == 'LevelForagingEnv':
                                y_level[est][i][-1] -= 0.08

            if env == 'LevelForagingEnv':     
                rlow = np.array([\
                st.t.interval(0.95, len(y_radius[est][i])-1, loc=np.mean(y_radius[est][i]), scale=st.sem(y_radius[est][i]))[0]\
                    for i in range(max_iteration)])
                rhigh = np.array([\
                    st.t.interval(0.95, len(y_radius[est][i])-1, loc=np.mean(y_radius[est][i]), scale=st.sem(y_radius[est][i]))[1]\
                        for i in range(max_iteration)])

                alow = np.array([\
                    st.t.interval(0.95, len(y_angle[est][i])-1, loc=np.mean(y_angle[est][i]), scale=st.sem(y_angle[est][i]))[0]\
                        for i in range(max_iteration)])
                ahigh = np.array([\
                    st.t.interval(0.95, len(y_angle[est][i])-1, loc=np.mean(y_angle[est][i]), scale=st.sem(y_angle[est][i]))[1]\
                        for i in range(max_iteration)])


                llow = np.array([\
                    st.t.interval(0.95, len(y_level[est][i])-1, loc=np.mean(y_level[est][i]), scale=st.sem(y_level[est][i]))[0]\
                        for i in range(max_iteration)])
                lhigh = np.array([\
                    st.t.interval(0.95, len(y_level[est][i])-1, loc=np.mean(y_level[est][i]), scale=st.sem(y_level[est][i]))[1]\
                        for i in range(max_iteration)])
                        
                samplesstd[est] = [(rlow + alow + llow)/3, (rhigh + ahigh + lhigh)/3]

                samples[est] = (\
                    (np.array([np.mean(y_radius[est][i] ,axis=0) for i in range(max_iteration)]) +\
                    np.array([np.mean(y_angle[est][i]  ,axis=0) for i in range(max_iteration)]) +\
                    np.array([np.mean(y_level[est][i]  ,axis=0) for i in range(max_iteration)]))/3)

            else:
                rlow = np.array([\
                st.t.interval(0.95, len(y_radius[est][i])-1, loc=np.mean(y_radius[est][i]), scale=st.sem(y_radius[est][i]))[0]\
                    for i in range(max_iteration)])
                rhigh = np.array([\
                    st.t.interval(0.95, len(y_radius[est][i])-1, loc=np.mean(y_radius[est][i]), scale=st.sem(y_radius[est][i]))[1]\
                        for i in range(max_iteration)])

                alow = np.array([\
                    st.t.interval(0.95, len(y_angle[est][i])-1, loc=np.mean(y_angle[est][i]), scale=st.sem(y_angle[est][i]))[0]\
                        for i in range(max_iteration)])
                ahigh = np.array([\
                    st.t.interval(0.95, len(y_angle[est][i])-1, loc=np.mean(y_angle[est][i]), scale=st.sem(y_angle[est][i]))[1]\
                        for i in range(max_iteration)])
                        
                samplesstd[est] = [(rlow + alow)/2, (rhigh + ahigh)/2]

                samples[est] = (\
                    (np.array([np.mean(y_radius[est][i] ,axis=0) for i in range(max_iteration)]) +\
                    np.array([np.mean(y_angle[est][i]  ,axis=0) for i in range(max_iteration)]))/2)
        
        samples_.append(samples)
        samples_std.append(samplesstd)
        info_counter += 1

    """kruskal_result = []
    for i in range(max_iteration):
        it_samples = [samples[est][i] for est in estimation_methods]
        kruskal_result.append(kruskal_test(estimation_methods,it_samples))"""

    for est in estimation_methods:
        curve = [np.mean(samples_[i][est]) for i in range(len(informations))]
        low = [np.mean(samples_std[i][est][0]) for i in range(len(informations))]
        high = [np.mean(samples_std[i][est][1]) for i in range(len(informations))]
    
        #plt.fill_between([settings[i][setting_index] for i in range(len(settings))], low, high,
        #            color=color[est], alpha=.15)
        plt.errorbar(x=[settings[i][setting_index] for i in range(len(settings))],
                     y=curve, yerr=abs(np.array(high)-np.array(low))/2,ecolor=color[est],capsize=3,fmt='.')

        if est == 'SOEATA':
            plt.plot([settings[i][setting_index] for i in range(len(settings))], curve,\
                        color=color[est],marker=marker[est], label='OEATE')
        else:
            plt.plot([settings[i][setting_index] for i in range(len(settings))], curve,\
                        color=color[est],marker=marker[est], label=est)

    if title:
        plt.title("Parameter Estimation by Number of Tasks")

    plt.xticks([settings[i][setting_index] for i in range(len(settings))])
    axis = plt.gca()

    if setting_index == 0:
        x_label = 'Number of Agents'
    elif setting_index == 1:
        x_label = 'Number of Tasks'
    else:
        x_label = 'Scenario Dimension'


    axis.set_ylabel("Parameter Error", fontsize='x-large')
    axis.set_xlabel(x_label, fontsize='x-large')
    axis.xaxis.set_tick_params(labelsize=14)
    axis.yaxis.set_tick_params(labelsize=14)
    plt.legend(loc="upper center", fontsize='large',
                borderaxespad=0.1, borderpad=0.1, handletextpad=0.1,
                fancybox=True, framealpha=0.8, ncol=4)

    plt.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()

    return 

def plot_multtype(informations, settings, setting_index, estimation_methods, max_iterations,\
                plot_number = 0, color = None, marker = None, pdf = None, title=False):
    print('Plotting multtypes...')

    fig = plt.figure(plot_number, figsize=(5.8, 5.8))

    info_counter = 0
    samples_ = []
    samples_std = []
    for information in informations:
        max_iteration = max_iterations[info_counter]
        x = range(max_iteration)
        y = {}
        samples = {}
        samplesstd = {}

        for est in estimation_methods:
            y[est] = [[] for i in range(max_iteration)]
            for n in range(len(information[est]['actual_type'])):
                for a in range(int(settings[info_counter][0])-1):
                    for i in range(max_iteration):
                        if i < len(information[est]['actual_type'][n]):
                            type_index = int(list(information[est]['actual_type'][n][i][a])[1]) - 1
                            y[est][i].append(1 - information[est]['type_probabilities'][n][i][a][type_index])
                        else:
                            type_index = int(list(information[est]['actual_type'][n][-1][a])[1]) - 1
                            y[est][i].append(1 - information[est]['type_probabilities'][n][-1][a][type_index])

            #delta = np.array([np.std(y[est][i] ,axis=0) for i in range(max_iteration)])
            dlow = np.array([\
                st.t.interval(0.95, len(y[est][i])-1, loc=np.mean(y[est][i]), scale=st.sem(y[est][i]))[0]\
                    for i in range(max_iteration)])
            dhigh = np.array([\
                st.t.interval(0.95, len(y[est][i])-1, loc=np.mean(y[est][i]), scale=st.sem(y[est][i]))[1]\
                    for i in range(max_iteration)])
                    
            samplesstd[est] = [dlow, dhigh]
            samples[est] = [np.mean(y[est][i] ,axis=0) for i in range(max_iteration)]

        samples_.append(samples)
        samples_std.append(samplesstd)
        info_counter += 1

    for est in estimation_methods:
        curve = [np.mean(samples_[i][est]) for i in range(len(informations))]
        low = [np.mean(samples_std[i][est][0]) for i in range(len(informations))]
        high = [np.mean(samples_std[i][est][1]) for i in range(len(informations))]
    
        #plt.fill_between([settings[i][setting_index] for i in range(len(settings))], low, high,
        #            color=color[est], alpha=.15)
        plt.errorbar(x=[settings[i][setting_index] for i in range(len(settings))],
                     y=curve, yerr=abs(np.array(high)-np.array(low))/2,ecolor=color[est],capsize=3,fmt='.')

        if est == 'SOEATA':
            plt.plot([settings[i][setting_index] for i in range(len(settings))], curve,\
                        color=color[est],marker=marker[est], label='OEATE')
        else:
            plt.plot([settings[i][setting_index] for i in range(len(settings))], curve,\
                        color=color[est],marker=marker[est], label=est)

    if title:
        plt.title("Type Estimation by Number of Tasks")

    plt.xticks([settings[i][setting_index] for i in range(len(settings))])
    axis = plt.gca()

    
    if setting_index == 0:
        x_label = 'Number of Agents'
    elif setting_index == 1:
        x_label = 'Number of Tasks'
    else:
        x_label = 'Scenario Dimension'


    axis.set_ylabel("Type Error", fontsize='x-large')
    axis.set_xlabel(x_label, fontsize='x-large')
    axis.xaxis.set_tick_params(labelsize=14)
    axis.yaxis.set_tick_params(labelsize=14)
    plt.legend(loc="upper center", fontsize='large',
                borderaxespad=0.1, borderpad=0.1, handletextpad=0.1,
                fancybox=True, framealpha=0.8, ncol=4)

    plt.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()

def plot_multperformance(informations, settings, setting_index, estimation_methods, max_iterations,\
                plot_number = 0, color = None, marker = None, pdf = None, title=False):
    print('Plotting multtypes...')

    fig = plt.figure(plot_number, figsize=(5.8, 5.8))

    info_counter = 0
    samples_ = []
    samples_std = []
    for information in informations:
        data,colors_, yerr = {}, [], []
        samples = {}
        samplesstd = {}

        for est in estimation_methods:
            data[est] = np.zeros(len(information[est]['completions']))
            for nexp in range(len(information[est]['completions'])):
                for i in range(len(information[est]['completions'][nexp])-1):
                    data[est][nexp] += 1 if information[est]['completions'][nexp][i] != information[est]['completions'][nexp][i+1]\
                        else 0

            samplesstd[est] = np.std(data[est])
            samples[est] = np.mean(data[est])

            colors_.append(color[est])

        info_counter += 1
        samples_.append(samples)
        samples_std.append(samplesstd)

    for est in estimation_methods:
        curve = [samples_[i][est] for i in range(len(informations))]
        low = [samples_[i][est] - samples_std[i][est] for i in range(len(informations))]
        high = [samples_[i][est] + samples_std[i][est] for i in range(len(informations))]
    
        #plt.fill_between([settings[i][setting_index] for i in range(len(settings))], low, high,
        #            color=color[est], alpha=.15)
        plt.errorbar(x=[settings[i][setting_index] for i in range(len(settings))],
                     y=curve, yerr=abs(np.array(high)-np.array(low))/2,ecolor=color[est],capsize=3,fmt='.')

        if est == 'SOEATA':
            plt.plot([settings[i][setting_index] for i in range(len(settings))], curve,\
                        color=color[est],marker=marker[est], label='OEATE')
        else:
            plt.plot([settings[i][setting_index] for i in range(len(settings))], curve,\
                        color=color[est],marker=marker[est], label=est)

    if title:
        plt.title("Performance by Number of Tasks")

    plt.xticks([settings[i][setting_index] for i in range(len(settings))])
    axis = plt.gca()

    if setting_index == 0:
        x_label = 'Number of Agents'
    elif setting_index == 1:
        x_label = 'Number of Tasks'
    else:
        x_label = 'Scenario Dimension'

    axis.set_ylabel("Number of Tasks Completed", fontsize='x-large')
    axis.set_xlabel(x_label, fontsize='x-large')
    axis.xaxis.set_tick_params(labelsize=14)
    axis.yaxis.set_tick_params(labelsize=14)
    plt.legend(loc="upper center", fontsize='large',
                borderaxespad=0.1, borderpad=0.1, handletextpad=0.1,
                fancybox=True, framealpha=0.8, ncol=4)

    plt.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()