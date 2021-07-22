from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
import numpy as np

#####
# CUSTOMISATION
#####
PLOT_NUMBER = 0
COLOR = {'AGA':'b','ABU':'g','OEATA':'r'}

#####
# COLLECTING INFORMATION
#####
def collect_information(a, i, d, n_experiments, env,mode=""):
    print('Collecting information...',a,i,d,env,mode)
    information = {}
    for est in ['AGA','ABU','OEATA']:
        # initialising 
        information[est] = {}
        information[est]['iterations'] = []
        information[est]['completions'] = []
        information[est]['actual_radius'] = []
        information[est]['actual_angle'] = []
        information[est]['actual_level'] = []
        information[est]['actual_type'] = []
        information[est]['estimated_radius'] = []
        information[est]['estimated_angle'] = []
        information[est]['estimated_level'] = []
        information[est]['type_probabilities'] = []

        # collecting the results
        for exp in range(n_experiments):

            information[est]['iterations'].append([])
            information[est]['completions'].append([])
            information[est]['actual_radius'].append([])
            information[est]['actual_angle'].append([])
            information[est]['actual_level'].append([])
            information[est]['actual_type'].append([])
            information[est]['estimated_radius'].append([])
            information[est]['estimated_angle'].append([])
            information[est]['estimated_level'].append([])
            information[est]['type_probabilities'].append([])

            with open('results/'+mode+env+'_a'+str(a)+'_i'+str(i)+\
             '_dim'+str(d)+'_'+str(est)+'_exp'+str(exp)+'.csv','r') as csv:
                line_counter = 0
                for line in csv:
                    if line_counter != 0:
                        tokens = line.split(';')

                        # getting information
                        iteration = tokens[0]
                        completion = tokens[1]
                        environment = tokens[2]
                        estimation = tokens[3]
                        act_radius = tokens[4]
                        act_angle = tokens[5]
                        if environment == 'LevelForagingEnv':
                            act_level = tokens[6]
                        act_type = tokens[7]
                        est_radius = tokens[8]
                        est_angle = tokens[9]
                        if environment == 'LevelForagingEnv':
                            est_level = tokens[10]
                        type_prob = tokens[11]
                        
                        # storing
                        information[est]['iterations'][-1].append(literal_eval(iteration))
                        information[est]['completions'][-1].append(literal_eval(completion))
                        information[est]['actual_radius'][-1].append(literal_eval(act_radius))
                        information[est]['actual_angle'][-1].append(literal_eval(act_angle))
                        if environment == 'LevelForagingEnv':
                            information[est]['actual_level'][-1].append(literal_eval(act_level))
                        information[est]['actual_type'][-1].append(literal_eval(act_type))
                        information[est]['estimated_radius'][-1].append(literal_eval(est_radius))
                        information[est]['estimated_angle'][-1].append(literal_eval(est_angle))
                        if environment == 'LevelForagingEnv':
                            information[est]['estimated_level'][-1].append(literal_eval(est_level))
                        information[est]['type_probabilities'][-1].append(literal_eval(type_prob))

                    line_counter += 1
    return information

#####
# PLOT METHODS
#####
def plot_performance(information):
    print('Plotting performance...')
    global PLOT_NUMBER, COLOR, pdf

    fig = plt.figure(PLOT_NUMBER, figsize=(8,6))

    data, max_iterations = {}, []
    for est in ['AGA','ABU','OEATA']:
        max_iterations.append(max([information[est]['iterations'][i][-1] for i in range(len(information[est]['iterations']))]))
        mean_iteration = np.mean([information[est]['iterations'][i][-1] for i in range(len(information[est]['iterations']))])
        data[est] = mean_iteration

    X_estimation_methods = list(data.keys())
    Y_mean_iteration = list(data.values())

    plt.bar(X_estimation_methods,Y_mean_iteration,color=list(COLOR.values()))

    plt.title("Performance")
    plt.xlabel("Estimation Methods")
    plt.ylabel("Performance")
    #plt.show()
    pdf.savefig( fig )
    PLOT_NUMBER += 1

    return max(max_iterations)

def plot_number_of_completed_tasks(information):
    print('Plotting performance...')
    global PLOT_NUMBER, COLOR, pdf

    fig = plt.figure(PLOT_NUMBER, figsize=(8,6))

    data = {}
    for est in ['AGA','ABU','OEATA']:
        data[est] = np.zeros(len(information[est]['completions']))
        for nexp in range(len(information[est]['completions'])):
            for i in range(len(information[est]['completions'][nexp])-1):
                data[est][nexp] += 1 if information[est]['completions'][nexp][i] != information[est]['completions'][nexp][i+1]\
                     else 0
        data[est] = np.mean(data[est])

    X_estimation_methods = list(data.keys())
    Y_mean_iteration = list(data.values())

    plt.bar(X_estimation_methods,Y_mean_iteration,color=list(COLOR.values()))

    plt.title("Number of Completed Tasks")
    plt.xlabel("Estimation Methods")
    plt.ylabel("Mean Completed Tasks")
    #plt.show()
    pdf.savefig( fig )
    PLOT_NUMBER += 1

def plot_completion(information, max_iterations=200):
    print('Plotting performance...')
    global PLOT_NUMBER, COLOR, pdf

    fig = plt.figure(PLOT_NUMBER, figsize=(8,6))

    data = {}
    for est in ['AGA','ABU','OEATA']:
        data[est] = np.zeros(max_iterations)
        for nexp in range(len(information[est]['completions'])):
            for i in range(max_iterations):
                if len(information[est]['completions'][nexp]) - 1 < i:
                    data[est][i] += information[est]['completions'][nexp][-1]
                else:
                    data[est][i] += information[est]['completions'][nexp][i]
        data[est] /= len(information[est]['completions'])
        plt.plot(range(max_iterations),data[est],color=COLOR[est])

    plt.title("Completion")
    plt.xlabel("Estimation Methods")
    plt.ylabel("Completion")
    #plt.show()
    pdf.savefig( fig )
    PLOT_NUMBER += 1

def plot_type_estimation_by_iteration(information, n_agents, max_iteration=200, n_experiments=100):
    print('Plotting type_estimation by iteration...')
    global PLOT_NUMBER, COLOR, pdf

    fig = plt.figure(PLOT_NUMBER, figsize=(8,6))

    x = range(max_iteration)
    y = {'AGA':np.zeros(max_iteration),'ABU': np.zeros(max_iteration), 'OEATA': np.zeros(max_iteration)}

    for est in ['AGA','ABU','OEATA']:
        for n in range(n_experiments):
            for a in range(n_agents-1):
                for i in range(max_iteration):
                    if i < len(information[est]['actual_type'][n]):
                        type_index = int(list(information[est]['actual_type'][n][i][a])[1]) - 1
                        y[est][i] += (1 - information[est]['type_probabilities'][n][i][a][type_index])
                    else:
                        type_index = int(list(information[est]['actual_type'][n][-1][a])[1]) - 1
                        y[est][i] += (1 - information[est]['type_probabilities'][n][-1][a][type_index])

        y[est] /= n_experiments*n_agents
        plt.plot(x,y[est],color=COLOR[est])

    plt.title("Type Estimation by Iteration")
    plt.xlabel("Itertions")
    plt.ylabel("Error")
    #plt.show()
    pdf.savefig( fig )
    PLOT_NUMBER += 1
    return

def plot_type_estimation_by_completion(information, n_agents, n_tasks, max_iteration=200, n_experiments=100):
    print('Plotting type_estimation by completion...')
    global PLOT_NUMBER, COLOR,pdf

    fig = plt.figure(PLOT_NUMBER, figsize=(8,6))

    x = np.linspace(0,1,n_tasks+1)
    y = {'AGA':np.zeros(len(x)),'ABU': np.zeros(len(x)), 'OEATA': np.zeros(len(x))}
    y_norm_factor = {'AGA':np.zeros(len(x)),'ABU': np.zeros(len(x)), 'OEATA': np.zeros(len(x))}

    for est in ['AGA','ABU','OEATA']:
        for n in range(n_experiments):
            for a in range(n_agents-1):
                for i in range(max_iteration):
                    if i < len(information[est]['actual_type'][n]):
                        completion = int(information[est]['completions'][n][i]*n_tasks)
                        type_index = int(list(information[est]['actual_type'][n][i][a])[1]) - 1
                        y[est][completion] +=\
                            (1 - information[est]['type_probabilities'][n][i][a][type_index])
                        y_norm_factor[est][completion] += 1
                    else:
                        completion = int(information[est]['completions'][n][-1]*n_tasks)
                        type_index = int(list(information[est]['actual_type'][n][-1][a])[1]) - 1
                        y[est][completion] +=\
                            (1 - information[est]['type_probabilities'][n][-1][a][type_index])
                        y_norm_factor[est][completion] += 1

        for i in range(len(y[est])):
            y[est][i] /= y_norm_factor[est][i]
        plt.plot(x,y[est],color=COLOR[est])

    plt.title("Type Estimation by Completion")
    plt.xlabel("Completion")
    plt.ylabel("Error")
    #plt.show()
    pdf.savefig( fig )
    PLOT_NUMBER += 1
    return

def plot_parameter_estimation_by_iteration(env, information, n_agents, max_iteration=200, n_experiments=100, mode='gt'):
    print('Plotting parameter_estimation by iteration...')
    global PLOT_NUMBER, COLOR, pdf

    fig, axs= plt.subplots(nrows = 2, ncols=2, figsize=(15,15))
    ax_r, ax_a, ax_l, ax_p = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

    x = range(max_iteration)
    y_radius = {'AGA':np.zeros(max_iteration),'ABU': np.zeros(max_iteration), 'OEATA': np.zeros(max_iteration)}
    y_angle = {'AGA':np.zeros(max_iteration),'ABU': np.zeros(max_iteration), 'OEATA': np.zeros(max_iteration)}
    if env == 'LevelForagingEnv':
        y_level = {'AGA':np.zeros(max_iteration),'ABU': np.zeros(max_iteration), 'OEATA': np.zeros(max_iteration)}

    for est in ['AGA','ABU','OEATA']:
        for n in range(n_experiments):
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

                        y_radius[est][i] += abs(act_radius - information[est]['estimated_radius'][n][i][a][agent_type])
                        y_angle[est][i] += abs(act_angle - information[est]['estimated_angle'][n][i][a][agent_type])
                        if env == 'LevelForagingEnv':
                            y_level[est][i] += abs(act_level - information[est]['estimated_level'][n][i][a][agent_type])
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

                        y_radius[est][i] += abs(act_radius - information[est]['estimated_radius'][n][-1][a][agent_type])
                        y_angle[est][i] += abs(act_angle - information[est]['estimated_angle'][n][-1][a][agent_type])
                        if env == 'LevelForagingEnv':
                            y_level[est][i] += abs(act_level - information[est]['estimated_level'][n][-1][a][agent_type])

        y_radius[est] /= (n_experiments*n_agents)
        y_angle[est] /= (n_experiments*n_agents)
        if env == 'LevelForagingEnv':
            y_level[est] /= n_experiments*n_agents

        ax_r.plot(x,y_radius[est],color=COLOR[est])
        ax_a.plot(x,y_angle[est],color=COLOR[est])
        if env == 'LevelForagingEnv':
            ax_l.plot(x,y_level[est],color=COLOR[est])
            ax_p.plot(x,(y_radius[est] + y_angle[est] + y_level[est])/3,color=COLOR[est])
        else:
            ax_p.plot(x,(y_radius[est] + y_angle[est])/2,color=COLOR[est])

    parameter_ax = [ax_r, ax_a, ax_l, ax_p]
    parameter_title = ['Radius', 'Angle', 'Level', 'General']
    for i in range(len(parameter_ax)):
        parameter_ax[i].set_title(parameter_title[i])
        parameter_ax[i].set(xlabel="Iteration",ylabel='Error')
    #plt.show()
    pdf.savefig( fig )
    PLOT_NUMBER += 1
    return

def plot_parameter_estimation_by_completion(env, information, n_agents, n_tasks, max_iteration=200, n_experiments=100, mode='gt'):
    print('Plotting parameter_estimation by completion...')
    global PLOT_NUMBER, COLOR, pdf

    fig, axs= plt.subplots(nrows = 2, ncols=2, figsize=(15,15))
    ax_r, ax_a, ax_l, ax_p = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

    x = np.linspace(0,1,n_tasks+1)
    y_radius = {'AGA':np.zeros(len(x)),'ABU': np.zeros(len(x)), 'OEATA': np.zeros(len(x))}
    y_angle = {'AGA':np.zeros(len(x)),'ABU': np.zeros(len(x)), 'OEATA': np.zeros(len(x))}
    if env == 'LevelForagingEnv':
        y_level = {'AGA':np.zeros(len(x)),'ABU': np.zeros(len(x)), 'OEATA': np.zeros(len(x))}
    y_norm_factor = {'AGA':np.zeros(len(x)),'ABU': np.zeros(len(x)), 'OEATA': np.zeros(len(x))}

    for est in ['AGA','ABU','OEATA']:
        for n in range(n_experiments):
            for a in range(n_agents-1):
                for i in range(max_iteration):
                    if i < len(information[est]['actual_radius'][n]):
                        completion = int(information[est]['completions'][n][i]*n_tasks)
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

                        y_radius[est][completion] += abs(act_radius - information[est]['estimated_radius'][n][i][a][agent_type])
                        y_angle[est][completion] += abs(act_angle - information[est]['estimated_angle'][n][i][a][agent_type])
                        if env == 'LevelForagingEnv':
                            y_level[est][completion] += abs(act_level - information[est]['estimated_level'][n][i][a][agent_type])
                        y_norm_factor[est][completion] += 1
                    else:
                        completion = int(information[est]['completions'][n][-1]*n_tasks)
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

                        y_radius[est][completion] += abs(act_radius - information[est]['estimated_radius'][n][-1][a][agent_type])
                        y_angle[est][completion] += abs(act_angle - information[est]['estimated_angle'][n][-1][a][agent_type])
                        if env == 'LevelForagingEnv':
                            y_level[est][completion] += abs(act_level - information[est]['estimated_level'][n][-1][a][agent_type])
                        y_norm_factor[est][completion] += 1

        for i in range(len(y_radius[est])):
            y_radius[est][i] /= y_norm_factor[est][i]
            y_angle[est][i] /= y_norm_factor[est][i]
            if env == 'LevelForagingEnv':
                y_level[est][i] /= y_norm_factor[est][i]

        ax_r.plot(x,y_radius[est],color=COLOR[est])
        ax_a.plot(x,y_angle[est],color=COLOR[est])
        if env == 'LevelForagingEnv':
            ax_l.plot(x,y_level[est],color=COLOR[est])
            ax_p.plot(x,(y_radius[est] + y_angle[est] + y_level[est])/3,color=COLOR[est])
        else:
            ax_p.plot(x,(y_radius[est] + y_angle[est])/2,color=COLOR[est])

    #plt.show()
    parameter_ax = [ax_r, ax_a, ax_l, ax_p]
    parameter_title = ['Radius', 'Angle', 'Level', 'General']
    for i in range(len(parameter_ax)):
        parameter_ax[i].set_title(parameter_title[i])
        parameter_ax[i].set(xlabel="Iteration",ylabel='Error')
    pdf.savefig( fig )
    PLOT_NUMBER += 1
    return
#####
# PLOT SCRIPT
#####
mode = ''
env = 'LevelForagingEnv' # LevelForagingEnv, CaptureEnv
n_experiments = 1

"""for na in [5,7,10]:
    for nt in [20,25,30]:
        for d in [20,25,30]:
            setting = [na, nt, d]"""
for setting in [[5,10,10]]:
    n_agents, n_tasks, dim = setting[0], setting[1], setting[2]
    info = collect_information(n_agents,n_tasks,dim, n_experiments,env,mode)

    pdf = backend_pdf.PdfPages("results_"+mode+env+"_a"+str(n_agents)+"_i"+str(n_tasks)+"_d"+str(dim)+".pdf")

    max_iteration = plot_performance(info)
    #plot_number_of_completed_tasks(info)
    plot_completion(info,max_iteration)
    plot_type_estimation_by_iteration(info, n_agents, max_iteration, n_experiments)
    plot_type_estimation_by_completion(info, n_agents, n_tasks, max_iteration, n_experiments)
    plot_parameter_estimation_by_iteration(env, info, n_agents, max_iteration, n_experiments, mode='gt')
    plot_parameter_estimation_by_completion(env, info, n_agents, n_tasks, max_iteration, n_experiments, mode='gt')

    pdf.close()