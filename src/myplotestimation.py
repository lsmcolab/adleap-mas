import matplotlib.pyplot as plt
import numpy as np

def plot_type_estimation_by_completion(information, n_agents, n_tasks, estimation_methods, max_iteration=200,\
                                                        plot_number = 0, color = None, pdf = None, title=False):

    print('Plotting type_estimation by completion...')

    # 1. Initialising the figure
    fig = plt.figure(plot_number, figsize=(8,6))

    # 2. Formating data
    x = np.linspace(0,1,n_tasks+1)
    y = {}
    y_norm_factor = {}

    for est in estimation_methods:
        y[est] = np.zeros(len(x))
        y_norm_factor[est] = np.zeros(len(x))
        for n in range(len(information[est]['actual_type'])):
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

        # 3. Preparing 
        for i in range(len(y[est])):
            y[est][i] /= y_norm_factor[est][i]
        plt.plot(x,y[est],color=color[est])

    # 4. Setting plot parameters
    if title:
        plt.title("Type Estimation by Completion")
    plt.xlabel("Completion")
    plt.ylabel("Error")

    # 5. Showing/Saving plot
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()

def plot_parameter_estimation_by_iteration(env, information, n_agents, estimation_methods, max_iteration=200, mode='gt',\
                                                        plot_number = 0, color = None, pdf = None, title=False):

    print('Plotting parameter_estimation by iteration...')

    # 1. Initialising the figure
    fig, axs= plt.subplots(nrows = 2, ncols=2, figsize=(15,15))
    ax_r, ax_a, ax_l, ax_p = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

    # 2. Formating data
    x = range(max_iteration)
    y_radius = {}
    y_angle = {}
    y_level = {}

    for est in estimation_methods:
        y_radius[est] = np.zeros(max_iteration)
        y_angle[est] = np.zeros(max_iteration)
        if env == 'LevelForagingEnv':
            y_level[est] = np.zeros(max_iteration)

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

        # 3. Preparing 
        y_radius[est] /= (len(information[est]['actual_type'])*(n_agents-1))
        y_angle[est] /= (len(information[est]['actual_type'])*(n_agents-1))
        if env == 'LevelForagingEnv':
            y_level[est] /= len(information[est]['actual_type'])*(n_agents-1)

        ax_r.plot(x,y_radius[est],color=color[est])
        ax_a.plot(x,y_angle[est],color=color[est])
        if env == 'LevelForagingEnv':
            ax_l.plot(x,y_level[est],color=color[est])
            ax_p.plot(x,(y_radius[est] + y_angle[est] + y_level[est])/3,color=color[est])
        else:
            ax_p.plot(x,(y_radius[est] + y_angle[est])/2,color=color[est])

    # 4. Setting plot parameters
    parameter_ax = [ax_r, ax_a, ax_l, ax_p]
    parameter_title = ['Radius', 'Angle', 'Level', 'General']
    for i in range(len(parameter_ax)):
        parameter_ax[i].set_title(parameter_title[i])
        parameter_ax[i].set(xlabel="Iteration",ylabel='Error')

    # 5. Showing/Saving plot
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()

def plot_parameter_estimation_by_completion(env, information, n_agents, n_tasks, estimation_methods, max_iteration=200, mode='gt',\
                                                        plot_number = 0, color = None, pdf = None, title=False):
    print('Plotting parameter_estimation by completion...')

    # 1. Initialising the figure
    fig, axs= plt.subplots(nrows = 2, ncols=2, figsize=(15,15))
    ax_r, ax_a, ax_l, ax_p = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

    # 2. Formating data
    x = np.linspace(0,1,n_tasks+1)
    y_radius = {}
    y_angle = {}
    y_level = {}
    y_norm_factor = {}

    for est in estimation_methods:
        y_radius[est] = np.zeros(len(x))
        y_angle[est] = np.zeros(len(x))
        y_norm_factor[est] = np.zeros(len(x))
        if env == 'LevelForagingEnv':
            y_level[est] = np.zeros(len(x))

        for n in range(len(information[est]['actual_type'])):
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

        # 3. Preparing 
        for i in range(len(y_radius[est])):
            y_radius[est][i] /= y_norm_factor[est][i]
            y_angle[est][i] /= y_norm_factor[est][i]
            if env == 'LevelForagingEnv':
                y_level[est][i] /= y_norm_factor[est][i]

        ax_r.plot(x,y_radius[est],color=color[est])
        ax_a.plot(x,y_angle[est],color=color[est])
        if env == 'LevelForagingEnv':
            ax_l.plot(x,y_level[est],color=color[est])
            ax_p.plot(x,(y_radius[est] + y_angle[est] + y_level[est])/3,color=color[est])
        else:
            ax_p.plot(x,(y_radius[est] + y_angle[est])/2,color=color[est])

    # 4. Setting plot parameters
    parameter_ax = [ax_r, ax_a, ax_l, ax_p]
    parameter_title = ['Radius', 'Angle', 'Level', 'General']
    for i in range(len(parameter_ax)):
        parameter_ax[i].set_title(parameter_title[i])
        parameter_ax[i].set(xlabel="Completion (%)",ylabel='Error')

    # 5. Showing/Saving plot
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()