from ast import literal_eval
from matplotlib.backends import backend_pdf
from myplotlib import *
import numpy as np

#####
# CUSTOMISATION
#####
PLOT_NUMBER = 0
PLOT_SETTINGS =[[7,30,30]]

COLOR = {'AGA':'b','ABU':'g','SOEATA':'r','POMCP':'y'}
MARKER = {'AGA':'^','ABU':'v','SOEATA':'o','POMCP':'s'}

REMOVE = False
MIN_ITERATIONS = np.inf

mode = 'Respawn_'
env = 'LevelForagingEnv' # LevelForagingEnv, CaptureEnv
n_experiments = 20
estimation_methods = ['AGA', 'ABU','POMCP','SOEATA']

#####
# COLLECTING INFORMATION
#####
def collect_information(a, i, d, n_experiments, env, estimation_methods, mode=""):
    print('Collecting information...',a,i,d,env,mode)
    global MIN_ITERATIONS, REMOVE

    information = {}
    for est in estimation_methods:
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

            with open('./results/'+mode+env+'_a'+str(a)+'_i'+str(i)+\
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
            
            if REMOVE and line_counter < MIN_ITERATIONS:
                information[est]['iterations'].pop()
                information[est]['completions'].pop()
                information[est]['actual_radius'].pop()
                information[est]['actual_angle'].pop()
                information[est]['actual_level'].pop()
                information[est]['actual_type'].pop()
                information[est]['estimated_radius'].pop()
                information[est]['estimated_angle'].pop()
                information[est]['estimated_level'].pop()
                information[est]['type_probabilities'].pop()
            elif line_counter <= 1:
                information[est]['iterations'].pop()
                information[est]['completions'].pop()
                information[est]['actual_radius'].pop()
                information[est]['actual_angle'].pop()
                information[est]['actual_level'].pop()
                information[est]['actual_type'].pop()
                information[est]['estimated_radius'].pop()
                information[est]['estimated_angle'].pop()
                information[est]['estimated_level'].pop()
                information[est]['type_probabilities'].pop()
            elif (line_counter - 2) < MIN_ITERATIONS:
                MIN_ITERATIONS = line_counter - 2

    return information

def get_max_iteration(information, estimation_methods):
    max_iterations = []
    for est in estimation_methods:
        max_iterations.append(max([information[est]['iterations'][i][-1] for i in range(len(information[est]['iterations']))]))
    return max(max_iterations)

#####
# PLOT SCRIPT
#####
# SINGLE WAY
"""
for setting in PLOT_SETTINGS:
    if not REMOVE:
        MIN_ITERATIONS = np.inf
        
    # 1. Collecting info
    n_agents, n_tasks, dim = setting[0], setting[1], setting[2]
    information = collect_information(n_agents,n_tasks,dim, n_experiments,env,estimation_methods,mode)
    max_iteration = get_max_iteration(information,estimation_methods)

    #pdf = backend_pdf.PdfPages("./plots/results_"+mode+env+"_a"+str(n_agents)+"_i"+str(n_tasks)+"_d"+str(dim)+".pdf")

    # 2. Starting to plot
    # A. COMPLETED TASKS
    pdf = backend_pdf.PdfPages("./plots/CompletedTasks_"+mode+env+"_a"+str(n_agents)+"_i"+str(n_tasks)+"_d"+str(dim)+".pdf")
    plot_number_of_completed_tasks(information,estimation_methods,\
                                    plot_number=PLOT_NUMBER,color=COLOR,pdf=pdf)
    pdf.close()
    PLOT_NUMBER += 1

    # B. PARAMETER ESTIMATION
    pdf = backend_pdf.PdfPages("./plots/ParameterEstimation_"+mode+env+"_a"+str(n_agents)+"_i"+str(n_tasks)+"_d"+str(dim)+".pdf")
    kruskal_result = plot_parameter(env, information, n_agents, estimation_methods,\
                     max_iteration,plot_number=PLOT_NUMBER,color=COLOR, marker=MARKER,pdf=pdf)
    pdf.close()
    PLOT_NUMBER += 1
    
    pdf = backend_pdf.PdfPages("./plots/ParameterKruskal_"+mode+env+"_a"+str(n_agents)+"_i"+str(n_tasks)+"_d"+str(dim)+".pdf")
    plot_pvalues(kruskal_result,estimation_methods,max_iteration,COLOR,MARKER,pdf)
    pdf.close()
    PLOT_NUMBER += 1

    # C. TYPE ESTIMATION
    pdf = backend_pdf.PdfPages("./plots/TypeEstimation_"+mode+env+"_a"+str(n_agents)+"_i"+str(n_tasks)+"_d"+str(dim)+".pdf")
    plot_type_estimation_by_iteration(information, n_agents, estimation_methods,\
                                         max_iteration,plot_number=PLOT_NUMBER,color=COLOR, marker=MARKER, pdf=pdf)
    pdf.close()
    PLOT_NUMBER += 1

    pdf = backend_pdf.PdfPages("./plots/TypeKruskal_"+mode+env+"_a"+str(n_agents)+"_i"+str(n_tasks)+"_d"+str(dim)+".pdf")
    plot_pvalues(kruskal_result,estimation_methods,max_iteration,COLOR,MARKER,pdf)
    pdf.close()
    PLOT_NUMBER += 1

    # 3. Saving it
    #pdf.close()
"""

# MULTIWAY
# N AGENTS
"""
PLOT_SETTINGS = [[5,30,30],
                [7,30,30],
                [10,30,30],
                [12,30,30],
                [15,30,30]]

information, max_iteration = [], []
for setting in PLOT_SETTINGS:
    if not REMOVE:
        MIN_ITERATIONS = np.inf
        
    # 1. Collecting info
    n_agents, n_tasks, dim = setting[0], setting[1], setting[2]
    information.append(collect_information(n_agents,n_tasks,dim, n_experiments,env,estimation_methods,mode))
    max_iteration.append(get_max_iteration(information[-1],estimation_methods))

# 2. Starting to plot
pdf = backend_pdf.PdfPages("./plots/MultAgents_"+mode+env+"_i"+str(n_tasks)+"_d"+str(dim)+".pdf")
plot_multparameter(env, information, PLOT_SETTINGS, 0, estimation_methods, max_iteration,\
                     plot_number=PLOT_NUMBER,color=COLOR, marker=MARKER,pdf=pdf)
pdf.close()
PLOT_NUMBER += 1
"""

# N TASKS

PLOT_SETTINGS = [[7,30,20],
                [7,30,25],
                [7,30,30],
                [7,30,35],
                [7,30,40]]
                
information, max_iteration = [], []
for setting in PLOT_SETTINGS:
    if not REMOVE:
        MIN_ITERATIONS = np.inf
        
    # 1. Collecting info
    n_agents, n_tasks, dim = setting[0], setting[1], setting[2]
    information.append(collect_information(n_agents,n_tasks,dim, n_experiments,env,estimation_methods,mode))
    max_iteration.append(get_max_iteration(information[-1],estimation_methods))

# 2. Starting to plot
pdf = backend_pdf.PdfPages("./plots/MultTasks_"+mode+env+"_a"+str(n_agents)+"_i"+str(n_tasks)+".pdf")
plot_multparameter(env, information, PLOT_SETTINGS, 1, estimation_methods, max_iteration,\
                     plot_number=PLOT_NUMBER,color=COLOR, marker=MARKER,pdf=pdf)
pdf.close()
PLOT_NUMBER += 1


# N DIMS
"""
PLOT_SETTINGS = [[7,30,20],
                [7,30,25],
                [7,30,30],
                [7,30,35],
                [7,30,40]]
                
information, max_iteration = [], []
for setting in PLOT_SETTINGS:
    if not REMOVE:
        MIN_ITERATIONS = np.inf
        
    # 1. Collecting info
    n_agents, n_tasks, dim = setting[0], setting[1], setting[2]
    information.append(collect_information(n_agents,n_tasks,dim, n_experiments,env,estimation_methods,mode))
    max_iteration.append(get_max_iteration(information[-1],estimation_methods))

# 2. Starting to plot
pdf = backend_pdf.PdfPages("./plots/MultDims_"+mode+env+"_a"+str(n_agents)+"_i"+str(n_tasks)+".pdf")
plot_multparameter(env, information, PLOT_SETTINGS, 2, estimation_methods, max_iteration,\
                     plot_number=PLOT_NUMBER,color=COLOR, marker=MARKER,pdf=pdf)
pdf.close()
PLOT_NUMBER += 1
"""