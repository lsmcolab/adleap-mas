import matplotlib.pyplot as plt
import numpy as np

def plot_mean_iterations(information, estimation_methods,\
             plot_number = 0, color = None, pdf = None, title=False):
             
    print('Plotting mean iterations...')

    # 1. Initialising the figure
    fig = plt.figure(plot_number, figsize=(8,6))

    # 2. Formating data
    data, colors_ = {}, []
    for est in estimation_methods:
        mean_iteration = np.mean([information[est]['iterations'][i][-1] for i in range(len(information[est]['iterations']))])

        data[est] = mean_iteration
        colors_.append(color[est])

    # 3. Preparing 
    X_estimation_methods = list(data.keys())
    Y_mean_iteration = list(data.values())

    plt.bar(X_estimation_methods,Y_mean_iteration,color=colors_)

    # 4. Setting plot parameters
    if title:
        plt.title("Performance")
    plt.xlabel("Estimation Methods")
    plt.ylabel("Mean Iterations")

    # 5. Showing/Saving plot
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()

def plot_completion(information, estimation_methods, max_iterations=200,\
             plot_number = 0, color = None, pdf = None, title=False):

    print('Plotting performance...')

    # 1. Initialising the figure
    fig = plt.figure(plot_number, figsize=(8,6))

    # 2. Formating data
    data = {}
    for est in estimation_methods:
        data[est] = np.zeros(max_iterations)
        for nexp in range(len(information[est]['completions'])):
            for i in range(max_iterations):
                if len(information[est]['completions'][nexp]) - 1 < i:
                    data[est][i] += information[est]['completions'][nexp][-1]
                else:
                    data[est][i] += information[est]['completions'][nexp][i]
        data[est] /= len(information[est]['completions'])
        plt.plot(range(max_iterations),data[est],color=color[est])

    # 3. Setting plot parameters
    if title:
        plt.title("Completion")
    plt.xlabel("Estimation Methods")
    plt.ylabel("Completion (%)")

    # 4. Showing/Saving plot
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
    plt.close()