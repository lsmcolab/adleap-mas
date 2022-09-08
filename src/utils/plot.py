import matplotlib.pyplot as plt
from utils.stats import *

NEXP = 50

FIG_COUNTER = 0
FIGSIZE = (6.4,5.4)

FONTSIZE = 28
LEGEND_FONTSIZE = 22
FONT_DICT = {
        'weight': 'bold',
        'size': 26,
        }
TICK_FONTSIZE = 20

COLOR   = ['gold','red','blue','orange','purple']
            
MARKER_SIZE = 24
MARK_EVERY = 20
MARKER = ['o','s','^','p','X']

LINEWIDTH = 8
LINESTYLE = ['--','-.','-',':','-.']

def ablation_study(results,label,marker,savepath='.plots/'):
    plt.figure(num=FIG_COUNTER,figsize=FIGSIZE)
    for i in range(len(results)):
        plt.plot(get_cummulative_reward(results[i]),color=COLOR[i],marker=marker[i], label=label[i],
            linewidth=LINEWIDTH,linestyle=LINESTYLE[i],markersize=MARKER_SIZE,markevery=MARK_EVERY,markeredgecolor='black')
    #plt.legend(loc='best',ncol=1,fontsize=LEGEND_FONTSIZE,edgecolor='black')
    plt.xlabel('Iteration',fontdict=FONT_DICT)
    plt.xticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.ylabel('Cumulative reward',fontdict=FONT_DICT)
    plt.yticks(fontsize=TICK_FONTSIZE,rotation=45)
    plt.tight_layout()
    plt.savefig(savepath+'ablation_study.pdf')

    FIG_COUNTER += 1