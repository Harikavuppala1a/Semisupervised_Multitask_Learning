import csv
import pandas
import matplotlib.pyplot as plt
from loadPreProc import *
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from numpy import arange

textDF=pandas.read_csv('results/classwise_performance.csv')

fig, ax = plt.subplots(figsize=(10,5))
textDF['best_baseline_F score'].plot.bar(width=0.3,  ylim=[0.1, 0.9], xlim=[0,22], position=2.0, color="blue", ax=ax, alpha=1)
textDF['best_proposed_F score'].plot.bar(width=0.3, position=1.0, ylim=[0, 0.9], xlim=[0,22],color="violet", ax=ax, alpha=1)
ax.set_facecolor("white")
ax.set_xticklabels(range(0,NUM_CLASSES), rotation=0, fontsize=7)
ax.set_yticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],rotation=0, fontsize=7)
ax.set_xlabel("Label IDs",fontsize=7)

for i in range(NUM_CLASSES):
    plt.text(x = i-0.6 , y = max(textDF['best_baseline_F score'][i],textDF['best_proposed_F score'][i])+.0125, s = ("%.1f" % textDF["train cov"][i]), size = 7,rotation=0, bbox=dict(facecolor='lightgray', edgecolor='none',alpha=1, pad=1))

tr_c = mpatches.Patch(color='lightgray', label='Label coverage %')
best_base = mpatches.Patch(color='blue', label='F score for the best baseline')
best_proposed = mpatches.Patch(color='violet', label='F score for our best method')
ax.legend(handles=[tr_c,best_base,best_proposed],loc="upper right", prop={'size': 7},bbox_to_anchor=(1.0035,1.007))
plt.savefig('results/class_wise_performance.pdf',bbox_extra_artists=(ax,), bbox_inches='tight')

