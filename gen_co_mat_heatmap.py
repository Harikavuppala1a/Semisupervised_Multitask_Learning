import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from loadPreProc import *

with open("saved/norm_corr_mat.pickle", 'rb') as corr_data:
    co_mat = pickle.load(corr_data)

flat=co_mat.flatten()
flat.sort()

Categories = ["%d. %s" % (i,FOR_LMAP[i]) for i in range(NUM_CLASSES)]
heat_map = sns.heatmap(co_mat, cmap= 'tab20b', linewidths=.5)

heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=0, fontsize=8)
heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0, fontsize=8)
plt.xlabel("Label IDs",fontsize=8)
plt.ylabel("Label IDs",fontsize=8)
extra = Rectangle((0, 0), 0.5, 0.5, fc="w", fill=False, linewidth=0)
heat_map.legend([extra]*NUM_CLASSES, Categories, loc = (1,1), fontsize='small', framealpha=0, handlelength=0, handletextpad=0, bbox_to_anchor=(1.207,-0.050))
cbar = heat_map.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)
plt.savefig('corr.pdf', bbox_extra_artists=(heat_map,), bbox_inches='tight')

