import csv
import pandas
import matplotlib.pyplot as plt

textDF=pandas.read_csv('results/train_ratio.csv')
labels=[]

print(textDF['train ratio'])

fig, ax =  plt.subplots(figsize=(4,5))

plt.plot(textDF['train ratio'], textDF['Fmac for best baseline'], 'or-',markersize=3)
plt.plot(textDF['train ratio'], textDF['Fmac for best method'], 'oy-',markersize=3)
ax.legend(['$F_{mac}$ for the best baseline', '$F_{mac}$ for our best method'],loc=(1,0), prop={'size': 12},bbox_to_anchor=(0.012,0.845))

ax.set_xlabel("Training data %",fontsize=12)
ax.set_xticks([50,70,85,100])
ax.set_xticklabels([50,70,85,100], rotation=0, fontsize=11)
ax.set_yticks([0.51,0.52,0.53,0.54,0.55,0.56,0.57])
ax.set_yticklabels([0.51,0.52,0.53,0.54,0.55,0.56,0.57],rotation=0, fontsize=11)

plt.savefig('results/train_ratio.pdf',bbox_extra_artists=(ax,), bbox_inches='tight')

