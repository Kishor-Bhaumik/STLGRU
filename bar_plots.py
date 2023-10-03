'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create data
x = [1, 3, 6, 9, 12, 15, 18, 21, 24]
task3_1 = [4.92, 4.45, 3.99, 3.90, 3.80, 3.82, 3.85, 3.89, 3.91]
task2_1 = [5.57, 5.12, 4.92, 4.69, 4.45, 4.55, 4.56, 4.57, 4.58]
task3_2 = [30.78, 27.98, 25.83, 22.90, 20.85, 21.82, 20.81, 20.89, 20.90]
task2_2 = [35.43, 33.29, 30.91, 26.81, 24.12, 23.11, 23.12, 23.15, 23.14]
task3_3 = [6.12, 5.85, 5.05, 4.18, 4.01, 3.59, 3.16, 3.17, 3.18]
task2_3 = [7.50, 7.12, 6.15, 5.87, 5.18, 4.97, 4.15, 4.18, 4.14]

# Set Seaborn style
sns.set_style("whitegrid")

# Create a figure with 3 subplots horizontally
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Plot for the first set of data
sns.lineplot(x=x, y=task2_1, ax=axes[0], marker="^", label="Task = 2")
sns.lineplot(x=x, y=task3_1, ax=axes[0], marker="*", label="Task = 3")
#axes[0].xaxis.grid(False)
axes[0].set_title("METR-LA", fontsize = 25)
axes[0].set_xlabel("Epochs", fontsize = 25)
axes[0].set_ylabel("MAE", fontsize = 25)
axes[0].legend(fontsize='large') 

# Plot for the second set of data
sns.lineplot(x=x, y=task2_2, ax=axes[1], marker="^", label="Task = 2")
sns.lineplot(x=x, y=task3_2, ax=axes[1], marker="*", label="Task = 3")
#axes[1].xaxis.grid(False)
axes[1].set_title("PEMSD4", fontsize = 25)
axes[1].set_xlabel("Epochs", fontsize = 25)
axes[1].set_ylabel("MAE", fontsize = 25)
axes[1].legend(fontsize='large') 

# Plot for the third set of data
sns.lineplot(x=x, y=task2_3, ax=axes[2], marker="^", label="Task = 2")
sns.lineplot(x=x, y=task3_3, ax=axes[2], marker="*", label="Task = 3")
#axes[2].xaxis.grid(False)
axes[2].set_title("Didi-Chengdu", fontsize = 25)
axes[2].set_xlabel("Epochs", fontsize = 25)
axes[2].set_ylabel("MAE", fontsize = 25)
axes[2].legend(fontsize='large') 

# Display the plot
plt.tight_layout()
plt.savefig("epochsVsscore.pdf")
plt.show()

'''


'''

import matplotlib.pyplot as plt
import numpy as np
  
# create data
x = [1,3,6,9,12,15,18,21, 24]
task3 = [4.92, 4.45, 3.99,3.90, 3.80,3.82,3.85, 3.89, 3.91, ]#3.90]
task2 = [5.57, 5.12, 4.92, 4.69, 4.45, 4.55, 4.56, 4.57, 4.58]
# plot lines

plt.plot(x, task2, label = "Task = 2 ")
plt.plot(x, task3, label = "Task = 3")
plt.legend()
plt.show()


task_2 = [35.43, 33.29, 30.91, 26.81, 24.12, 23.11, 23.12, 23.15, 23.14]
task_3 = [30.78, 27.98, 25.83, 22.90, 20.85, 21.82, 20.81, 20.89, 20.90]

plt.plot(x, task_2, label = "Task = 2 ")
plt.plot(x, task_3, label = "Task = 3")
plt.legend()
plt.show()


task_2 = [7.50, 7.12, 6.15, 5.87, 5.18, 4.97, 4.15, 4.18, 4.14,]
task_3 = [6.12, 5.85, 5.05, 4.18, 4.01, 3.59, 3.16, 3.17, 3.18, ]


plt.plot(x, task_2, label = "Task = 2 ")
plt.plot(x, task_3, label = "Task = 3")
plt.legend()
plt.show()
'''

'''
import numpy as np
import matplotlib.pyplot as plt
import os
print(os.getcwd())
# Sample data
models = ['10', '15', '20', '40', '60']
mem_scores = [3.87, 3.54, 3.45, 3.51, 3.50]  # Sample RMSE scores
poi_scores = [3.72, 3.65, 3.49, 3.52, 3.61]   # Sample MAE scores
ssmt_scores = [3.42, 3.27, 3.16, 3.25, 3.23]

bar_width = 0.25
index = np.arange(len(models))

fig, ax = plt.subplots(figsize=(12, 7))
bar1 = ax.bar(index, mem_scores, bar_width, label='W/O memory', color='#777777', hatch='/', edgecolor='black')
bar2 = ax.bar(index + bar_width, poi_scores, bar_width, label='W/O MPE', color='#C94845', hatch='.', edgecolor='black')
bar3 = ax.bar(index + 2*bar_width, ssmt_scores, bar_width, label='SSMT', color='#49D845', hatch='*', edgecolor='black')

# Labeling, title, and legends
ax.set_xlabel('Memory Size', fontsize =25 )
ax.set_ylabel('MAE',  fontsize =25 )
ax.set_xticks(index + bar_width)
ax.set_xticklabels(models)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)  # Adjusted this line to position the legend

# Display the plot
plt.tight_layout()

plt.savefig("barplot.pdf")
plt.show()
'''


import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
print(os.getcwd())
fig = plt.figure(figsize=(18, 7))
gs = gridspec.GridSpec(1, 3)
plt.subplots_adjust(wspace=0.6, top=0.85)  # Adjust wspace for space between subplots, and top for space at the top of the figure
#fig.suptitle("Your Title Here", fontsize=20, y=0.98)  # Adjust y value to set title position


# Sample data
models = ['MAE',  'RMSE', 'MAPE']

#PeMS04
gcn_1d_cnn = [24.14, 28.34, 16.45]
gcn_lstm = [23.15, 27.54, 15.13]
gcn_gru = [22.99, 27.01, 14.69]
STLGRU= [21.05, 25.41, 13.87]


bar_width = 0.15
index = np.arange(len(models))
ax0 = fig.add_subplot(gs[0, 0])
#fig, ax = plt.subplots(figsize=(12, 7))
bar1 = ax0.bar(index, gcn_1d_cnn, bar_width, label='GCN+1D CNN', color='#777777', hatch='/', edgecolor='black')
bar2 = ax0.bar(index + bar_width, gcn_lstm,bar_width, label='GCN+lstm', color='#C94845', hatch='.', edgecolor='black')
bar3 = ax0.bar(index + 2*bar_width, gcn_gru, bar_width, label='GCN+GRU', color='#e18e96', hatch='*', edgecolor='black')
bar4 = ax0.bar(index + 3*bar_width, STLGRU, bar_width, label='STLGRU (Ours)', color='#ffff9f', hatch='x', edgecolor='black')
# Labeling, title, and legends
#ax.set_xlabel('Errors', fontsize =25 )
#ax.set_ylabel('Error Scores',  fontsize =25 )
ax0.set_title("PeMSD4")
ax0.set_xticks(index + bar_width)
ax0.set_xticklabels(models,  fontsize=20)
ax0.legend(loc='upper center', bbox_to_anchor=(0.86, 1.01), ncol=1, fontsize=13)  # Adjusted this line to position the legend
ax0.title.set_size(20)
ax0.tick_params(axis='y', labelsize=14) 
# Display the plot
plt.tight_layout()

#plt.savefig("barplot.pdf")


gcn_1d_cnn = [25.14, 34.34, 11.45]
gcn_lstm = [24.15, 33.54, 10.13]
gcn_gru = [24.99, 33.67, 10.69]
STLGRU= [23.06,32.19, 9.12]


bar_width = 0.15
index = np.arange(len(models))
ax1 = fig.add_subplot(gs[0, 1])
#fig, ax = plt.subplots(figsize=(12, 7))
bar1 = ax1.bar(index, gcn_1d_cnn, bar_width, label='GCN+1D CNN', color='#777777', hatch='/', edgecolor='black')
bar2 = ax1.bar(index + bar_width, gcn_lstm,bar_width, label='GCN+lstm', color='#C94845', hatch='.', edgecolor='black')
bar3 = ax1.bar(index + 2*bar_width, gcn_gru, bar_width, label='GCN+GRU', color='#e18e96', hatch='*', edgecolor='black')
bar4 = ax1.bar(index + 3*bar_width, STLGRU, bar_width, label='STLGRU (Ours)', color='#ffff9f', hatch='x', edgecolor='black')
# Labeling, title, and legends
#ax.set_xlabel('Errors', fontsize =25 )
#ax.set_ylabel('Error Scores',  fontsize =25 )
ax1.set_title("PeMSD7")
ax1.set_xticks(index + bar_width)
ax1.set_xticklabels(models,  fontsize=20)
ax1.legend(loc='upper center', bbox_to_anchor=(0.86, 1.01), ncol=1, fontsize=13)  # Adjusted this line to position the legend
ax1.title.set_size(20)
ax1.tick_params(axis='y', labelsize=14) 
# Display the plot
plt.tight_layout()




gcn_1d_cnn = [19.01, 28.59, 12.87]
gcn_lstm = [19.15, 28.54, 11.99]
gcn_gru = [17.87, 27.99, 11.69]
STLGRU= [16.83,26.35, 10.74]



bar_width = 0.15
index = np.arange(len(models))
ax2 = fig.add_subplot(gs[0, 2])
#fig, ax = plt.subplots(figsize=(12, 7))
bar1 = ax2.bar(index, gcn_1d_cnn, bar_width, label='GCN+1D CNN', color='#777777', hatch='/', edgecolor='black')
bar2 = ax2.bar(index + bar_width, gcn_lstm,bar_width, label='GCN+lstm', color='#C94845', hatch='.', edgecolor='black')
bar3 = ax2.bar(index + 2*bar_width, gcn_gru, bar_width, label='GCN+GRU', color='#e18e96', hatch='*', edgecolor='black')
bar4 = ax2.bar(index + 3*bar_width, STLGRU, bar_width, label='STLGRU (Ours)', color='#ffff9f', hatch='x', edgecolor='black')


ax2.set_title("PeMSD8")
ax2.set_xticks(index + bar_width)
ax2.set_xticklabels(models,  fontsize=20)
ax2.legend(loc='upper center', bbox_to_anchor=(0.86, 1.01), ncol=1, fontsize=13)  # Adjusted this line to position the legend
ax2.title.set_size(20)
ax2.tick_params(axis='y', labelsize=14) 
# Display the plot

plt.subplots_adjust(wspace=8)
plt.tight_layout()
plt.savefig("stlgru_barplot.pdf")
plt.show()
