import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.dpi'] = 150

import pandas as pd
dim_h_Y = pd.read_csv("Y_size.csv").to_numpy().flatten()[1:][::-1][::3]
batch_size = pd.read_csv("batch_size.csv").to_numpy().flatten()[1:][::-1][::3]
err_Y = pd.read_csv("Y_abs.csv").to_numpy().flatten()[1:][::-1][::3]
err_Z = pd.read_csv("Z_abs.csv").to_numpy().flatten()[1:][::-1][::3]
training_time = pd.read_csv("training_time.csv").to_numpy().flatten()[1:][::-1][::3]

unique_Y = np.unique(dim_h_Y)
unique_batch = np.unique(batch_size)

fig = plt.figure(figsize=(16,5))

ax1 = plt.subplot(1, 3, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_batch)))
markers = ['o', 's', '^', 'd']

for i, bs in enumerate(unique_batch):
    mask = batch_size == bs
    ax1.semilogy(dim_h_Y[mask], err_Y[mask], marker=markers[i], color=colors[i],
                 label=f'Batch={bs}', linewidth=2, markersize=8)

ax1.set_xlabel('Hidden Layer Size of the Y-Network ($h_Y$)', fontweight='bold')
ax1.set_ylabel('MSE Y', fontweight='bold')
#ax1.set_title('(a) Error Y vs Network Width', fontweight='bold')
ax1.legend(framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(unique_Y)

# =================== PLOT 2: Error Z vs Architecture ===================
ax2 = plt.subplot(1, 3, 2)
for i, bs in enumerate(unique_batch):
    mask = batch_size == bs
    ax2.semilogy(dim_h_Y[mask], err_Z[mask], marker=markers[i], color=colors[i],
                 label=f'Batch={bs}', linewidth=2, markersize=8)

ax2.set_xlabel('Hidden Layer Size of the Y-Network ($h_Y$)', fontweight='bold')
ax2.set_ylabel('MSE Z', fontweight='bold')
#ax2.set_title('(b) Error Z vs Network Width', fontweight='bold')
ax2.legend(framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(unique_Y)

# =================== PLOT 3: Training Time vs Architecture ===================
ax3 = plt.subplot(1, 3, 3)
for i, bs in enumerate(unique_batch):
    mask = batch_size == bs
    ax3.plot(dim_h_Y[mask], training_time[mask], marker=markers[i], color=colors[i],
             label=f'Batch={bs}', linewidth=2, markersize=8)

ax3.set_xlabel('Hidden Layer Size of the Y-Network ($h_Y$)', fontweight='bold')
ax3.set_ylabel('Training Time (min)', fontweight='bold')
#ax3.set_title('(c) Training Time vs Network Width', fontweight='bold')
ax3.legend(framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xticks(unique_Y)

plt.tight_layout()
plt.savefig('sensitivity_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('sensitivity_analysi.png', dpi=300, bbox_inches='tight')
plt.show()
# ====