import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

# Batch sizes for x-axis
batch_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]

gds_multiget_with_pagecache = [37278.00696572323, 37486.535790089834, 33826.594507251124, 31769.747770568156, 32552.304019474537, 32593.649881824527, 43130.07593017624, 58466.67732895733, 59666.23556672029, 70390.90911395168, 78194.32732022433, 72905.117026290085]
nogds_multiget = [29256.435018595035, 24244.69109020999, 23669.380855064286, 25929.807698465644, 38048.812307148255, 45439.857408099895, 64091.9548889385, 65856.83058003969, 69522.81721591974, 76108.27251153758, 78238.12869441882, 85585.464139516]


# Create academic-style plot
fig, ax = plt.subplots()

# Plot the data

ax.plot(batch_sizes, nogds_multiget, marker='s', linestyle='-', 
        label='Normal Multiget / With Page Cache', color='#404040', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#404040')

ax.plot(batch_sizes, gds_multiget_with_pagecache, marker='o', linestyle='-', 
        label='GDS Multiget / With G-Page Cache', color='#8B0000', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#8B0000')

# Set logarithmic scale for x-axis (batch sizes span multiple orders of magnitude)
ax.set_xscale('log', base=2)

# Configure axes
ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Throughput (ops/s)', fontweight='bold')

# Add grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Format y-axis to show values in scientific notation or with K/M suffix
ax.ticklabel_format(axis='y', style='plain')
y_ticks = ax.get_yticks()
ax.set_yticklabels([f'{int(y/1000000)}M' if y >= 1000000 else f'{int(y)}' for y in y_ticks])

# Add legend
ax.legend(loc='upper left')

# Tight layout for better spacing
plt.tight_layout()


plt.savefig('gds_multiget.pdf', bbox_inches='tight')
plt.savefig('gds_multiget.png', bbox_inches='tight', dpi=1000)
