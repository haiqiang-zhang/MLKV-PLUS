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
record_sizes = [5000, 10000,50000, 100000, 500000, 1000000, 5000000, 10000000]

gds_reader = [16666, 15693.198097713463, 15468.372241564235, 15051.40526760442, 3283.442027775461, 3070.037924219002, 2624.278131711851, 749.3531674991636]
no_gds_reader_no_pagecache = [16021, 16409.79658856916, 13807.336690835413, 15209.545975858173, 3623.995011497227, 3071.484223458151, 1452.3246431438113, 1874.0589081718445]
no_gds_reader_with_pagecache = [13112, 13889.562501804532, 13598.305719425247, 13105.309990892569, 11360.319702807783, 9474.640183768908, 7083.246253497622, 7465.888949912377]


# Create academic-style plot
fig, ax = plt.subplots()

# Plot the data



ax.plot(record_sizes, no_gds_reader_no_pagecache, marker='s', linestyle='-', 
        label='Normal Reader / No Page Cache', color='#404040', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#404040')

ax.plot(record_sizes, no_gds_reader_with_pagecache, marker='^', linestyle='-', 
        label='Normal Reader / With Page Cache', color='#808080', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#808080')

ax.plot(record_sizes, gds_reader, marker='o', linestyle='-', 
        label='GDS Compat Reader / No Page Cache', color='#8B0000', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#8B0000')


# Configure axes
ax.set_xlabel('Record Size', fontweight='bold')
ax.set_ylabel('Throughput (ops/s)', fontweight='bold')

# Set logarithmic scale for x-axis
ax.set_xscale('log')

# Format x-axis labels with K/M suffix
ax.set_xticks(record_sizes)
ax.set_xticklabels(['5K', '10K', '50K', '100K', '500K', '1M', '5M', '10M'])

# Add grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Format y-axis to show values in scientific notation or with K/M suffix
ax.ticklabel_format(axis='y', style='plain')
y_ticks = ax.get_yticks()
# ax.set_yticklabels([f'{int(y/1000000)}M' if y >= 1000000 else f'{int(y)}' for y in y_ticks])

# Add legend
ax.legend(loc='lower left')

# Tight layout for better spacing
plt.tight_layout()


plt.savefig('gds_reader.pdf', bbox_inches='tight')
plt.savefig('gds_reader.png', bbox_inches='tight', dpi=1000)
