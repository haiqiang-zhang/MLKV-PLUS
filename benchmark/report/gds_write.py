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
batch_sizes = [1024, 4096, 16384, 65536, 262144]

gds_writer = [200870, 219579, 255200.4531334261, 308945.69193585627, 283460.8127441842]
no_gds_writer = [223784.5443437784, 224741.03704098114, 326329.5612041819, 344855.769745066, 300253.02623967215]

# Create academic-style plot
fig, ax = plt.subplots()

# Plot the data



ax.plot(batch_sizes, gds_writer, marker='o', linestyle='-', 
        label='G-WAL enabled', color='#8B0000', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#8B0000')

ax.plot(batch_sizes, no_gds_writer, marker='^', linestyle='-', 
        label='Normal WAL Writer', color='#404040', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#404040')


# Configure axes
ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Throughput (ops/s)', fontweight='bold')

# Set logarithmic scale for x-axis
ax.set_xscale('log', base=2)


# Add grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Format y-axis to show values in scientific notation or with K/M suffix
ax.ticklabel_format(axis='y', style='plain')
y_ticks = ax.get_yticks()
# ax.set_yticklabels([f'{int(y/1000000)}M' if y >= 1000000 else f'{int(y)}' for y in y_ticks])

# Add legend
ax.legend(loc='lower right')

# Tight layout for better spacing
plt.tight_layout()


plt.savefig('gds_writer.pdf', bbox_inches='tight')
plt.savefig('gds_writer.png', bbox_inches='tight', dpi=1000)