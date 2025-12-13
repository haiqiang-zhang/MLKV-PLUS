import matplotlib.pyplot as plt
import numpy as np

from benchmark.report.dist_multiget import core3_rank0_tp

# Configure matplotlib for academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

# Write operation ratio for x-axis (percentage)
write_ratios = [0, 20, 40, 60, 80, 100]

# Core == 2: Raw data from two ranks

core2_rank1_tp = [1182533.4332174417, 1179493.4564991107, 1158064.4796068212, 1119960.676173258, 1047798.5768158315, 1261593.0254964165]
core2_rank2_tp = [1182524.9920093862, 1179641.8687818672, 1153203.3939544463, 1120161.5825173915, 1047895.8015228763, 999740.1955720279]

# Calculate minimum throughput across ranks for core2 (bottleneck performance)
core2_tp = [min(r1, r2) for r1, r2 in zip(core2_rank1_tp, core2_rank2_tp)]

# Core == 3
core3_rank0_tp = [10615159.385738745, 5854715.287111421, 4257295.112405858, 3325138.116347172, 2721815.135617492, 2825593.971800999]
core3_rank1_tp = [10618793.32163674, 5858199.857408465, 4258084.302036299, 3325752.308097097, 2710287.2091243872, 2263802.5386933535]
core3_rank2_tp = [10614140.11613423,5854625.919057392, 4257458.202267564, 3325196.843651896, 2720646.496408537, 2753114.1229379782]
core3_tp = [min(r0, r1, r2) for r0, r1, r2 in zip(core3_rank0_tp, core3_rank1_tp, core3_rank2_tp)]

# Core == 4
core4_rank0_tp = [53851769.015626274,11864078.139326505, 7586788.127888918, 5232653.984389486, 4157658.3551033004, 4163323.3373331614]
core4_rank1_tp = [53837269.7118188, 11863450.888433058, 7585924.452415196, 5232705.284191191, 4153565.0159817063, 4040222.4335764325]
core4_rank2_tp = [53828311.66806275, 11870529.869975079, 7587833.700018391, 5233093.302644267, 4119860.047552312, 3276991.2220958727]
core4_rank3_tp = [53860833.81033386,11865359.7294705, 7586295.547053751, 5233349.854786192, 4152357.811010388, 3963985.357658018]

core4_tp = [min(r0, r1, r2, r3) for r0, r1, r2, r3 in zip(core4_rank0_tp, core4_rank1_tp, core4_rank2_tp, core4_rank3_tp)]


# Create academic-style plot
fig, ax = plt.subplots()

# Plot the data for different core configurations
ax.plot(write_ratios, core2_tp, marker='o', linestyle='-', 
        label='Core=2', color='#8B0000', markerfacecolor='white', 
        markeredgewidth=2, markeredgecolor='#8B0000')


ax.plot(write_ratios, core3_tp, marker='s', linestyle='-', 
        label='Core=3', color='#00008B', markerfacecolor='white', 
        markeredgewidth=2, markeredgecolor='#00008B')

ax.plot(write_ratios, core4_tp, marker='^', linestyle='-', 
        label='Core=4', color='#404040', markerfacecolor='white', 
        markeredgewidth=2, markeredgecolor='#404040')


# Configure axes labels
ax.set_xlabel('Write Ratio (%)', fontweight='bold')
ax.set_ylabel('Throughput (ops/s)', fontweight='bold')

# Set x-axis limits and ticks to show all write ratios clearly
ax.set_xlim(-5, 105)
ax.set_xticks(write_ratios)

# Add grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Format y-axis to show values in scientific notation or with K/M suffix
ax.ticklabel_format(axis='y', style='plain')
y_ticks = ax.get_yticks()
ax.set_yticklabels([f'{int(y/1000000)}M' if y >= 1000000 else f'{int(y)}' for y in y_ticks])

# Add legend
ax.legend(loc='upper right')

# Tight layout for better spacing
plt.tight_layout()

# Save the figure in high resolution
plt.savefig('dist_write_performance.pdf', bbox_inches='tight')