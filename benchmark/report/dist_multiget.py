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
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

# Batch sizes for x-axis
batch_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]

# Core == 2: Raw data from two ranks
core2_rank1_tp = [285967.97374001687, 498461.6388777506, 509615.70838606026, 517931.7658592939, 547077.5846308878, 536161.3359819114, 555888.3744347643, 594276.0775508514, 643249.7458245547, 720863.4139730241, 884329.7269598308, 1188589.878125369]
core2_rank2_tp = [323237.524738066, 413370.00317308406, 458519.70893118053, 493523.5793314313, 533830.9532489632, 539272.274586475, 555584.8321598751, 592472.8708900223, 641435.8259757912, 721445.462853517, 883018.5733551871, 1187471.8474149264]

# Calculate average throughput across ranks for core2
core2_avg_tp = [(r1 + r2)/2 for r1, r2 in zip(core2_rank1_tp, core2_rank2_tp)]

# Core == 3
core3_rank0_tp = [750810.7631013761, 1681539.0000606845, 2119512.267973531, 2767285.196951775, 3211916.434820393, 3693809.430484104, 3797862.6765961563, 4055338.605116953, 4404447.641676569, 5018578.026063284, 6176178.9801752465, 8542781.10989249]
core3_rank1_tp = [436678.11961614736, 1667129.7365128936, 2354062.972897239, 2916840.41365798, 3422269.643202931, 3750543.482885332, 4065896.3876436516, 4068518.232995291, 4425771.431681195, 4935846.914030711, 6300432.790035546, 8832535.829506254]
core3_rank2_tp = [638713.7138789867, 1012962.5912857432, 1591273.6639347381, 2091800.570500943, 2943909.4333695164, 3364785.9026540243, 3782683.5810080906, 3968338.7105284105, 4395777.11790393, 4984771.684801463, 6167682.149777898, 8280488.426738397]

core3_avg_tp = [(r0 + r1 + r2)/3 for r0, r1, r2 in zip(core3_rank0_tp, core3_rank1_tp, core3_rank2_tp)]

# Core == 4
core4_rank0_tp = [1117003.27351745, 2053345.918917766, 4344001.227986182, 8727885.643252261, 9943246.122862315, 19492612.87913573, 26128420.428953074, 41262600.75575998, 58260536.17809891, 47007222.56630642, 59251012.22580109, 59836321.380013965]
core4_rank1_tp = [868781.5320317905, 3541968.574634142, 5766846.212955551, 11127106.009241624, 14622422.363518354, 26462029.802163824, 35956166.896525905, 49962193.75451919, 63824723.76622258, 70999882.00880726, 69453546.46151504, 69883956.37691906]
core4_rank2_tp = [1134539.3034681308, 3495342.446716381, 5664378.620764408, 9019896.868073897, 12623078.046489403, 26771186.273039836, 28847344.400486633, 44235019.956883885, 48819606.03842633, 52457443.76558879, 56995095.74056231, 73192596.52739917]
core4_rank3_tp = [1066496.0663889544, 3949291.091879977, 3915963.5201172973, 10126999.499651384, 13212931.228363603, 26184007.153218877, 26548251.396026064, 37045961.279916845, 50650217.433210485, 58627917.4326096, 61294965.95822579, 68892040.53315422]


core4_avg_tp = [(r0 + r1 + r2 + r3)/4 for r0, r1, r2, r3 in zip(core4_rank0_tp, core4_rank1_tp, core4_rank2_tp, core4_rank3_tp)]

# Create academic-style plot
fig, ax = plt.subplots()

# Plot the data
ax.plot(batch_sizes, core2_avg_tp, marker='o', linestyle='-', 
        label='Core=2', color='#404040', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#404040')


ax.plot(batch_sizes, core3_avg_tp, marker='s', linestyle='-', 
        label='Core=3', color='#8B0000', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#8B0000')

ax.plot(batch_sizes, core4_avg_tp, marker='^', linestyle='-', 
        label='Core=4', color='#00008B', markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#00008B')

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


plt.savefig('dist_multiget_performance.pdf', bbox_inches='tight')

print("Academic plot generated successfully!")
print(f"Core=2 Average Throughput Range: {min(core2_avg_tp):.2f} - {max(core2_avg_tp):.2f} ops/s")