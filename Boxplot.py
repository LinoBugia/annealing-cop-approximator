import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------------
# Read the CSV
# ------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "Code", "Data.csv"))

# Columns holding the values (0..20)
value_columns = df.columns[2:]

# All cooling constants (column T)
T_values = sorted(df['T'].unique())

# ------------------------------
# Prepare the data
# ------------------------------
data = []
positions = []
colors = []
width = 0.35  # spacing between the DA and SA boxes

for i, T in enumerate(T_values):
    # DA
    da_vals = df[(df['T'] == T) & (df['type'] == 'DA')][value_columns].values.flatten()
    # replace NaN and 0 by a small value so they show on the log scale
    da_vals = np.where(np.isnan(da_vals) | (da_vals <= 0), 0.1, da_vals)
    data.append(da_vals)
    positions.append(i - width/2)
    colors.append('red')

    # SA
    sa_vals = df[(df['T'] == T) & (df['type'] == 'SA')][value_columns].values.flatten()
    sa_vals = np.where(np.isnan(sa_vals) | (sa_vals <= 0), 0.1, sa_vals)
    data.append(sa_vals)
    positions.append(i + width/2)
    colors.append('blue')

# ------------------------------
# Draw the boxplot
# ------------------------------
fig, ax = plt.subplots(figsize=(10,6))
bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=True)

# Assign colours
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# ------------------------------
# Logarithmic scale
# ------------------------------
ax.set_yscale('log')
ax.set_ylabel('final energy (log scale)')

# ------------------------------
# Axes, labels, legend
# ------------------------------
ax.set_xticks(range(len(T_values)))
ax.set_xticklabels(T_values)
ax.set_xlabel('cooling constant c')
ax.set_title('Number partitioning, n = 1500: DA (red) vs SA (blue) at equal compute')
ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.5)

# Legend
legend_elements = [Patch(facecolor='red', label='DA', alpha=0.6),
                   Patch(facecolor='blue', label='SA', alpha=0.6)]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(os.path.join(HERE, "docs", "img", "grouped_boxplot_da_sa_log.png"), dpi=300)
plt.show()
