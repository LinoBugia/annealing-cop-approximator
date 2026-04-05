import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------------
# CSV einlesen
# ------------------------------
df = pd.read_csv("/Users/lino/Documents/python/AnnealingCopApproximator/Code/Data.csv")

# Spalten mit den Werten (0..20)
value_columns = df.columns[2:]

# Alle T-Werte
T_values = sorted(df['T'].unique())

# ------------------------------
# Daten vorbereiten
# ------------------------------
data = []
positions = []
colors = []
width = 0.35  # Abstand zwischen DA/SA Boxplots

for i, T in enumerate(T_values):
    # DA
    da_vals = df[(df['T'] == T) & (df['type'] == 'DA')][value_columns].values.flatten()
    # NaN entfernen und 0 durch minimalen Wert ersetzen (log Skala)
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
# Boxplot zeichnen
# ------------------------------
fig, ax = plt.subplots(figsize=(10,6))
bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=True)

# Farben zuweisen
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# ------------------------------
# Logarithmische Skala
# ------------------------------
ax.set_yscale('log')
ax.set_ylabel('Werte (log scale)')

# ------------------------------
# Achsen, Labels, Legende
# ------------------------------
ax.set_xticks(range(len(T_values)))
ax.set_xticklabels(T_values)
ax.set_xlabel('T')
ax.set_title('Zahlenpartitionierung: DA (rot) vs SA (blau) mit gleichem Rechenaufwand')
ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.5)

# Legende
legend_elements = [Patch(facecolor='red', label='DA', alpha=0.6),
                   Patch(facecolor='blue', label='SA', alpha=0.6)]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig("grouped_boxplot_da_sa_log.png", dpi=300)
plt.show()
