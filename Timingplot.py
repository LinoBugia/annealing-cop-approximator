"""Step time of DA against SA, drawn from Aggregiert.csv.

Aggregiert.csv holds, per number of variables, the measured time for 300
iterations of each annealer on dense quadratic PBFs (n(n+1)/2 monomials).
Writes docs/img/step_time_da_sa.png.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "Aggregiert.csv"))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["vars"], df["time_DA"], "o-", color="red", label="DA")
ax.plot(df["vars"], df["time_SA"], "o-", color="blue", label="SA")
ax.set_xlabel("number of variables n")
ax.set_ylabel("time for 300 iterations")
ax.set_title("Step time on dense quadratic PBFs")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(HERE, "docs", "img", "step_time_da_sa.png"), dpi=300)
plt.show()
