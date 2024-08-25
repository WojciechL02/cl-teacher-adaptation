from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from constants import (
    COLOR_PALETTE,
    HUE_ORDER,
    LEGEND_FONTSIZE,
    NAME_FT,
    NAME_NMC_EX,
    NAME_NMC_FULL,
    PLOT_LINEWIDTH,
    TEXT_FONTSIZE,
    TICK_FONTSIZE,
)

root = Path(__file__).parent
output_dir = root / "plots"
output_dir.mkdir(exist_ok=True, parents=True)
output_path_png = output_dir / "fig3d.png"
output_path_pdf = output_dir / "fig3d.pdf"

plt.figure()
plt.clf()
plt.cla()

# FT
# text = "0.12 0.022 0.021 0.035 0.8 0.031 0.17 0.035 0.025 0.74 0.019 0.021 0.19 0.037 0.73 0.015 0.021 0.033 0.24 0.69 0.008 0.0085 0.0055 0.0085 0.97"
# NMC
# text = "0.33 0.1 0.097 0.14 0.33 0.1 0.39 0.1 0.12 0.28 0.077 0.09 0.41 0.13 0.29 0.068 0.061 0.11 0.5 0.26 0.06 0.056 0.07 0.089 0.72"
# JOINT
# text = "0.53 0.1 0.087 0.11 0.17 0.09 0.6 0.079 0.081 0.15 0.074 0.08 0.56 0.13 0.16 0.068 0.069 0.099 0.63 0.14 0.085 0.064 0.079 0.1 0.67"
# JOINT NMC
text = "0.56 0.11 0.096 0.11 0.12 0.1 0.61 0.091 0.097 0.096 0.091 0.089 0.59 0.13 0.1 0.084 0.086 0.12 0.61 0.098 0.11 0.096 0.12 0.13 0.54"
values = [round(100 * float(t), 1) for t in text.split()]
matrix = np.array(values).reshape((5, 5))
plot = sns.heatmap(matrix, annot=True, cmap='Blues', cbar=True, vmin=0, vmax=100)

title = "Joint NMC"
xlabel="Predicted task"
ylabel="True task"

plot.set_title(title)
plot.set_xlabel(xlabel)

# Set sizes for text and ticks
plot.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE)
plot.set_ylabel(ylabel, fontsize=TEXT_FONTSIZE)
plot.set_title(title, fontsize=TEXT_FONTSIZE)
# plot.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(str(output_path_png))
plt.savefig(str(output_path_pdf))
