"""
============================================================
Filename: plot.py
Author: Moritz Baumgart
Affiliation: University of Siegen, Intelligent Systems Group (ISG)
Date: December, 2024
============================================================

Description:
This file is part of the `knn` template for use with the AI scientist ('AI-S'), https://github.com/SakanaAI/AI-Scientist.
It uses the final_info.json dropped by experiment.py and matplotlib to generate plots for that AI-S can then embed in its final writeup/paper.
============================================================
"""

from glob import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt


results_dir = Path(__file__).parent
run_dirs = glob("run_*", root_dir=results_dir)

plot_data = {}

for run_dir in run_dirs:
    run_dir_path = results_dir / run_dir

    with open(run_dir_path / "final_info.json", "r") as f:
        final_results: dict = json.load(f)
        plot_data[run_dir] = {
            ds: final_results[ds]["means"] for ds in final_results.keys()
        }

# ADD RUNS HERE THAT WILL BE PLOTTED
runs = [
    "run_0",
]

for run in runs:
    plt.figure(figsize=(10, 6))

    for dataset, scores in plot_data[run].items():
        plt.bar(scores.keys(), scores.values(), label=dataset)

    plt.title(f"Scores for {run}")
    plt.xlabel("Metric")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(results_dir / f"plot_{run}.png")
    plt.close()
