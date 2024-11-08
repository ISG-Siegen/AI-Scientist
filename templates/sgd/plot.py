from glob import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt


results_dir = Path(__file__).parent
run_dirs = glob("run_*", root_dir=results_dir)

plot_data = {}

for run_dir in run_dirs:
    run_dir_path = results_dir / run_dir

    with open(run_dir_path / "all_results.json", "r") as f:
        final_results = json.load(f)

        plot_data[run_dir] = {
            "metrics": final_results["metrics"],
            "test_rmse": final_results["test_rmse"],
        }


# ADD RUNS HERE THAT WILL BE PLOTTED
runs = [
    "run_0",
]

for run in runs:
    plt.figure(figsize=(10, 6))
    plt.plot(
        plot_data[run]["metrics"]["Loss"].values(), label="Validation Loss", color="r"
    )
    plt.plot(
        plot_data[run]["metrics"]["RMSE"].values(), label="Validation RMSE", color="b"
    )
    plt.plot(
        plot_data[run]["metrics"]["MAE"].values(), label="Validation MAE", color="g"
    )

    plt.axhline(plot_data[run]["test_rmse"], label="Test RMSE", color="y")

    plt.title(f"Validation Loss, RMSE and MAE Across Epochs for {run}")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(results_dir / f"plot_{run}.png")
    plt.close()
