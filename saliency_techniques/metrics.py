import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def compute_auc(x, y):
    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    return np.trapz(y_sorted, x_sorted)


def plot_auc_from_json(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    for mode in ["positive", "negative"]:
        for metric in ["nds", "iou"]:
            percents = list(map(float, data[mode][metric].keys()))
            values = list(data[mode][metric].values())

            auc = compute_auc(percents, values)

            plt.figure()
            plt.plot(percents, values, marker='o', label=f"AUC = {auc:.4f}")
            plt.title(f"{mode.capitalize()} - {metric.upper()}")
            plt.xlabel("Percentage of Perturbed Pixels")
            plt.ylabel(metric.upper())
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            filename = f"{mode}_{metric}_auc.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

            print(f"Guardado: {filename} | AUC: {auc:.4f}")

def plot_combined_aucs(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_files = glob(os.path.join(json_dir, "*.json"))

    modes = ["positive", "negative"]
    metrics = ["nds", "iou"]

    for mode in modes:
        for metric in metrics:
            plt.figure()
            plt.title(f"{mode.capitalize()} - {metric.upper()}")
            plt.xlabel("Percentage")
            plt.ylabel(metric.upper())
            plt.grid(True)

            legend_entries = []
            handles = []

            for json_path in json_files:
                with open(json_path, "r") as f:
                    data = json.load(f)

                percents = list(map(float, data[mode][metric].keys()))
                values = list(data[mode][metric].values())
                auc = compute_auc(percents, values)

                label = os.path.splitext(os.path.basename(json_path))[0]

                if label.lower() == "random":
                    line, = plt.plot(
                        percents,
                        values,
                        color='gray',
                        linestyle='-',
                        marker='',
                        label=f"{label} (AUC={auc:.4f})"
                    )
                    handles_random = line
                    legend_random = f"{label} (AUC={auc:.4f})"
                else:
                    line, = plt.plot(
                        percents,
                        values,
                        marker='o',
                        label=f"{label} (AUC={auc:.4f})"
                    )
                    handles.append(line)
                    legend_entries.append(f"{label} (AUC={auc:.4f})")

            if 'handles_random' in locals():
                handles.append(handles_random)
                legend_entries.append(legend_random)

            plt.legend(handles, legend_entries, fontsize='small')
            plt.tight_layout()

            filename = f"{mode}_{metric}_combined.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

            print(f"Guardado: {filename}")


            
def main_one_plot():
    json_path = "./saliency_techniques/perturbation_tests/perturbation_results_bbox_base.json"
    output_dir = "./saliency_techniques/plots/bbox_base"
    plot_auc_from_json(json_path, output_dir)


if __name__ == "__main__":
    json_dir = "./saliency_techniques/perturbation_tests" 
    output_dir = "./saliency_techniques/plots"
    plot_combined_aucs(json_dir, output_dir)