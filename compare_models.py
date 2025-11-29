import json
import os

import numpy as np

def compare_model_performances():
    if not os.path.exists("./metrics.json"):
        print("No model performance is available.")
        return

    with open("./metrics.json", "r") as f:
        metrics = json.load(f)

    names = []
    valid_mses = []

    for name, values in metrics.items():
        names.append(name)
        valid_mses.append(values["metrics"]["valid"][-1])

    indices = np.argsort(valid_mses)
    sorted_names = np.array(names)[indices]
    sorted_mses = np.array(valid_mses)[indices]

    print("\n".join([f"{name:40s}: {mse:.6f}" for name, mse in zip(sorted_names, sorted_mses)]))

if __name__ == "__main__":
    compare_model_performances()