import json
import os

import numpy as np

def compare_model_performances():
    if not os.path.exists("./predictions/metrics.json"):
        print("No model performance is available.")
        return

    with open("./predictions/metrics.json", "r") as f:
        metrics = json.load(f)

    names = []
    valid_mses = []

    for name, values in metrics.items():
        names.append(name)
        valid_mses.append(values["metrics"]["valid"])

    indices = np.argsort(valid_mses)
    sorted_names = np.array(names)[indices]
    sorted_mses = np.array(valid_mses)[indices]

    print("\n".join([f"{name:25s}: {mse:.6f}" for name, mse in zip(sorted_names, sorted_mses)]))
