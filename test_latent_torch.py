import os

from utils import *
from rate_prediction_latent_torch import *
from test_torch import update_test_results
import load_datasets

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def calculate_mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def discrete_rating(y_pred):
    y_pred = torch.clamp(y_pred, min=0, max=5)
    y_pred = torch.round(y_pred)

    return y_pred


def test_model(name, test_dataloader, model, device):
    with torch.no_grad():
        total = 0
        mse, rmse = 0, 0
        n_corrects = 0

        model.to(device)
        model.eval()

        feat_names = model.feat_names

        for feats in test_dataloader:
            ratings = feats[-1].to(device)
            feats = {name: f.to(device) for name, f in zip(feat_names, feats[:-1])}

            pred_ratings = model(feats)

            batch_size = feats["alpha"].size(0)
            mse += calculate_mse(ratings, pred_ratings).item() * batch_size
            rmse += calculate_rmse(ratings, pred_ratings).item() * batch_size

            pred_discrete = discrete_rating(pred_ratings)

            n_corrects += torch.sum(pred_discrete == ratings).item()
            total += batch_size

        test_mse = mse / total
        test_rmse = rmse / total
        test_accuracy = n_corrects / total

        return {"name": name, "mse": test_mse, "rmse": test_rmse, "accuracy": test_accuracy}

if __name__ == "__main__":
    subset = True

    with open("./params.json", "r") as f:
        params = json.load(f)

    for param_dict in params:
        feat = param_dict["feat"]
        feat_names = param_dict["feat_names"]
        latent_names = param_dict["latent_names"]
        latent_pairs = param_dict["latent_pairs"]
        lamb_dict = param_dict["lamb_dict"]
        share_latents = param_dict.get("share_latents", 0)

        test = param_dict["test"]

        lamb_str = "_".join([f"{name}-{value}" for name, value in lamb_dict.items()])
        name = f"{feat}_{lamb_str}"
        if subset:
            name += "_subset"

        if not test:
            continue

        batch_size = 2048
        feat_dicts, avg_rating = preprocess_data_latent(feat_names, subset=subset)

        test_dataset = CafeDatasetLatent("test", feat_names, feat_dicts, subset=subset)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        model_path = f"./models/{name}.pt"

        if not os.path.exists(model_path):
            continue

        print(f"Loading model from {model_path}")
        model = torch.load(model_path, weights_only=False)

        device = torch.device("cpu")
        result = test_model(name, test_dataloader, model, device)

        update_test_results(result)
