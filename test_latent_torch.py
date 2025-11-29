import os

from utils import *
from rate_prediction_latent_torch import *
from test_torch import update_test_results

import torch.nn as nn
from torch.utils.data import DataLoader

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

        for feats in test_dataloader:
            ratings = feats[-1].to(device)
            feats = [f.to(device) for f in feats[:-1]]

            pred_ratings = model(feats)

            batch_size = feats[0].size(0)
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
    name = "latent_torch_0-2-2-2-2_10_0.01_16_subset"

    feat_names = ["alpha", "user", "cafe"]
    latent_names = ["user", "cafe"]

    batch_size = 2048
    feat_dicts, avg_rating = preprocess_data_latent(feat_names)

    test_dataset = CafeDatasetLatent("test", feat_names, feat_dicts, subset=subset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = torch.load(f"./models/{name}.pt", weights_only=False)

    device = torch.device("cpu")
    result = test_model(name, test_dataloader, model, device)

    update_test_results(result)