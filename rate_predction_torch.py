import json
from collections import defaultdict
import pickle
import os

import numpy as np
import pandas as pd
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataset import CafeDataset


class RatePredictor(nn.Module):
    def __init__(self, name, feat_size, avg_rating):
        super().__init__()

        self.name = name

        weights = torch.zeros(feat_size)
        weights[0] = avg_rating

        self.weights = nn.Parameter(weights, requires_grad=True)

    def forward(self, feats):
        return torch.einsum("bd,d->b", feats, self.weights)

def preprocess_data():
    reviews = pd.read_csv("./datasets/processed/reviews.csv")
    cafes = pd.read_csv("./datasets/processed/cafes.csv")

    unique_user_ids = np.sort(np.unique(reviews["user_id"].values))
    user2index = {user_id: index for index, user_id in enumerate(unique_user_ids)}

    unique_gmap_ids = np.sort(np.unique(cafes["gmap_id"]))
    cafe2index = {gmap_id: index for index, gmap_id in enumerate(unique_gmap_ids)}

    avg_rating = reviews["rating"].mean()

    return user2index, cafe2index, avg_rating

class CafeDataset(Dataset):
    def __init__(self, mode, user2index, cafe2index):
        self.user2index = user2index
        self.cafe2index = cafe2index
        self.reviews = pd.read_csv(f"./datasets/splits/{mode}.csv").values

        self.user_size = len(self.user2index.keys())
        self.cafe_size = len(self.cafe2index.keys())
        self.feat_size = 1 + self.user_size + self.cafe_size

    def __len__(self):
        return self.reviews.shape[0]

    def __getitem__(self, index):
        review = self.reviews[index]
        feat = torch.zeros(self.feat_size)
        feat[0] = 1.

        user_index = self.user2index[review[1]]
        feat[1 + user_index] = 1

        cafe_index = self.cafe2index[review[0]]
        feat[1 + self.user_size + cafe_index] = 1

        rating = torch.tensor(review[4])

        return feat, rating

class RateTrainer():
    def __init__(self, model, lamb, train_dataloader, valid_dataloader, device):
        self.model = model
        self.lamb = lamb
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device

        self.optim =  torch.optim.Adam(model.parameters(), lr=0.1)

    def train(self, n_epochs):
        train_mses, valid_mses = [], []
        for i in range(n_epochs):
            train_mse = 0
            total = 0
            for (feats, ratings) in tqdm.tqdm(self.train_dataloader):
                feats = feats.to(self.device)
                ratings = ratings.to(self.device)

                self.optim.zero_grad()

                pred_ratings = self.model(feats)
                mse = self.mse(ratings, pred_ratings)
                mse_reg = mse + self.regularizer()
                mse_reg.backward()

                self.optim.step()

                batch_size = feats.size(0)
                train_mse += mse.item() * batch_size
                total += batch_size

            train_mse /= total
            valid_mse = self.validate()
            print(f"Step[{i:2d}]: train {train_mse:2.6f} / valid {valid_mse:2.6f}")

            train_mses.append(train_mse)
            valid_mses.append(valid_mse)

        return train_mses, valid_mses

    def validate(self):
        with torch.no_grad():
            total = 0
            mse = 0

            for (feats, ratings) in self.valid_dataloader:
                feats = feats.to(self.device)
                ratings = ratings.to(self.device)

                pred_ratings = self.model(feats)

                batch_size = feats.size(0)
                mse += self.mse(ratings, pred_ratings).item() * batch_size
                total += batch_size

            return mse / total

    def mse(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def regularizer(self):
        return self.lamb * torch.mean(self.model.weights[1:] ** 2)
