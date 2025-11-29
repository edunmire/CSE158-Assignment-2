import json
from collections import defaultdict
import pickle
import os

import numpy as np
import pandas as pd
import tqdm

from dataset import CafeDataset

class RatePredictor():
    def __init__(self, name):
        self.name = name
        self.dataset = CafeDataset()
        feat_size, self.user_size, self.cafe_size = self.dataset.get_feat_size()

        # Initialize weights
        self.weights = np.zeros(feat_size)

    def evaluate(self, feats, ratings):
        predictions = self.predict(feats)

        mse = np.mean((predictions - ratings) ** 2)
        return mse

    def validate(self):
        reviews = self.dataset.load("valid")
        feats, ratings = self.dataset.load_batch(reviews, reviews.shape[0], 0)
        mse = self.evaluate(feats, ratings)
        return mse

    def predict(self, feats):
        # Perform matrix multiplication (num data x feat size) x (feat size)
        return np.einsum("bd,d->b", feats, self.weights)

class BaseRatePredictor(RatePredictor):
    def fit(self, lambs, n_epochs, batch_size=20000):
        # Load dataset
        reviews = self.dataset.load("train")

        # Initialize bias term as global average
        self.weights[0] = self.dataset.get_rating_average()

        lamb_user, lamb_cafe = lambs

        train_mses, valid_mses = [], []
        for i in range(n_epochs):
            running_mses = []
            batch_index = 0
            while True:
                feats, ratings = self.dataset.load_batch(reviews, batch_size, batch_index)

                if feats is None:
                    break

                self.update_alpha(feats, ratings)
                self.update_beta_user(feats, ratings, lamb_user)
                self.update_beta_cafe(feats, ratings, lamb_cafe)

                running_mses.append(float(self.evaluate(feats, ratings)))

                batch_index += 1

            train_mse = float(np.mean(running_mses))
            train_mses.append(train_mse)

            valid_mse = float(self.validate())
            valid_mses.append(valid_mse)

            print(f"epoch[{i+1}] train: {train_mse:.5f} valid: {valid_mse:.5f}")

        return train_mses, valid_mses

    def update_alpha(self, feats, ratings):
        total = np.sum(ratings - np.einsum("bd,d->b", feats[:, 1:], self.weights[1:]))
        self.weights[0] = total / feats.shape[0]

    def update_beta_user(self, feats, ratings, lamb):
        total = 0

        for user_index in tqdm.tqdm(range(self.user_size)):
            feat_index = 1 + user_index
            indices = feats[:, feat_index] == 1
            size = np.sum(indices)

            user_feats = feats[indices]
            user_ratings = ratings[indices]

            # Avoid calculating beta u.
            user_feats[:, feat_index] = np.zeros(size)
            assert np.sum(user_feats) == 2 * size

            total = np.sum(user_ratings - np.einsum("bd,d->b", user_feats, self.weights))

            self.weights[feat_index] = total / (lamb + size)

    def update_beta_cafe(self, feats, ratings, lamb):
        total = 0

        for cafe_index in tqdm.tqdm(range(self.cafe_size)):
            feat_index = 1 + self.user_size + cafe_index
            indices = feats[:, feat_index] == 1
            size = np.sum(indices)

            cafe_feats = feats[indices]
            cafe_ratings = ratings[indices]

            # Avoid calculating beta u.
            cafe_feats[:, feat_index] = np.zeros(size)
            assert np.sum(cafe_feats) == 2 * size

            total = np.sum(cafe_ratings - np.einsum("bd,d->b", cafe_feats, self.weights))

            self.weights[feat_index] = total / (lamb + size)