from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Converts time text to values
def parse_time(t):
    t = t.strip().upper()

    # Match hh or hh:mm formats
    m = re.match(r"(\d{1,2})(?::(\d{2}))?(AM|PM)", t)
    if not m:
        raise ValueError(f"Invalid time format: {t}")

    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    period = m.group(3)

    # Convert to 24-hour
    if period == "AM":
        if hour == 12:
            hour = 0
    else:  # PM
        if hour != 12:
            hour += 12

    return hour + minute / 60.0

# One Hot Encoding for Unix Time Weekday
def unix_weekday_to_onehot(time):
    feature_weekday = [0]*7

    day = datetime.fromtimestamp(time / 1000, tz=timezone.utc).weekday()
    feature_weekday[day] = 1.

    return feature_weekday

# One Hot Encoding for Unix Time Hour
def unix_hour_to_onehot(time):
    feature_dayhour = [0]*24

    hr = datetime.fromtimestamp(time / 1000, tz=timezone.utc).hour
    feature_dayhour[hr] = 1.

    return feature_dayhour

# One Hot Encoding for Price
def price_to_onehot(price):
    feature_price = [0]*3
    if price is not np.nan:
        feature_price[len(price)-1] += 1
    return feature_price

# One Hot Encoding for Open Hours
def hours_to_onehot(hours):
    before_noon = 0
    after_noon = 0

    for entry in hours:
        if entry[1] == "Open 24 hours":
            return [1,0,0]
        
        open_str, close_str = entry[1].split("–")
        start_hr = int(np.floor(parse_time(open_str)))
        if start_hr < 13:
            before_noon += 1
        else: after_noon += 1
    
    if before_noon > after_noon:
        return [0,1,0]
    return [0,0,1]

class RatePredictorLatent(nn.Module):
    def __init__(self, name, dim, feat_sizes, latent_names, latent_pairs, avg_rating):
        super().__init__()

        self.name = name
        self.num_feats = len(feat_sizes)
        self.num_latents = len(latent_names)

        self.latent_names = latent_names
        self.latent_pairs = latent_pairs

        self.latent_indices = []
        weights = []
        for i, (name, feat_size) in enumerate(feat_sizes.items()):
            if name == "alpha":
                weight = torch.tensor(avg_rating).unsqueeze(0)
            else:
                weight = torch.zeros(feat_size)
            weights.append(nn.Parameter(weight, requires_grad=True))

            if name in latent_names:
                self.latent_indices.append(i)

        self.weights =  nn.ParameterList(weights)

        latents = []
        for name in latent_names:
            feat_size = feat_sizes[name]
            latent = torch.randn(feat_size, dim) / dim
            latents.append(nn.Parameter(latent, requires_grad=True))

        self.latents = nn.ParameterList(latents)

    def forward(self, feats):
        assert len(feats) == self.num_feats

        out = torch.zeros(feats[0].size(0))
        for i in range(self.num_feats):
            out += torch.einsum("bd,d->b", feats[i], self.weights[i])

        gammas = {}
        for i in range(self.num_latents):
            index = self.latent_indices[i]
            gammas[self.latent_names[i]] = torch.einsum("bd,di->bi", feats[index], self.latents[i])

        for (latent_i, latent_j) in self.latent_pairs:
            out += torch.einsum("bi,bi->b", gammas[latent_i], gammas[latent_j])

        return out

def preprocess_data_latent(feat_names):
    reviews = pd.read_csv("./datasets/processed/reviews.csv")
    cafes = pd.read_csv("./datasets/processed/cafes.csv")

    feat_dicts = {}
    for name in feat_names:
        if name == "user":
            unique_user_ids = np.sort(np.unique(reviews["user_id"].values))
            user2index = {user_id: index for index, user_id in enumerate(unique_user_ids)}
            feat_dicts[name] = user2index

        elif name == "cafe":
            unique_gmap_ids = np.sort(np.unique(cafes["gmap_id"]))
            cafe2index = {gmap_id: index for index, gmap_id in enumerate(unique_gmap_ids)}
            feat_dicts[name] = cafe2index

        elif name == "price":
            unique_gmap_ids, indices = np.unique(cafes["gmap_id"], return_index=True)
            order = np.argsort(unique_gmap_ids)
            unique_gmap_ids = unique_gmap_ids[order]
            indices = indices[order]
            cafe2price = {gmap_id: cafes["prices"][index] for gmap_id, index in zip(unique_gmap_ids, indices)}
            feat_dicts[name] = cafe2price

        elif name == "open_hours":
            unique_gmap_ids, indices = np.unique(cafes["gmap_id"], return_index=True)
            order = np.argsort(unique_gmap_ids)
            unique_gmap_ids = unique_gmap_ids[order]
            indices = indices[order]
            cafe2hours = {gmap_id: cafes["hours"][index] for gmap_id, index in zip(unique_gmap_ids, indices)}
            feat_dicts[name] = cafe2hours

    avg_rating = reviews["rating"].mean()

    return feat_dicts, avg_rating

class CafeDatasetLatent(Dataset):
    def __init__(self, mode, feat_names, feat_dicts, subset):
        if subset:
            self.reviews = pd.read_csv(f"./datasets/splits/{mode}_subset.csv").values
        else:
            self.reviews = pd.read_csv(f"./datasets/splits/{mode}.csv").values

        self.feat_names = feat_names
        self.feat_dicts = feat_dicts

    def get_feat_sizes(self):
        feat_sizes = {}

        for name in self.feat_names:
            if name == "alpha":
                feat_sizes[name] = 1

            elif name == "user":
                feat_sizes[name] = len(self.feat_dicts[name].keys())

            elif name == "cafe":
                feat_sizes[name] = len(self.feat_dicts[name].keys())

            elif name == "weekday":
                feat_sizes[name] = 7

            elif name == "hour":
                feat_sizes[name] = 24

            elif name == "price":
                feat_sizes[name] = 3

            elif name == "open_hours":
                feat_sizes[name] = 3

            else:
                raise NotImplementedError

        return feat_sizes

    def __len__(self):
        return self.reviews.shape[0]

    def __getitem__(self, index):
        review = self.reviews[index]
        feats = []
        for name in self.feat_names:
            if name == "alpha":
                feat = torch.ones(1)
                feats.append(feat)

            elif name == "user":
                feat_dict = self.feat_dicts[name]
                feat = torch.zeros(len(feat_dict.keys()))
                feat[feat_dict[review[1]]] = 1.
                feats.append(feat)

            elif name == "cafe":
                feat_dict = self.feat_dicts[name]
                feat = torch.zeros(len(feat_dict.keys()))
                feat[feat_dict[review[0]]] = 1.
                feats.append(feat)

            elif name == "weekday":
                feat = torch.tensor(unix_weekday_to_onehot(int(review[3])))
                feats.append(feat)

            elif name == "hour":
                feat = torch.tensor(unix_hour_to_onehot(int(review[3])))
                feats.append(feat)

            elif name == "price":
                feat_dict = self.feat_dicts[name]
                feat = torch.tensor(price_to_onehot(feat_dict[review[0]]))
                feats.append(feat)

            elif name == "open_hours":
                feat_dict = self.feat_dicts[name]
                feat = torch.tensor(hours_to_onehot(feat_dict[review[0]]))
                feats.append(feat)

            else:
                raise NotImplementedError

        rating = torch.tensor(review[4])

        return *feats, rating

class RateTrainerLatent():
    def __init__(self, model, lambs, lr, train_dataloader, valid_dataloader, device):
        self.model = model
        self.lambs = lambs
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device

        self.optim =  torch.optim.Adam(model.parameters(), lr=lr)

    def train(self, n_epochs):
        train_mses, valid_mses = [], []
        for i in range(n_epochs):
            train_mse = 0
            total = 0

            bar = tqdm.tqdm(self.train_dataloader, desc="Training Model")
            for feats in bar:
                ratings = feats[-1].to(self.device)
                feats = [f.to(self.device) for f in feats[:-1]]

                self.optim.zero_grad()

                pred_ratings = self.model(feats)
                mse = self.mse(ratings, pred_ratings)
                mse_reg = mse + self.regularizer()

                mse_reg.backward()
                self.optim.step()

                batch_size = feats[0].size(0)
                train_mse += mse.item() * batch_size
                total += batch_size

                bar.set_description(f"Training Model ({mse.item():.6f})")

            train_mse /= total
            valid_mse = self.validate()
            print(f"Step[{i + 1:2d}]: train {train_mse:2.6f} / valid {valid_mse:2.6f}")

            train_mses.append(train_mse)
            valid_mses.append(valid_mse)

        return train_mses, valid_mses

    def validate(self):
        with torch.no_grad():
            total = 0
            mse = 0

            for feats in self.valid_dataloader:
                ratings = feats[-1].to(self.device)
                feats = [f.to(self.device) for f in feats[:-1]]

                pred_ratings = self.model(feats)

                batch_size = feats[0].size(0)
                mse += self.mse(ratings, pred_ratings).item() * batch_size
                total += batch_size

            return mse / total

    def mse(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def regularizer(self):
        weight_size = len(self.model.weights)
        latent_size = len(self.model.latents)
        dim = self.model.latents[0].size(1)
        assert len(self.lambs) == weight_size + latent_size
        reg = 0
        for i in range(len(self.lambs)):
            if i < len(self.model.weights):
                reg += self.lambs[i] * torch.mean(self.model.weights[i] ** 2)
            else:
                reg += self.lambs[i] * dim * torch.mean(self.model.latents[i - weight_size] ** 2)

        return reg
