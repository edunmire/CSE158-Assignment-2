from datetime import datetime, timezone
import json

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from rate_prediction_latent_torch import *

def recommend_random_sampled_users(recommenders, num_samples, subset):
    if subset:
        users = pd.read_csv("./datasets/subset/users.csv")
        reviews = pd.read_csv("./datasets/subset/reviews.csv")
        cafes = pd.read_csv("./datasets/subset/cafes.csv")
    else:
        users = pd.read_csv("./datasets/processed/users.csv")
        reviews = pd.read_csv("./datasets/processed/reviews.csv")
        cafes = pd.read_csv("./datasets/processed/cafes.csv")

    reviews = reviews.sort_values(by=["user_id", "time"])

    sampled_users = users.sample(num_samples, random_state=42)

    for (user_id, num_reviews) in sampled_users.values:
        user_reviews = reviews[reviews["user_id"] == user_id]

        for i, user_review in enumerate(user_reviews.values):
            if i == 0:
                user_name = user_review[2]
                print(f"{user_name:20s}: ({num_reviews} reviews)")
            gmap_id = user_review[0]
            review_time = user_review[3]
            rating = user_review[4]


            cafe = cafes[cafes["gmap_id"] == gmap_id].values[0]
            cafe_name = cafe[1]

            date = datetime.fromtimestamp(review_time / 1000, tz=timezone.utc)
            date_str = date.strftime("%Y/%m/%d")
            print(f"    {cafe_name:50s}: {rating} rated at {date_str}")

        print()

        for recommender_name, recommender in recommenders.items():
            recommended_cafes, pred_ratings = recommender.recommend(user_id)
            print(f"{recommender_name} based recommendation for {user_name}")
            for recommended_cafe, pred_rating in zip(recommended_cafes, pred_ratings):
                cafe = cafes[cafes["gmap_id"] == recommended_cafe].values[0]
                cafe_name = cafe[1]
                print(f"    {cafe_name:50s}: {pred_rating}")

            print()

        print("=" * 80)
        print()

def analyze_consumption_vs_recommendation(recommender, num_samples, subset):
    if subset:
        users = pd.read_csv("./datasets/subset/users.csv")
        reviews = pd.read_csv("./datasets/subset/reviews.csv")
        cafes = pd.read_csv("./datasets/subset/cafes.csv")
    else:
        users = pd.read_csv("./datasets/processed/users.csv")
        reviews = pd.read_csv("./datasets/processed/reviews.csv")
        cafes = pd.read_csv("./datasets/processed/cafes.csv")

    reviews = reviews.sort_values(by=["user_id", "time"])

    sampled_users = users.sample(num_samples, random_state=42)

    all_cafes = []
    for (user_id, num_reviews) in sampled_users.values:
        recommended_cafes, pred_ratings = recommender.recommend(user_id)
        all_cafes += recommended_cafes

class CafeDatasetRecommender(CafeDatasetLatent):
    def __init__(self, user_id, feat_names, feat_dicts, subset):
        if subset:
            reviews = pd.read_csv(f"./datasets/subset/reviews.csv")
            cafes = pd.read_csv(f"./datasets/subset/cafes.csv")
        else:
            reviews = pd.read_csv(f"./datasets/processed/reviews.csv")
            cafes = pd.read_csv(f"./datasets/processed/cafes.csv")

        user_reviews = reviews[reviews["user_id"] == user_id]
        user_reviews = user_reviews.sort_values(by=["time"])
        user_gmap_ids = user_reviews["gmap_id"].values
        self.gmap_ids = cafes[np.logical_not(cafes["gmap_id"].isin(user_gmap_ids).values)]["gmap_id"].values

        self.feat_names = feat_names
        self.feat_dicts = feat_dicts

        self.user_id = user_id

    def __len__(self):
        return self.gmap_ids.shape[0]

    def __getitem__(self, index):
        gmap_id = self.gmap_ids[index]
        feats = []
        for name in self.feat_names:
            if name == "alpha":
                feat = torch.ones(1)
                feats.append(feat)

            elif name == "user":
                feat_dict = self.feat_dicts[name]
                feat = torch.zeros(len(feat_dict.keys()))
                feat[feat_dict[self.user_id]] = 1.
                feats.append(feat)

            elif name == "cafe":
                feat_dict = self.feat_dicts[name]
                feat = torch.zeros(len(feat_dict.keys()))
                feat[feat_dict[gmap_id]] = 1.
                feats.append(feat)

            elif name == "price":
                feat_dict = self.feat_dicts[name]
                feat = torch.tensor(price_to_onehot(gmap_id))
                feats.append(feat)

            elif name == "open_hours":
                feat_dict = self.feat_dicts[name]
                feat = torch.tensor(hours_to_onehot(gmap_id))
                feats.append(feat)

            elif name == "prev":
                cafe_feat_dict = self.feat_dicts['cafe']    # All cafes
                feat_dict = self.feat_dicts[name]           # List of user -> list of all cafes user rated

                feat = torch.zeros(len(cafe_feat_dict.keys()))
                feat[cafe_feat_dict[feat_dict[self.user_id][-1]]] = 1.
                feats.append(feat)

            else:
                raise NotImplementedError

        return feats

class LatentBasedCafeRecommender():
    def __init__(self, name, num_recommends, device, subset):
        self.name = name
        model = torch.load(f"./models/{name}.pt", weights_only=False)

        self.user_latents = model.latents[0]
        self.cafe_latents = model.latents[1]

        self.subset = subset

        self.feat_names = self.get_feat_names(name)
        self.feat_dicts, _ = preprocess_data_latent(self.feat_names, subset=subset)

        self.num_recommends = num_recommends

        self.index2cafe = {index: cafe for (cafe, index) in self.feat_dicts["cafe"].items()}

    def get_user_gmap_ids(self, user_id):
        if self.subset:
            reviews = pd.read_csv(f"./datasets/subset/reviews.csv")
            cafes = pd.read_csv(f"./datasets/subset/cafes.csv")
        else:
            reviews = pd.read_csv(f"./datasets/processed/reviews.csv")
            cafes = pd.read_csv(f"./datasets/processed/cafes.csv")

        user_reviews = reviews[reviews["user_id"] == user_id]
        user_gmap_ids = user_reviews["gmap_id"].values

        return user_gmap_ids

    def get_feat_names(self, name):
        with open("params.json", "r") as f:
            params = json.load(f)

        for param in params:
            feat = param["feat"]
            feat_names = param["feat_names"]
            latent_names = param["latent_names"]
            lamb_dict = param["lamb_dict"]

            lambs = [lamb_dict[feat] for feat in feat_names + latent_names]
            lamb_str = "-".join([str(l) for l in lambs])

            model_name = f"{feat}_{lamb_str}"
            if self.subset:
                model_name += "_subset"

            if model_name == name:
                return feat_names

        return None

class CosineBasedCafeRecommender(LatentBasedCafeRecommender):
    def recommend(self, user_id):
        user_index = self.feat_dicts["user"][user_id]
        user_latent = self.user_latents[user_index]

        dot = torch.einsum("bd,d->b", self.cafe_latents, user_latent)
        user_norm = torch.norm(user_latent, p=2, dim=0)
        cafe_norm = torch.norm(self.cafe_latents, p=2, dim=1)

        cosines = dot / (user_norm * cafe_norm + 1e-5)
        indices = torch.flip(torch.argsort(cosines), dims=[0])
        user_gmap_ids = self.get_user_gmap_ids(user_id)

        i = 0
        recommendeds = []
        metrics = []
        while True:
            index = indices[i].item()
            recommended_cafe = self.index2cafe[index]
            if recommended_cafe not in user_gmap_ids:
                recommendeds.append(recommended_cafe)
                metrics.append(cosines[index].item())

            if len(recommendeds) == self.num_recommends:
                break

            i += 1

        return recommendeds, metrics

class NormBasedCafeRecommender(LatentBasedCafeRecommender):
    def recommend(self, user_id):
        user_index = self.feat_dicts["user"][user_id]
        user_latent = self.user_latents[user_index]

        norms = torch.norm(self.cafe_latents - user_latent, p=2, dim=1)
        indices = torch.argsort(norms)
        user_gmap_ids = self.get_user_gmap_ids(user_id)

        i = 0
        recommendeds = []
        metrics = []
        while True:
            index = indices[i].item()
            recommended_cafe = self.index2cafe[index]
            if recommended_cafe not in user_gmap_ids:
                recommendeds.append(recommended_cafe)
                metrics.append(norms[index].item())

            if len(recommendeds) == self.num_recommends:
                break

            i += 1

        return recommendeds, metrics

class RankBasedCafeRecommender():
    def __init__(self, name, num_recommends, device, subset):
        self.name = name
        self.model = torch.load(f"./models/{name}.pt", weights_only=False)
        self.device = device
        self.subset = subset

        self.feat_names = self.get_feat_names(name)
        self.feat_dicts, _ = preprocess_data_latent(self.feat_names, subset=subset)

        self.num_recommends = num_recommends

    def get_feat_names(self, name):
        with open("params.json", "r") as f:
            params = json.load(f)

        for param in params:
            feat = param["feat"]
            feat_names = param["feat_names"]
            latent_names = param["latent_names"]
            lamb_dict = param["lamb_dict"]

            lambs = [lamb_dict[feat] for feat in feat_names + latent_names]
            lamb_str = "-".join([str(l) for l in lambs])

            model_name = f"{feat}_{lamb_str}"
            if self.subset:
                model_name += "_subset"

            if model_name == name:
                return feat_names

        return None

    def recommend(self, user_id):
        dataset = CafeDatasetRecommender(user_id, self.feat_names, self.feat_dicts, subset=subset)
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)

        predictions = np.array(self.run_predictions(dataloader))
        gmap_ids = np.array(dataset.gmap_ids)

        indices = np.argsort(predictions)[::-1][:num_recommends]

        return gmap_ids[indices], predictions[indices]

    def run_predictions(self, dataloader):
        predictions = []
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()

            for feats in dataloader:
                feats = [f.to(self.device) for f in feats]
                pred_ratings = self.model(feats).tolist()
                predictions += pred_ratings

        return predictions

if __name__ == "__main__":
    subset = True
    device = torch.device("cpu")

    name = "latent_0-0.1-1-0.1-1_subset"
    num_recommends = 10

    rank_recommender = RankBasedCafeRecommender(name, num_recommends, device, subset=subset)
    cosine_recommender = CosineBasedCafeRecommender(name, num_recommends, device, subset=subset)
    norm_recommender = NormBasedCafeRecommender(name, num_recommends, device, subset=subset)
    recommenders = {"Rank": rank_recommender, "Cosine": cosine_recommender, "Norm": norm_recommender}
    recommend_random_sampled_users(recommenders, 10, subset=subset)
