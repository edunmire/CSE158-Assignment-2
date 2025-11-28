import pandas as pd
import numpy as np

class CafeDataset():
    def __init__(self):
        reviews = pd.read_csv("./datasets/processed/reviews.csv")
        cafes = pd.read_csv("./datasets/processed/cafes.csv")

        unique_user_ids = np.sort(np.unique(reviews["user_id"].values))
        self.user2index = {user_id: index for index, user_id in enumerate(unique_user_ids)}

        unique_gmap_ids = np.sort(np.unique(cafes["gmap_id"]))
        self.cafe2index = {gmap_id: index for index, gmap_id in enumerate(unique_gmap_ids)}

    def get_feat_size(self):
        user_size = len(self.user2index.keys())
        cafe_size = len(self.cafe2index.keys())

        return 1 + user_size + cafe_size, user_size, cafe_size

    def get_rating_average(self):
        return pd.read_csv(f"./datasets/splits/train.csv")["rating"].mean()

    def load(self, mode):
        reviews = pd.read_csv(f"./datasets/splits/{mode}.csv")
        return reviews

    def load_batch(self, reviews, batch_size, batch_index):
        if batch_size * (batch_index + 1) > reviews.shape[0]:
            return None, None

        reviews = reviews.iloc[batch_size * batch_index : batch_size * (batch_index + 1)]

        feat_size, user_size, _ = self.get_feat_size()
        feats = np.zeros([batch_size, feat_size]).astype(np.int8)
        feats[:, 0] = np.ones(reviews.shape[0])

        ratings = []

        for i, review in enumerate(reviews.values):
            user_index = self.user2index[review[1]]
            feats[i, 1 + user_index] = 1.

            cafe_index = self.cafe2index[review[0]]
            feats[i, 1 + user_size + cafe_index] = 1.

            ratings.append(review[4])

        ratings = np.array(ratings).astype(np.float16)

        return feats, ratings

if __name__ == "__main__":
    dataset = CafeDataset()

    valid_feats, valid_ratings = dataset.load("valid")
    print(valid_feats.shape, valid_ratings.shape)