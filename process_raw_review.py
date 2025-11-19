# I reused my code from COGS108 project to process dataset.

import os
import json
import gzip
from functools import partial

import requests
import numpy as np
import pandas as pd
import tqdm

review_path = "./datasets/raw/review-California.json.gz"
review_keys = ["gmap_id", "user_id", "name", "time", "rating"]

total_reviews = 70529977

def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)

def download_review_data():
    url = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/review-California.json.gz"
    res = requests.get(url, stream=True)

    with open(review_path, "wb") as f:
        f.write(res.content)

def filter_by_gmap_id(data, gmap_ids):
    gmap_id = data.get("gmap_id", None)
    if gmap_id is None:
        return False

    return gmap_id in gmap_ids

def filter_raw_review_data(filters):
    reviews = []

    for review in tqdm.tqdm(parse(review_path), total=total_reviews):
        if all([f(data=review) for f in filters]):
            review = {key: review.get(key, None) for key in review_keys}
            review["review_id"] = f"{review['user_id']}_{review['gmap_id']}"
            reviews.append(review)

    print(f"We obtained total of {len(reviews)} after filtering")
    df = pd.DataFrame(reviews)
    df.to_csv("./datasets/raw/cafe_reviews.csv", index=False)

def extract_user_ids(reviews, min_num_reviews):
    user_ids = reviews["user_id"].dropna().values
    unique, counts = np.unique(np.array(user_ids), return_counts=True)
    users = pd.DataFrame({"user_id": unique, "num_reviews": counts})

    users = users[users["num_reviews"] >= min_num_reviews].reset_index(drop=True)
    users.to_csv("./datasets/processed/users.csv", index=False)

def filter_by_user_ids(reviews, user_ids):
    reviews = reviews[reviews["user_id"].isin(user_ids)]
    reviews.to_csv("./datasets/processed/reviews.csv", index=False)

if __name__ == "__main__":
    gmap_ids = set(pd.read_csv("./datasets/processed/cafes.csv")["gmap_id"].values)

    if not os.path.exists("./datasets/raw/review-California.json.gz"):
        download_review_data()

    if not os.path.exists("./datasets/raw/cafe_reviews.csv"):
        gmap_id_filter = partial(filter_by_gmap_id, gmap_ids=gmap_ids)
        filter_raw_review_data([gmap_id_filter])

    if not os.path.exists("./datasets/processed/users.csv"):
        print("Start processing user data")
        reviews = pd.read_csv("./datasets/raw/cafe_reviews.csv")
        min_num_reviews = 5
        extract_user_ids(reviews, min_num_reviews)

    if not os.path.exists("./datasets/processed/reviews.csv"):
        print("Start filtering review data")
        reviews = pd.read_csv("./datasets/raw/cafe_reviews.csv")
        user_ids = pd.read_csv("./datasets/processed/users.csv")["user_id"].values
        filter_by_user_ids(reviews, user_ids)