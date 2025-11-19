# I reused my code from COGS108 project to process dataset.

import os
import json
import gzip
from functools import partial

import requests
import numpy as np
import pandas as pd

meta_path = "./datasets/raw/meta-California.json.gz"
meta_keys = ["gmap_id", "name", "latitude", "longitude", "category", "avg_rating", "num_of_reviews", "price", "hours"]

def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield json.loads(l)

def download_meta_data():
    url = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/meta-California.json.gz"
    res = requests.get(url, stream=True)

    with open(meta_path, "wb") as f:
        f.write(res.content)

def get_cafe_categories():
    categories = []
    for business in parse(meta_path):
        if business["category"] is not None:
            categories += business["category"]

    categories = np.array(categories)
    unique = np.unique(categories)

    cafe_categories = [str(category) for category in unique if "cafe" in category.lower() or "coffee" in category.lower()]
    print(f"The number of categories containting 'cafe' substring is {len(cafe_categories)}")
    print(cafe_categories)

    with open(f"./datasets/processed/cafe_categories.txt", "w") as f:
        f.write("\n".join(cafe_categories))

def filter_by_category(data, categories):
    category = data.get("category", None)
    if category is None:
        return False

    return len(set(category) & categories) != 0

def filter_by_num_reviews(data, min_num_reviews):
    return data["num_of_reviews"] >= min_num_reviews

def filter_raw_business_data(filters):
    businesses = []
    for business in parse(meta_path):
        if all([f(data=business) for f in filters]):
            business = {key: business.get(key, None) for key in meta_keys}
            businesses.append(business)

    print(f"We obtained total of {len(businesses)} after filtering")

    df = pd.DataFrame(businesses)
    df.to_csv(f"./datasets/processed/cafes.csv", index=False)

if __name__ == "__main__":
    os.makedirs("./datasets/raw", exist_ok=True)
    os.makedirs("./datasets/processed", exist_ok=True)

    if not os.path.exists(meta_path):
        download_meta_data()

    if not os.path.exists("./datasets/processed/cafe_categories.txt"):
        get_cafe_categories()

    if not os.path.exists("./datasets/processed/cafes.csv"):
        min_num_reviews = 5

        with open("./datasets/processed/cafe_categories.txt", "r") as f:
            cafe_categories = set(f.read().split("\n"))

        cafe_filter = partial(filter_by_category, categories=cafe_categories)
        num_reviews_filter = partial(filter_by_num_reviews, min_num_reviews=min_num_reviews)

        filter_raw_business_data([cafe_filter, num_reviews_filter])