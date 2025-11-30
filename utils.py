import json
import random
import os

import pandas as pd

random.seed(42)

def split_reviews(subset):
    if subset:
        file_name = "./datasets/subset/reviews.csv"
    else:
        file_name = "./datasets/processed/reviews.csv"
        valid_size = 20000
        test_size = 20000

    reviews = pd.read_csv(file_name).sample(frac=1, random_state=42)

    if subset:
        valid_size = int(reviews.shape[0] * 0.1)
        test_size = int(reviews.shape[0] * 0.1)
    else:
        valid_size = 20000
        test_size = 20000


    valid_reviews = reviews.iloc[:valid_size].reset_index(drop=True)
    test_reviews = reviews.iloc[valid_size: valid_size + test_size].reset_index(drop=True)
    train_reviews = reviews.iloc[valid_size + test_size:].reset_index(drop=True)

    print(f"train: {train_reviews.shape[0]} / valid: {valid_reviews.shape[0]} / test: {test_reviews.shape[0]}")

    os.makedirs("./datasets/splits", exist_ok=True)
    if subset:
        train_reviews.to_csv("./datasets/splits/train_subset.csv", index=False)
        valid_reviews.to_csv("./datasets/splits/valid_subset.csv", index=False)
        test_reviews.to_csv("./datasets/splits/test_subset.csv", index=False)
    else:
        train_reviews.to_csv("./datasets/splits/train.csv", index=False)
        valid_reviews.to_csv("./datasets/splits/valid.csv", index=False)
        test_reviews.to_csv("./datasets/splits/test.csv", index=False)


def update_metrics(name, train, valid):
    if os.path.exists("./metrics.json"):
        with open("./metrics.json", "r") as f:
           metrics = json.load(f)
    else:
        metrics = {}

    metrics[name] = {"metrics": {"train": train, "valid": valid}}

    with open("./metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    split_reviews()