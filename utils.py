import json
import random
import os

import pandas as pd

random.seed(42)

def split_reviews():
    file_name = "./datasets/processed/reviews.csv"
    reviews = pd.read_csv(file_name).sample(frac=1, random_state=42)

    valid_size = 20000
    test_size = 20000

    valid_reviews = reviews.iloc[:valid_size].reset_index(drop=True)
    test_reviews = reviews.iloc[valid_size: valid_size + test_size].reset_index(drop=True)
    train_reviews = reviews.iloc[valid_size + test_size:].reset_index(drop=True)

    print(f"train: {train_reviews.shape[0]} / valid: {valid_reviews.shape[0]} / test: {test_reviews.shape[0]}")

    os.makedirs("./datasets/splits", exist_ok=True)
    train_reviews.to_csv("./datasets/splits/train.csv", index=False)
    valid_reviews.to_csv("./datasets/splits/valid.csv", index=False)
    test_reviews.to_csv("./datasets/splits/test.csv", index=False)

def update_metrics(name, train, valid, params):
    if os.path.exists("./metrics.json"):
        with open("./metrics.json", "r") as f:
           metrics = json.load(f)
    else:
        metrics = {}

    metrics[name] = {"metrics": {"train": train, "valid": valid}, "params": params}

    with open("./metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    split_reviews()