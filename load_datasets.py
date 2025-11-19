from collections import defaultdict

import pandas as pd

def load_table_data():
    cafes = pd.read_csv("./datasets/processed/cafes.csv")
    users = pd.read_csv("./datasets/processed/users.csv")
    reviews = pd.read_csv("./datasets/processed/reviews.csv")

    return cafes, users, reviews

def create_user_review_dicts(reviews):
    ratings = []
    user2cafes = defaultdict(list)
    cafe2users = defaultdict(list)

    for (user, cafe, rating) in reviews[["user_id", "gmap_id", "rating"]].values:
        ratings.append((user, cafe, rating))
        user2cafes[user].append((cafe, rating))
        cafe2users[cafe].append((user, rating))


    return ratings, user2cafes, cafe2users

if __name__ == "__main__":
    cafes, users, reviews = load_table_data()

    print(cafes.head())
    print(users.head())
    print(reviews.head())

    ratings, user2cafes, cafe2users = create_user_review_dicts(reviews)

    print(len(ratings))