import os

from utils import *
from rate_predction import *

if __name__ == "__main__":
    if not os.path.exists("./datasets/splits/train.csv"):
        split_reviews()

    lambs = [4, 4]
    n_epochs = 3
    name = f"base_{lambs[0]}_{lambs[1]}"

    model = BaseRatePredictor(name)
    train_mses, valid_mses = model.fit(lambs, n_epochs)

    os.makedirs("./models")
    with open(f"./models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)