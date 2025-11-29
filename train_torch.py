import os

from utils import *
from rate_predction_torch import *

from torch.utils.data import DataLoader

torch.manual_seed(0)

if __name__ == "__main__":
    subset = True

    if subset:
        if not os.path.exists("./datasets/splits/train_subset.csv"):
            split_reviews(subset)
    else:
        if not os.path.exists("./datasets/splits/train.csv"):
            split_reviews(subset)

    lambs = [4, 8, 2, 1]
    n_epochs = [5, 3, 10]
    lrs = [0.01, 0.001, 0.1]

    params = [(lamb, n_epoch, lr) for lamb in lambs for n_epoch in n_epochs for lr in lrs]

    for (lamb, n_epoch, lr) in params:
        name = f"base_torch_{lamb}_{n_epoch}_{lr}"
        if subset:
            name += "_subset"

        if os.path.exists(f"./models/{name}.pt"):
            continue

        print(f"Start training {name}")
        batch_size = 2048

        user2index, cafe2index, avg_rating = preprocess_data()

        train_dataset = CafeDataset("train", user2index, cafe2index, subset=subset)
        valid_dataset = CafeDataset("valid", user2index, cafe2index, subset=subset)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        model = RatePredictor(name, train_dataset.feat_size, avg_rating)

        device = torch.device("cpu")
        trainer = RateTrainer(model, lamb, lr, train_dataloader, valid_dataloader, device)

        train_mses, valid_mses = trainer.train(n_epoch)

        os.makedirs("./models", exist_ok=True)
        torch.save(model, f"./models/{name}.pt")

        update_metrics(name, train_mses, valid_mses)