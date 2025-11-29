import os

from utils import *
from rate_predction_torch import *

from torch.utils.data import DataLoader

if __name__ == "__main__":
    if not os.path.exists("./datasets/splits/train.csv"):
        split_reviews()

    lamb = 4
    n_epochs = 3
    name = f"base_torch_{lamb}"
    batch_size = 8192

    user2index, cafe2index, avg_rating = preprocess_data()

    train_dataset = CafeDataset("train", user2index, cafe2index)
    valid_dataset = CafeDataset("valid", user2index, cafe2index)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    model = RatePredictor(name, train_dataset.feat_size, avg_rating)

    device = torch.device("cpu")
    trainer = RateTrainer(model, lamb, train_dataloader, valid_dataloader, device)

    train_mses, valid_mses = trainer.train(n_epochs)

    os.makedirs("./models", exist_ok=True)
    torch.save(f"./models/{name}.pt")

    update_metrics(name, train_mses, valid_mses)