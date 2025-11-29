import os

from utils import *
from rate_prediction_latent_torch import *

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

    lambs = [0, 2, 2, 2, 2]
    n_epoch = 10
    lr = 0.01
    dims = [512, 256, 128, 64, 32, 8, 4, 2]

    feat_names = ["alpha", "user", "cafe"]
    latent_names = ["user", "cafe"]

    params = [(lambs, n_epoch, lr, dim) for dim in dims]

    for (lambs, n_epoch, lr, dim) in params:
        lamb_str = "-".join([str(l) for l in lambs])
        name = f"latent_torch_{lamb_str}_{n_epoch}_{lr}_{dim}"
        if subset:
            name += "_subset"

        if os.path.exists(f"./models/{name}.pt"):
            continue

        print(f"Start training {name}")
        batch_size = 2048

        feat_dicts, avg_rating = preprocess_data_latent(feat_names)
        train_dataset = CafeDatasetLatent("train", feat_names, feat_dicts, subset=subset)
        valid_dataset = CafeDatasetLatent("valid", feat_names, feat_dicts, subset=subset)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        feat_sizes = train_dataset.get_feat_sizes()
        model = RatePredictorLatent(name, dim, feat_sizes,latent_names, avg_rating)

        device = torch.device("cpu")
        trainer = RateTrainerLatent(model, lambs, lr, train_dataloader, valid_dataloader, device)

        train_mses, valid_mses = trainer.train(n_epoch)

        os.makedirs("./models", exist_ok=True)
        torch.save(model, f"./models/{name}.pt")

        update_metrics(name, train_mses, valid_mses)