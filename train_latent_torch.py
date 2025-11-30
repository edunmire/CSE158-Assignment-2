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

    n_epoch = 10
    lr = 0.01
    dim = 32

    with open("./params.json", "r") as f:
        params = json.load(f)

    for param_dict in params:
        feat = param_dict["feat"]
        feat_names = param_dict["feat_names"]
        latent_names = param_dict["latent_names"]
        latent_pairs = param_dict["latent_pairs"]
        lamb_dict = param_dict["lamb_dict"]

        lambs = [lamb_dict[feat] for feat in feat_names + latent_names]

        assert all([((latent_i in latent_names) and (latent_j in latent_names)) for (latent_i, latent_j) in latent_pairs])
        assert len(lambs) == len(feat_names) + len(latent_names)

        lamb_str = "-".join([str(l) for l in lambs])
        name = f"{feat}_{lamb_str}"
        if subset:
            name += "_subset"

        if os.path.exists(f"./models/{name}.pt"):
            continue

        print(f"Start training {name}")
        batch_size = 2048

        feat_dicts, avg_rating = preprocess_data_latent(feat_names, subset=subset)
        train_dataset = CafeDatasetLatent("train", feat_names, feat_dicts, subset=subset)
        valid_dataset = CafeDatasetLatent("valid", feat_names, feat_dicts, subset=subset)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        feat_sizes = train_dataset.get_feat_sizes()
        model = RatePredictorLatent(name, dim, feat_sizes, latent_names, latent_pairs, avg_rating)

        device = torch.device("cpu")
        trainer = RateTrainerLatent(model, lambs, lr, train_dataloader, valid_dataloader, device)

        train_mses, valid_mses = trainer.train(n_epoch)

        os.makedirs("./models", exist_ok=True)
        torch.save(model, f"./models/{name}.pt")

        update_metrics(name, train_mses, valid_mses)