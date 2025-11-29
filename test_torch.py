import os

from utils import *
from rate_predction_torch import *

from torch.utils.data import DataLoader

def calculate_mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def discrete_rating(y_pred):
    y_pred = torch.clamp(y_pred, min=0, max=5)
    y_pred = torch.round(y_pred)

    return y_pred

def test_model(name, test_dataloader, model, device):
    with torch.no_grad():
        total = 0
        mse, rmse = 0, 0
        n_corrects = 0

        model.to(device)
        model.eval()

        for (feats, ratings) in test_dataloader:
            feats = feats.to(device)
            ratings = ratings.to(device)

            pred_ratings = model(feats)

            batch_size = feats.size(0)
            mse += calculate_mse(ratings, pred_ratings).item() * batch_size
            rmse += calculate_rmse(ratings, pred_ratings).item() * batch_size

            pred_discrete = discrete_rating(pred_ratings)

            n_corrects += torch.sum(pred_discrete == ratings).item()
            total += batch_size

        test_mse = mse / total
        test_rmse = rmse / total
        test_accuracy = n_corrects / total

        return {"name": name, "mse": test_mse, "rmse": test_rmse, "accuracy": test_accuracy}

def update_test_results(new_result):
    new_result = pd.Series(new_result).to_frame().T

    if os.path.exists("./test_results.csv"):
        results = pd.read_csv("./test_results.csv")

        duplicate_index = results["name"] == new_result["name"]
        if sum(duplicate_index) == 0:
            results = pd.concat([results, new_result]).reset_index(drop=True)
        else:
            results[duplicate_index] = new_result
    else:
        results = new_result

    print(results)
    results.to_csv("./test_results.csv", index=False)

if __name__ == "__main__":
    subset = True
    name = "base_torch_4_subset"

    batch_size = 2048
    user2index, cafe2index, avg_rating = preprocess_data()

    test_dataset = CafeDataset("test", user2index, cafe2index, subset=subset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if name == "naive":
        model = RatePredictor(name, test_dataset.feat_size, avg_rating)
    else:
        model = torch.load(f"./models/{name}.pt", weights_only=False)

    device = torch.device("cpu")
    result = test_model(name, test_dataloader, model, device)

    update_test_results(result)