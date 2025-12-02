import os

from utils import *
from rate_prediction_latent_torch import *
from test_torch import update_test_results
import load_datasets

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import seaborn as sns


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

        for feats in test_dataloader:
            ratings = feats[-1].to(device)
            feats = [f.to(device) for f in feats[:-1]]

            pred_ratings = model(feats)

            batch_size = feats[0].size(0)
            mse += calculate_mse(ratings, pred_ratings).item() * batch_size
            rmse += calculate_rmse(ratings, pred_ratings).item() * batch_size

            pred_discrete = discrete_rating(pred_ratings)

            n_corrects += torch.sum(pred_discrete == ratings).item()
            total += batch_size

        test_mse = mse / total
        test_rmse = rmse / total
        test_accuracy = n_corrects / total

        return {"name": name, "mse": test_mse, "rmse": test_rmse, "accuracy": test_accuracy}


def visualize_cafe_latents_with_pca(model, run_name):
    
    print(f"[{run_name}] Building PCA visualization of cafe latents...")

    # Loading cafe metadata
    cafes, users, reviews = load_datasets.load_table_data()
    num_cafes = len(cafes)

    state = model.state_dict()

    # Finding 2D parameters whose first dimension is close to num_cafes
    candidates = []
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        if tensor.ndim != 2:
            continue
        n_rows, n_cols = tensor.shape
        score = abs(n_rows - num_cafes)
        candidates.append((score, name, n_rows, n_cols, tensor))

    if not candidates:
        print(f"[{run_name}] WARNING: No suitable 2D parameters found in model. "
              "Skipping PCA visualization.")
        return

    # Choosing the parameter that has row-count closest to the number of cafes
    candidates.sort(key=lambda x: x[0])
    best_score, best_name, n_rows, n_cols, best_tensor = candidates[0]

    print(
        f"[{run_name}] Using parameter '{best_name}' with shape "
        f"({n_rows}, {n_cols}) (score={best_score}, num_cafes={num_cafes})."
    )

    item_latents = best_tensor.detach().cpu().numpy()

    # Aligning the embeddings with cafes table
    n = min(n_rows, num_cafes)
    cafe_latents = item_latents[:n]
    cafes_vis = cafes.iloc[:n].copy()

    # Building metadata DataFrame for plotting
    cafe_meta = cafes_vis[["gmap_id", "avg_rating", "price", "name"]].copy()
    cafe_meta["avg_rating"] = pd.to_numeric(cafe_meta["avg_rating"], errors="coerce")

    def price_to_num(p):
        if pd.isna(p):
            return np.nan
        p = str(p).strip()
        if p == "" or p.lower() == "none":
            return np.nan
        return p.count("$") or np.nan

    cafe_meta["price_num"] = cafe_meta["price"].apply(price_to_num)

    if "name" in cafes_vis.columns:
        cafe_meta["name"] = cafes_vis["name"].astype(str)

        # Counting how many times each business name appears
        name_counts = cafe_meta["name"].value_counts()
        cafe_meta["chain_size"] = cafe_meta["name"].map(name_counts)

        CHAIN_SIZE_THRESHOLD = 10 # Considering a chain if the same name appears at least this many times
        cafe_meta["is_chain"] = cafe_meta["chain_size"] >= CHAIN_SIZE_THRESHOLD


        known_chain_keywords = [
            "starbucks",
            "dunkin",
            "dunkin donuts",
            "peet",
            "philz",
            "blue bottle",
            "coffee bean",
            "tim hortons",
            "caribou",
            "costa",
        ]

        name_lower = cafe_meta["name"].str.lower()
        for kw in known_chain_keywords:
            cafe_meta["is_chain"] = cafe_meta["is_chain"] | name_lower.str.contains(
                kw, na=False
            )
    else:
        cafe_meta["is_chain"] = False

    # PCA to 2D
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(cafe_latents)

    cafe_meta["pc1"] = Z[:, 0]
    cafe_meta["pc2"] = Z[:, 1]

    os.makedirs("plots", exist_ok=True)

    # Plot 1: colored by average rating
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        cafe_meta["pc1"],
        cafe_meta["pc2"],
        c=cafe_meta["avg_rating"],
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    plt.colorbar(sc, label="Average rating")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Cafe latent space (PCA) colored by avg rating")
    plt.tight_layout()
    out1 = f"plots/cafe_latent_pca_by_rating_{run_name}.png"
    plt.savefig(out1, dpi=150)
    plt.close()

    # Plot 2: colored by price level
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        cafe_meta["pc1"],
        cafe_meta["pc2"],
        c=cafe_meta["price_num"],
        cmap="plasma",
        alpha=0.6,
        s=10,
    )
    plt.colorbar(sc, label="Price level (# of $)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Cafe latent space (PCA) colored by price")
    plt.tight_layout()
    out2 = f"plots/cafe_latent_pca_by_price_{run_name}.png"
    plt.savefig(out2, dpi=150)
    plt.close()

    # Plot 3: chains vs non-chains
    out3 = None
    if "name" in cafes_vis.columns:
        plt.figure(figsize=(6, 5))

        mask_chain = cafe_meta["is_chain"]
        mask_non_chain = ~mask_chain

        # Non-chains / small places
        plt.scatter(
            cafe_meta.loc[mask_non_chain, "pc1"],
            cafe_meta.loc[mask_non_chain, "pc2"],
            alpha=0.3,
            s=8,
            label="Non-chain / small",
        )

        # Chains
        if mask_chain.any():
            plt.scatter(
                cafe_meta.loc[mask_chain, "pc1"],
                cafe_meta.loc[mask_chain, "pc2"],
                alpha=0.8,
                s=30,
                label="Chains",
            )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Cafe latent space (PCA) - chains vs non-chains")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        out3 = f"plots/cafe_latent_pca_by_chain_{run_name}.png"
        plt.savefig(out3, dpi=150)
        plt.close()
        
    # Plot 4: Starbucks and McDonald's in the same PCA space
    cafe_meta["name_lower"] = cafe_meta["name"].str.lower()

    is_starbucks = cafe_meta["name_lower"].str.contains("starbucks", na=False)
    is_mcdonalds = cafe_meta["name_lower"].str.contains("mcdonald", na=False)

    plt.figure(figsize=(6, 5))
    # All cafes in light gray
    plt.scatter(
        cafe_meta["pc1"],
        cafe_meta["pc2"],
        c="lightgray",
        alpha=0.4,
        s=10,
        label="All cafes",
    )
    # Starbucks in green stars
    plt.scatter(
        cafe_meta.loc[is_starbucks, "pc1"],
        cafe_meta.loc[is_starbucks, "pc2"],
        c="green",
        s=40,
        marker="*",
        label="Starbucks",
    )
    # McDonald's in red X
    plt.scatter(
        cafe_meta.loc[is_mcdonalds, "pc1"],
        cafe_meta.loc[is_mcdonalds, "pc2"],
        c="red",
        s=40,
        marker="x",
        label="McDonald's",
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Cafe latent space (PCA) with Starbucks & McDonald's")
    plt.legend()
    plt.tight_layout()
    out4 = f"plots/cafe_latent_pca_starbucks_mcdonalds_{run_name}.png"
    plt.savefig(out4, dpi=150)
    plt.close()
    
    
    # PCA to 10 dimensions
    pca10 = PCA(n_components=10, random_state=0)
    Z10 = pca10.fit_transform(cafe_latents)

    # Storing PC1–PC10
    for i in range(10):
        cafe_meta[f"pc{i+1}"] = Z10[:, i]

    # Elbow Plot
    explained_var = pca10.explained_variance_ratio_
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 11), explained_var, marker="o")
    plt.xlabel("PCA Dimension")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Elbow Plot of Top-10 PCA Dimensions")
    plt.tight_layout()
    out5 = f"plots/cafe_latent_pca_elbow_{run_name}.png"
    plt.savefig(out5, dpi=150)
    plt.close()

    # Starbucks vs McDonald's Pairplot (Top-10 PCs)
    sb_mask = is_starbucks | is_mcdonalds
    sb_df = cafe_meta.loc[sb_mask, [f"pc{i+1}" for i in range(10)]].copy()

    sb_df["brand"] = np.where(
        is_starbucks.loc[sb_mask], "Starbucks", "McDonald's"
    )

    sns.pairplot(
        sb_df,
        hue="brand",
        corner=True,
        plot_kws={"alpha": 0.7, "s": 18},
        diag_kws={"alpha": 0.7},
    )

    out6 = f"plots/cafe_latent_pca_starbucks_vs_mcdonalds_pairplot_{run_name}.png"
    plt.savefig(out6, dpi=150)
    plt.close()

    print(f"[{run_name}] Saved PCA plots to:")
    print(f"  {out1}")
    print(f"  {out2}")
    print(f"  {out3}")
    print(f"  {out4}")
    print(f"  {out5}")
    print(f"  {out6}")
    

if __name__ == "__main__":
    subset = True

    with open("./params.json", "r") as f:
        params = json.load(f)

    pca_done = False

    for param_dict in params:
        feat = param_dict["feat"]
        feat_names = param_dict["feat_names"]
        latent_names = param_dict["latent_names"]
        lamb_dict = param_dict["lamb_dict"]
        test = param_dict["test"]
        lambs = [lamb_dict[feat] for feat in feat_names + latent_names]

        if not test:
            continue

        lamb_str = "-".join([str(l) for l in lambs])
        name = f"{feat}_{lamb_str}"
        if subset:
            name += "_subset"

        
        batch_size = 2048
        feat_dicts, avg_rating = preprocess_data_latent(feat_names, subset=subset)

        test_dataset = CafeDatasetLatent("test", feat_names, feat_dicts, subset=subset)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        model_path = f"./models/{name}.pt"
        print(f"Loading model from {model_path}")
        model = torch.load(model_path, weights_only=False)

        device = torch.device("cpu")
        result = test_model(name, test_dataloader, model, device)
        
        update_test_results(result)

        # Running PCA visualization once (for the latent model)
        if feat == "latent" and not pca_done:
            visualize_cafe_latents_with_pca(model, name)
            pca_done = True
