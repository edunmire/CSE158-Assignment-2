import json
import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
from shapely.geometry import Point
from functools import lru_cache

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import r2_score

@lru_cache(maxsize=1)
def get_counties_ca():
    counties = gpd.read_file("resources/cb_2018_us_county_500k.shp")
    counties_ca = counties[counties["STATEFP"] == "06"]  # California only
    counties_ca = counties_ca.sort_values("NAME").reset_index(drop=True)
    counties_ca["COUNTY_NUM"] = counties_ca.index
    counties_ca = counties_ca.set_geometry("geometry")
    _ = counties_ca.sindex
    return counties_ca

def get_county(lat, lon):
    counties_ca = get_counties_ca()
    point = Point(lon, lat)  # geometry expects (lon, lat)
    idx = list(counties_ca.sindex.intersection(point.bounds))
    if not idx:
        return None
    candidates = counties_ca.iloc[idx]
    matches = candidates[candidates.contains(point)]
    return int(matches.iloc[0]["COUNTY_NUM"]) if len(matches) else None

def price_to_num(p):
    if pd.isna(p):
        return np.nan

    p = str(p).strip()

    if p == "" or p.lower() == "none":
        return np.nan

    return p.count("$") or np.nan

def load_cafe_dataset(subset):
    if subset:
        cafes = pd.read_csv("./datasets/subset/cafes.csv")
    else:
        cafes = pd.read_csv("./datasets/processed/cafes.csv")

    return cafes

def get_cafe2index(cafes):
    unique_gmap_ids = np.sort(np.unique(cafes["gmap_id"]))
    cafe2index = {gmap_id: index for index, gmap_id in enumerate(unique_gmap_ids)}
    return cafe2index

def get_cafe_latents(name):
    model_path = f"./models/{name}.pt"
    model = torch.load(model_path, weights_only=False)
    cafe_latents = model.latents["cafe"]

    return cafe_latents

def find_famous_chains(name):
    if "starbucks" in name.lower():
        return "Starbucks"

    elif "mcdonald" in name.lower():
        return "McDonald's"

    else:
        return "Others"

def get_category_mappings(cafes, category):
    if category == "chain":
        with open("./datasets/processed/chains.json", "r") as f:
            chains = json.load(f)

        chain_map = {0: "No Chain", 1: "Chain", 2: "Chain"}
        cafe2category = {gmap_id: chain_map[chains[name]] for gmap_id, name in cafes[["gmap_id", "name"]].values}

    elif category == "price":
        cafes["price_num"] = cafes["price"].apply(price_to_num)

        cafe2category = {gmap_id: str(price) for gmap_id, price in cafes[["gmap_id", "price_num"]].values}

    elif category == "famous_chain":
        with open("./datasets/processed/chains.json", "r") as f:
            chains = json.load(f)

        cafe2category = {gmap_id: find_famous_chains(name) for gmap_id, name in cafes[["gmap_id", "name"]].values}

    elif category == "avg_rating":
        with open("./datasets/processed/chains.json", "r") as f:
            chains = json.load(f)

        cafe2category = {gmap_id: avg_rating for gmap_id, avg_rating in cafes[["gmap_id", "avg_rating"]].values}

    elif category == "county":
        cafe2category = {gmap_id: get_county(latitude, longitude) for (gmap_id, latitude, longitude) in cafes[["gmap_id", "latitude", "longitude"]].values}
    else:
        raise NotImplementedError

    return cafe2category

def analyze_pca(name, category, subset):
    print(f"Analyzing PCA for {category}")
    cafes = load_cafe_dataset(subset)
    cafe2index = get_cafe2index(cafes)

    latents = get_cafe_latents(name)
    cafe2latent = {gmap_id: latents[index].detach().numpy() for gmap_id, index in cafe2index.items()}

    cafe2category = get_category_mappings(cafes, category)

    latents, categories = [], []

    for gmap_id in cafes["gmap_id"].values:
        latents.append(cafe2latent[gmap_id])
        categories.append(cafe2category[gmap_id])

    n_components = 10

    train_classifier(latents, category, categories, name)

    pca = PCA(n_components=n_components, random_state=0)
    components = pca.fit_transform(latents)


    data_dict = {f"component_{i}": components[:, i] for i in range(n_components)}
    data_dict["category"] = categories
    df = pd.DataFrame(data_dict)

    os.makedirs(f"./pca/{name}", exist_ok=True)

    sns.set_theme(style="whitegrid", palette="tab10")
    plot_elbow(components, name)
    plot_top_two_componets(df, name, category)
    plot_pairs(df, name, category, n_components)

def plot_elbow(components, name):
    stds = np.std(components, axis=0)

    df = pd.DataFrame({"std": stds, "component": np.arange(stds.shape[0])})

    plt.figure(figsize=(10, 5))

    sns.lineplot(df, x="component", y="std")
    sns.scatterplot(df, x="component", y="std")

    plt.title("Elbow plot for PCA")
    plt.xlabel("Components sorted by their standard deviations")
    plt.ylabel("Standard Deviations")
    plt.tight_layout()
    plt.savefig(f"./pca/{name}/elbow.png")
    plt.close()

def plot_top_two_componets(df, name, category):
    plt.figure(figsize=(20, 10))

    if category == "avg_rating":
        ax = sns.scatterplot(df, x="component_0", y="component_1", hue="category", palette="inferno", alpha=0.4)
        norm = plt.Normalize(2, 5)
        plt.colorbar(
            plt.cm.ScalarMappable(cmap="inferno", norm=norm),
            ax=plt.gca()
        )
        ax.get_legend().remove()

    elif category == "chain" or category == "famous_chain":
        sns.scatterplot(df, x="component_0", y="component_1", style="category", hue="category", alpha=0.4)

    else:
        sns.scatterplot(df, x="component_0", y="component_1", hue="category", alpha=0.4)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Cafe latent space (PCA) colored by {category}")
    plt.tight_layout()
    plt.savefig(f"./pca/{name}/top_two_{category}.png")
    plt.close()

def plot_pairs(df, name, category, n_components):
    variables = [f"component_{i}" for i in range(n_components)][:5]

    plt.figure(figsize=(50, 50))

    if category == "avg_rating":
        sns.pairplot(df, vars=variables, hue="category", palette="inferno", plot_kws={
            "alpha": 0.4, "s": 10, "palette": "inferno"})

    else:
        sns.pairplot(df, vars=variables, hue="category", plot_kws={"alpha": 0.4, "s": 10})

    plt.suptitle(f"Pair plots of components colored by {category}")
    plt.savefig(f"./pca/{name}/pair_{category}.png")
    plt.close()

def train_classifier(latents, category, categories, name):
    model = LogisticRegression()

    latents = np.array(latents)

    if category == "avg_rating":
        model = LinearRegression()
        model = model.fit(latents, categories)

        preds = model.predict(latents)
        r2 = r2_score(categories, preds)

        print(f"R^2 score for Lienar Regression to predict {category} is {r2}")

    elif category in ["chain", "price", "famous_chain"]:
        model = LogisticRegression()
        model = model.fit(latents, categories)

        preds = model.predict(latents)
        accuracy = np.sum(preds == categories) / preds.shape[0]

        print(f"Accuracy for Logistic Regression to predict {category} is {accuracy}")

        model = SVC()
        model = model.fit(latents, categories)

        preds = model.predict(latents)
        accuracy = np.sum(preds == categories) / preds.shape[0]

        print(f"Accuracy for SVM to predict {category} is {accuracy}")


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

    name = "final_alpha-0_user-0.1_cafe-1_chains-0.1_price-0.1_open_hours-0.1_period-0.1_subset"
    analyze_pca(name, "county", subset)
    # analyze_pca(name, "avg_rating", subset)
    # analyze_pca(name, "chain", subset)
    # analyze_pca(name, "price", subset)
    # analyze_pca(name, "famous_chain", subset)