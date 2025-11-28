from collections import defaultdict
import json

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def filter_by_bigrams():
    freq_thresh = 100
    num_occurences = 5

    cafes = pd.read_csv("./datasets/processed/cafes.csv")
    names, counts = np.unique(cafes["name"], return_counts=True)
    indices = np.argsort(counts)[::-1]
    sorted_counts = counts[indices]
    sorted_names = names[indices]

    unigrams = defaultdict(int)
    bigrams = defaultdict(int)
    for name, count in zip(sorted_names, sorted_counts):
        words = name.lower().strip().split()

        for i in range(len(words)):
            unigrams[words[i]] += 1

        for i in range(len(words) - 1):
            bigrams[f"{words[i]} {words[i+1]}"] += 1

    unigram_words = np.array([w for (w, _) in unigrams.items()])
    unigram_counts = np.array([c for (_, c) in unigrams.items()])
    bigram_words = np.array([w for (w, _) in bigrams.items()])
    bigram_counts = np.array([c for (_, c) in bigrams.items()])

    unigram_words = unigram_words[unigram_counts >= freq_thresh]
    unigram_counts = unigram_counts[unigram_counts >= freq_thresh]
    unigram_indices = np.argsort(unigram_counts)[::-1]
    popular_unigrams = unigram_words[unigram_indices]


    for bigram, count in zip(bigram_words,  bigram_counts):
        if count > num_occurences:
            words = bigram.split()
            common = (words[0] in popular_unigrams) or (words[1] in popular_unigrams)
            if not common:
                print(bigram)

def filter_by_stem():
    freq_thresh = 100
    num_occurences = 5

    cafes = pd.read_csv("./datasets/processed/cafes.csv")
    names, counts = np.unique(cafes["name"], return_counts=True)
    indices = np.argsort(counts)[::-1]
    sorted_counts = counts[indices]
    sorted_names = names[indices]

    stems = defaultdict(int)
    for name in sorted_names:
        words = name.lower().strip().split()

        for i in range(len(words)):
            stems[" ".join(words[:(i+1)])] += 1

    chains = {}
    chains_count = [0, 0, 0]
    for name, count in zip(sorted_names, sorted_counts):
        if count > 5:
            chains[name] = 2
            chains_count[2] += int(count)
            continue

        words = name.lower().strip().split()
        stem_matches = []
        for i in range(len(words)):
            stem_matches.append(stems[" ".join(words[:(i+1)])])

        if len(stem_matches) == 1 or len(stem_matches) > 10:
            chains[name] = 0
            chains_count[0] += int(count)
            continue

        if len(stem_matches) >= 2 and sum(stem_matches[1:]) < 10:
            chains[name] = 0
            chains_count[0] += int(count)
            continue

        if len(stem_matches) >= 3 and sum(stem_matches[2:]) < 5:
            chains[name] = 0
            chains_count[0] += int(count)
            continue

        chains[name] = 1
        chains_count[1] += int(count)

    big_chains_counts = sum([1 for _, c in chains.items() if c == 2])
    small_chains_counts = sum([1 for _, c in chains.items() if c == 1])
    non_chains_counts = sum([1 for _, c in chains.items() if c == 0])
    print(f"The number of big chains: {big_chains_counts}")
    print(f"The number of sub chains: {small_chains_counts}")
    print(f"The number of non chains: {non_chains_counts}")

    with open("./datasets/processed/chains.json", "w") as f:
        json.dump(chains, f)

def plot_avg_rating_by_chain():
    with open("./datasets/processed/chains.json", "r") as f:
        chains = json.load(f)

    cafes = pd.read_csv("./datasets/processed/cafes.csv")
    ratings = defaultdict(list)
    for name, rating in cafes[["name", "avg_rating"]].values:
        ratings[chains[name]].append(rating)

    categories = ["Non Chain", "Sub Chains", "Chains"]
    data = []
    for key, ratings in ratings.items():
        data.append((key, categories[key], sum(ratings) / len(ratings)))
    data = pd.DataFrame(data, columns=["key", "category", "avg_rating"])
    data = data.sort_values(by="key")

    sns.set_theme(style="whitegrid", palette="viridis")
    sns.barplot(data, x="category", y="avg_rating")

    plt.title("Average Rating of Cafes by Non Chains, Sub Chains, and Chains")
    plt.xlabel("Chain")
    plt.ylabel("Average Rating")
    plt.ylim((3.5, 4.5))
    plt.savefig("./avg_rating_by_chains.png")


if __name__ == "__main__":
    # filter_by_stem()
    plot_avg_rating_by_chain()