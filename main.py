from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RANDOM_SEED = 42

# Minimum support count for co-occurrence
MIN_COUNT = 20

# Number of top pairs/triples to report
TOP_K = 10

# Minimum and maximum product price
PRICE_MIN = 0.50
PRICE_MAX = 15.00

# Minimum items per basket
MIN_BASKET_SIZE = 2

INPUT_FILE = "groceries.csv"

# Output files
OUTPUT_TRANSACTIONS_CLEAN = "transactions_clean.parquet"
OUTPUT_PRODUCT_PRICES = "product_prices.csv"
OUTPUT_TRANSACTIONS_PRICED = "transactions_priced.csv"

# Visualization files
OUTPUT_VIZ_TOP_ITEMS = "viz1_top15_items.png"
OUTPUT_VIZ_TOP_PAIRS = "viz2_top_pairs.png"
OUTPUT_VIZ_HEATMAP = "viz3_cooccurrence_heatmap.png"
OUTPUT_VIZ_DISTRIBUTIONS = "viz4_distributions.png"

print("Loading groceries database")

transactions = []
with open(INPUT_FILE, "r") as f:
    for idx, line in enumerate(f):
        items = [item.strip() for item in line.strip().split(",") if item.strip()]
        transactions.append(
            {"transaction_id": idx, "items": items, "basket_size": len(items)}
        )

df_transactions = pd.DataFrame(transactions)
print(f"Loaded {len(transactions)} transactions")

print(f"EDA:\nNumber of transactions: {len(df_transactions)}")

all_items = []
for items in df_transactions["items"]:
    all_items.extend(items)

unique_products = set(all_items)
print(f"Number of unique products: {len(unique_products)}")

basket_sizes = df_transactions["basket_size"]
print(f"""Basket Size Distribution:
Minimum: {basket_sizes.min()}
Maximum: {basket_sizes.max()}
Median: {basket_sizes.median():.1f}
95th percentile: {basket_sizes.quantile(0.95):.1f}""")

print("Top 20 Products by Frequency:")
item_counts = Counter(all_items)
top_20_items = item_counts.most_common(20)
for idx, (item, count) in enumerate(top_20_items, 1):
    print(f"{idx:2d}. {item:35s} - {count:4d} occurrences")

print("Standardizing item names...")


def clean_items(items):
    cleaned = []
    for item in items:
        item_clean = item.lower().strip()
        if item_clean:
            cleaned.append(item_clean)
    return cleaned


df_transactions["items_clean"] = df_transactions["items"].apply(clean_items)
df_transactions["basket_size_clean"] = df_transactions["items_clean"].apply(len)

print(f"Before cleaning: {len(all_items)} total items")
all_items_clean = []
for items in df_transactions["items_clean"]:
    all_items_clean.extend(items)
print(f"After cleaning: {len(all_items_clean)} total items")

print(f"""Removing invalid items and small baskets
Baskets before filtering: {len(df_transactions)}""")

# Drop baskets with fewer than MIN_BASKET_SIZE items
df_transactions_clean = df_transactions[
    df_transactions["basket_size_clean"] >= MIN_BASKET_SIZE
].copy()

print(
    f"Baskets after filtering (>={MIN_BASKET_SIZE} items): {len(df_transactions_clean)}"
)
print(f"Removed {len(df_transactions) - len(df_transactions_clean)} baskets")

print("Creating canonical transactions table")
df_canonical = df_transactions_clean[
    ["transaction_id", "items_clean", "basket_size_clean"]
].copy()
df_canonical.columns = ["transaction_id", "items", "basket_size"]

df_canonical["transaction_id"] = range(len(df_canonical))

print(f"""Canonical table shape: {df_canonical.shape}
Columns: {list(df_canonical.columns)}""")

df_canonical.to_parquet(OUTPUT_TRANSACTIONS_CLEAN, index=False)
print(f"Saved to: {OUTPUT_TRANSACTIONS_CLEAN}")

print("Creating product-level price map")

unique_products_clean = set()
for items in df_canonical["items"]:
    unique_products_clean.update(items)

unique_products_list = sorted(list(unique_products_clean))
print(f"Number of unique products: {len(unique_products_list)}")

rng = np.random.default_rng(RANDOM_SEED)
prices = rng.uniform(PRICE_MIN, PRICE_MAX, size=len(unique_products_list))

df_prices = pd.DataFrame({"product": unique_products_list, "price": prices})

df_prices.to_csv(OUTPUT_PRODUCT_PRICES, index=False)
print(f"""Saved product prices to: {OUTPUT_PRODUCT_PRICES}
Price range: ${df_prices["price"].min():.2f} - ${df_prices["price"].max():.2f}""")

price_dict = dict(zip(df_prices["product"], df_prices["price"]))

print("Computing basket totals")

df_canonical["basket_total"] = df_canonical["items"].apply(
    lambda items: sum(price_dict[item] for item in items)
)

print(
    f"""Basket total range: ${df_canonical["basket_total"].min():.2f} - ${df_canonical["basket_total"].max():.2f}
Mean basket total: ${df_canonical["basket_total"].mean():.2f}
Median basket total: ${df_canonical["basket_total"].median():.2f}"""
)

print("Exporting transactions with prices...")

df_export = df_canonical.copy()
df_export["items_string"] = df_export["items"].apply(lambda x: ",".join(x))
df_export_final = df_export[
    ["transaction_id", "items_string", "basket_size", "basket_total"]
]
df_export_final.to_csv(OUTPUT_TRANSACTIONS_PRICED, index=False)
print(f"Saved to: {OUTPUT_TRANSACTIONS_PRICED}")

print("Counting item pairs")

pair_counter = Counter()
for items in df_canonical["items"]:
    if len(items) >= 2:
        for pair in combinations(sorted(items), 2):
            pair_counter[pair] += 1

print(f"Total unique pairs found: {len(pair_counter)}")

pairs_filtered = {
    pair: count for pair, count in pair_counter.items() if count >= MIN_COUNT
}
print(f"Pairs with support >= {MIN_COUNT}: {len(pairs_filtered)}")

total_transactions = len(df_canonical)
pairs_stats = []
for pair, count in pairs_filtered.items():
    pairs_stats.append(
        {
            "item1": pair[0],
            "item2": pair[1],
            "support_count": count,
            "support_fraction": count / total_transactions,
        }
    )

df_pairs = pd.DataFrame(pairs_stats)
df_pairs = df_pairs.sort_values("support_count", ascending=False)

print("Counting item triples")

triple_counter = Counter()
for items in df_canonical["items"]:
    if len(items) >= 3:
        for triple in combinations(sorted(items), 3):
            triple_counter[triple] += 1

print(f"Total unique triples found: {len(triple_counter)}")

triples_filtered = {
    triple: count for triple, count in triple_counter.items() if count >= MIN_COUNT
}
print(f"Triples with support >= {MIN_COUNT}: {len(triples_filtered)}")

triples_stats = []
for triple, count in triples_filtered.items():
    triples_stats.append(
        {
            "item1": triple[0],
            "item2": triple[1],
            "item3": triple[2],
            "support_count": count,
            "support_fraction": count / total_transactions,
        }
    )

if len(triples_stats) > 0:
    df_triples = pd.DataFrame(triples_stats)
    df_triples = df_triples.sort_values("support_count", ascending=False)
else:
    df_triples = pd.DataFrame(
        columns=["item1", "item2", "item3", "support_count", "support_fraction"]
    )
    print(f"Note: No triples meet the minimum support threshold of {MIN_COUNT}")

print(f"Top-{TOP_K} pairs by frequency:")
top_pairs = df_pairs.head(TOP_K)
for idx, row in top_pairs.iterrows():
    print(f"({row['item1']}, {row['item2']})")
    print(
        f"Support count: {row['support_count']}, Support fraction: {row['support_fraction']:.4f}"
    )

print(f"Top-{TOP_K} triples by frequency:")
if len(df_triples) > 0:
    top_triples = df_triples.head(TOP_K)
    for idx, row in top_triples.iterrows():
        print(f"({row['item1']}, {row['item2']}, {row['item3']})")
        print(
            f"Support count: {row['support_count']}, Support fraction: {row['support_fraction']:.4f}"
        )
else:
    print(f"""No triples meet the minimum support threshold of {MIN_COUNT}
Consider lowering MIN_COUNT to find triple associations""")

# Visualize

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

print("Creating bar chart of top 15 items")
all_items_for_viz = []
for items in df_canonical["items"]:
    all_items_for_viz.extend(items)

item_freq = Counter(all_items_for_viz)
top_15 = item_freq.most_common(15)
items_top15, counts_top15 = zip(*top_15)

plt.figure(figsize=(12, 6))
bars = plt.barh(range(len(items_top15)), counts_top15, color="steelblue")

plt.yticks(range(len(items_top15)), items_top15)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Product", fontsize=12)
plt.title("Top 15 Products by Frequency", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_VIZ_TOP_ITEMS, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_VIZ_TOP_ITEMS}")

print("Creating bar chart of top pairs")
top_k_pairs = df_pairs.head(TOP_K).copy()
top_k_pairs["pair_label"] = top_k_pairs.apply(
    lambda x: f"{x['item1'][:15]}+\n{x['item2'][:15]}", axis=1
)

plt.figure(figsize=(12, 6))
bars = plt.barh(range(len(top_k_pairs)), top_k_pairs["support_fraction"], color="coral")
plt.yticks(range(len(top_k_pairs)), top_k_pairs["pair_label"])
plt.xlabel("Support Fraction", fontsize=12)
plt.ylabel("Item Pair", fontsize=12)
plt.title(f"Top {TOP_K} Item Pairs by Support Fraction", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_VIZ_TOP_PAIRS, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_VIZ_TOP_PAIRS}")

print("Creating co-occurrence heatmap")
top_25_items = [item for item, count in item_freq.most_common(25)]

cooc_matrix = np.zeros((25, 25))
for i, item1 in enumerate(top_25_items):
    for j, item2 in enumerate(top_25_items):
        if i <= j:
            pair = tuple(sorted([item1, item2]))
            if pair in pair_counter:
                cooc_matrix[i, j] = pair_counter[pair]
                cooc_matrix[j, i] = pair_counter[pair]

np.fill_diagonal(cooc_matrix, 0)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cooc_matrix,
    xticklabels=[item[:20] for item in top_25_items],
    yticklabels=[item[:20] for item in top_25_items],
    cmap="YlOrRd",
    fmt="g",
    cbar_kws={"label": "Co-occurrence Count"},
)
plt.title("Co-occurrence Heatmap (Top 25 Items)", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_VIZ_HEATMAP, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_VIZ_HEATMAP}")

print("Creating distribution plots")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(
    df_canonical["basket_size"], bins=3, color="skyblue", edgecolor="black", alpha=0.7
)
axes[0].set_xlabel("Basket Size (Number of Items)", fontsize=11)
axes[0].set_ylabel("Frequency", fontsize=11)
axes[0].set_title("Distribution of Basket Size", fontsize=12, fontweight="bold")
axes[0].axvline(
    df_canonical["basket_size"].median(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Median: {df_canonical['basket_size'].median():.1f}",
)
axes[0].legend()

axes[1].hist(
    df_canonical["basket_total"],
    bins=30,
    color="lightcoral",
    edgecolor="black",
    alpha=0.7,
)
axes[1].set_xlabel("Basket Total ($)", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)
axes[1].set_title("Distribution of Basket Total", fontsize=12, fontweight="bold")
axes[1].axvline(
    df_canonical["basket_total"].median(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Median: ${df_canonical['basket_total'].median():.2f}",
)
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_VIZ_DISTRIBUTIONS, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_VIZ_DISTRIBUTIONS}")

print(f"""
Data processed:
Transactions analyzed: {len(df_canonical)}
Unique products: {len(unique_products_clean)}
Item pairs with support >= {MIN_COUNT}: {len(pairs_filtered)}
Item triples with support >= {MIN_COUNT}: {len(triples_filtered)}
""")
