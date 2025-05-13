import json
from collections import Counter

with open("../testing_datasets/gambling_addict.json") as f:
    data = json.load(f)

categories = Counter()
merchants = Counter()
amounts = []

for tx in data:
    categories[tx.get("category", "Unknown")] += 1
    merchants[tx.get("creditorName", "Unknown")] += 1
    try:
        amounts.append(float(tx["transactionAmount"]["amount"]))
    except Exception:
        pass

print("Unique categories:", len(categories))
print("Category counts:", categories)
print("Unique merchants:", len(merchants))
print("Top 10 merchants:", merchants.most_common(10))
print("Amount range: min =", min(amounts), "max =", max(amounts))