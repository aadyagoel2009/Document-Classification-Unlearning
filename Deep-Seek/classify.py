import os
import re
import math
import pickle
from pathlib import Path
import random
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.manifold import TSNE

# ──────────────────────────────────────────────────────────────────────────────
# Import your cache‐paths from the build_cache module
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).resolve().parent
CLASS_CACHE         = BASE_DIR / "class_imp.pkl"
TEST_SCORES_CACHE   = BASE_DIR / "test_scores.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# Load the cached class‐importance and per‐doc scores
# ──────────────────────────────────────────────────────────────────────────────
with open(CLASS_CACHE, "rb") as f:
    class_imp: dict[str, list[float]] = pickle.load(f)

with open(TEST_SCORES_CACHE, "rb") as f:
    test_scores: dict[int, list[tuple[str, float]]] = pickle.load(f)

# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────
def char_entropy(word: str) -> float:
    counts = Counter(word)
    total = len(word)
    return -sum((c/total) * math.log((c/total) + 1e-12) for c in counts.values())

def clean_data(text: str) -> list[str]:
    text = re.sub(r"\W+", " ", text.lower())
    words = [w for w in text.split() if w.isalpha() and len(w) >= 2]
    return [w for w in words if 1.0 <= char_entropy(w) <= 4.0]

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

# ──────────────────────────────────────────────────────────────────────────────
# Classifier using only cached scores (no LLM calls)
# ──────────────────────────────────────────────────────────────────────────────
def classify(doc_idx: int) -> tuple[int, np.ndarray]:
    """
    Given the index of a test document, build its class‐score vector
    from the precomputed `test_scores` and `class_imp` caches.
    """
    scores = test_scores.get(doc_idx, [])
    num_classes = len(next(iter(class_imp.values())))
    vec = np.zeros(num_classes)
    for term, score in scores:
        vec += score * np.array(class_imp.get(term, [0.0]*num_classes))
    if vec.sum() == 0:
        return 0, vec
    probs = softmax(vec)
    return int(probs.argmax()), probs

"""
# ──────────────────────────────────────────────────────────────────────────────
# Membership Inference Attack --> needs fixing 
# ──────────────────────────────────────────────────────────────────────────────
def membership_inference_attack(train_docs, test_docs, train_labels, test_labels, target_class: int) -> float:
    X, y = [], []
    # In‐class training samples → label=1
    for i, lbl in enumerate(train_labels):
        if lbl == target_class:
            # if this train doc wasn't in test set, fallback to zero‐vector
            _, p = classify(i - len(train_docs)) if i >= len(train_docs) else (0, np.zeros(len(next(iter(class_imp.values())))))
            margin = np.partition(p, -2)[-1] - np.partition(p, -3)[-2] if len(p) > 1 else 0.0
            X.append([p.max(), margin, -sum(pi*math.log(pi+1e-12) for pi in p)])
            y.append(1)
    # Out‐of‐class test samples → label=0
    for j, lbl in enumerate(test_labels):
        if lbl == target_class:
            _, p = classify(j)
            margin = np.partition(p, -2)[-1] - np.partition(p, -3)[-2] if len(p) > 1 else 0.0
            X.append([p.max(), margin, -sum(pi*math.log(pi+1e-12) for pi in p)])
            y.append(0)

    Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, random_state=42)
    atk = LogisticRegression(class_weight="balanced").fit(Xa, ya)
    return roc_auc_score(yt, atk.predict_proba(Xt)[:, 1])
"""

# ──────────────────────────────────────────────────────────────────────────────
# t-SNE Visualization of test docs
# ──────────────────────────────────────────────────────────────────────────────
def tsne_visualization(docs, labels, cls_to_unlearn: int):
    points, colors = [], []
    num_classes = len(next(iter(class_imp.values())))
    for idx, (doc, lbl) in enumerate(zip(docs, labels)):
        vec = np.zeros(num_classes)
        for term, score in test_scores.get(idx, []):
            vec += score * np.array(class_imp.get(term, [0.0]*num_classes))
        points.append(vec)
        colors.append("red" if lbl == cls_to_unlearn else "blue")

    proj = TSNE(n_components=2, init="random", random_state=42).fit_transform(np.array(points))
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.6)
    plt.title(f"t-SNE (class {cls_to_unlearn} in red)")
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Load & split exactly as in build_cache.py
    data = fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))
    docs, labels = data.data, data.target
    split_index = int(0.8 * len(docs))
    train_docs, test_docs = docs[:split_index], docs[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]

    # Evaluate before unlearning
    preds = [classify(i)[0] for i in range(len(test_docs))]
    print("Accuracy before unlearning:", accuracy_score(test_labels, preds))

    # Choose a class to unlearn and visualize + MIA before
    cls = random.randint(0, max(train_labels))
    tsne_visualization(test_docs, test_labels, cls)
    #print("MIA AUC before unlearning:", membership_inference_attack(train_docs, test_docs, train_labels, test_labels, cls))

    # Unlearn by zeroing out that class in class_imp
    for term, vec in class_imp.items():
        vec[cls] = 0.0
        s = sum(vec)
        class_imp[term] = [v/s for v in vec] if s > 0 else [0.0]*len(vec)

    # Re-evaluate after unlearning
    preds2 = [classify(i)[0] for i in range(len(test_docs))]
    print(f"Accuracy after unlearning class {cls}:", accuracy_score(test_labels, preds2))

    # Accuracy on remaining classes only
    docs_no = [d for d,l in zip(test_docs, test_labels) if l != cls]
    labels_no = [l for l in test_labels if l != cls]
    preds3 = [classify(i)[0] for i,(d,l) in enumerate(zip(test_docs, test_labels)) if l != cls]
    print("Post-unlearning (without that class) acc:", accuracy_score(labels_no, preds3))

    # Visualize & MIA again
    tsne_visualization(test_docs, test_labels, cls)
    #print("MIA AUC after unlearning:", membership_inference_attack(train_docs, test_docs, train_labels, test_labels, clsert)

if __name__ == "__main__":
    main()