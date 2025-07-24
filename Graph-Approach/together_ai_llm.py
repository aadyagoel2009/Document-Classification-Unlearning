import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import re
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import fetch_20newsgroups
import json
import time

from sentence_transformers import SentenceTransformer
import openai

# ──────────────────────────────────────────────────────────────────────────────
# TogetherAI client setup
# ──────────────────────────────────────────────────────────────────────────────
client = openai.OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
ENT_MIN, ENT_MAX = 1.0, 4.0                # entropy gate for any char-based filter
DOC_KWS_K = 20                             # how many keywords per document
CLASS_KWS_TOP = 2000                       # how many top terms per class

# ──────────────────────────────────────────────────────────────────────────────
# Utility: character‐level entropy
# ──────────────────────────────────────────────────────────────────────────────
def char_entropy(word):
    counts = Counter(word)
    total = len(word)
    return -sum((c/total)*math.log((c/total)+1e-12) for c in counts.values())

def clean_data(text):
    text = re.sub(r"\W+", " ", text.lower())
    words = [w for w in text.split() if w.isalpha() and len(w)>=2]
    return [w for w in words if 0.5 <= char_entropy(w) <= 5.0]

# ──────────────────────────────────────────────────────────────────────────────
# 1) Per‐class keyword mining via TogetherAI
# ──────────────────────────────────────────────────────────────────────────────
def get_class_keywords(docs, top_n=CLASS_KWS_TOP):
    joined      = "\n\n".join(docs)[:7500]
    instruction = (
        f"Return the {top_n} single- or multi-word terms that best "
        "characterize these documents, as a JSON-style array of lowercase strings."
    )
    full_text = joined + "\n\n" + instruction

    resp = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[
            {"role":"system", "content":"Output only a JSON-style array of strings, no extra text."},
            {"role":"user",   "content": full_text}
        ],
        temperature=0.2,
        max_tokens=3000,
    )

    text = resp.choices[0].message.content
    kws  = re.findall(r'"([^"]+)"', text)
    words = set(kws)
    time.sleep(1)
    return words

# ──────────────────────────────────────────────────────────────────────────────
# 2) Per‐document keyword mining via TogetherAI
# ──────────────────────────────────────────────────────────────────────────────
def build_doc_kws(doc, k=DOC_KWS_K):
    excerpt = doc[:7500]

    instruction = (
        f"List the top {k} single- or multi-word terms that best "
        "characterize this document, one term per line, lowercase, no numbering."
    )
    messages = [
        {"role":"system", "content":"You are a list generator. Return exactly one term per line, no extra text."},
        {"role":"user",   "content": excerpt + "\n\n" + instruction}
    ]

    # 3) Call TogetherAI
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=messages,
        temperature=0.0,
        max_tokens=k * 5,   # a little buffer
    )

    text = resp.choices[0].message.content.strip()

    # 4) Split on newlines, strip whitespace, drop any empties
    terms = [line.strip() for line in text.splitlines() if line.strip()]

    # 5) Convert to set for uniqueness (you lose order, which you said is OK)
    result = set(terms)

    time.sleep(1)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3) Graph & class‐importance machinery (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def build_word_document_graph(cleaned_docs, labels, top_words):
    G = nx.Graph()
    for i, doc in enumerate(cleaned_docs):
        node = f"doc_{i}"
        G.add_node(node, label=labels[i])
        for w in set(doc):
            if w in top_words:
                G.add_node(w)
                G.add_edge(node, w)
    return G

def create_class_importance_nodes(G, labels):
    num_classes = len(set(labels))
    imp = defaultdict(lambda: [0] * num_classes)
    for node in G.nodes():
        if node.startswith("doc_"):
            idx = int(node.split("_")[1])
            cls = labels[idx]
            for neigh in G.neighbors(node):
                imp[neigh][cls] += 1

    return imp


# ──────────────────────────────────────────────────────────────────────────────
# 4) Classifier & unlearning (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def train_classifier(imp, word_embeddings, embed_model, idf):
    def clean_data(text):
        text = re.sub(r"\W+", " ", text.lower())
        words = [w for w in text.split() if w.isalpha() and len(w) >= 3]
        return [w for w in words if ENT_MIN <= char_entropy(w) <= ENT_MAX]

    def softmax(logits):
        shifted = logits - np.max(logits)
        exp_shifted = np.exp(shifted)
        return exp_shifted / exp_shifted.sum()

    def classify(doc):
        # embed the doc locally
        raw = embed_model.encode(doc, convert_to_numpy=True, normalize_embeddings=True)
        norm = np.linalg.norm(raw)
        doc_emb = raw / norm if norm > 0 else raw

        # TF × sim × IDF
        tf = Counter(clean_data(doc))
        vec = np.zeros(len(next(iter(imp.values()))))
        for w, count in tf.items():
            if w in imp:
                sim = float(np.dot(doc_emb, word_embeddings[w]))
                sim = max(sim, 0.0)
                tf_log = 1 + math.log(count)
                weight = sim * tf_log * idf[w]
                vec += weight * np.array(imp[w])

        # plus LLM‐driven doc keywords
        for w in build_doc_kws(doc, k=5):
            if w in imp:
                vec += np.array(imp[w])

        if vec.sum() == 0:
            return 0, vec.tolist()
        probs = softmax(vec)
        u = random.random()
        cumulative = np.cumsum(probs)
        pred = int(np.searchsorted(cumulative, u))
        return pred, probs

    return classify

def zero_class_importance(importances, cls):
    for w, arr in importances.items():
        arr[cls] = 0
        s = sum(arr)
        importances[w] = [c / s if s else 0 for c in arr]
    return importances


# ──────────────────────────────────────────────────────────────────────────────
# 5) Visualization & MIA (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def tsne_visualization(cleaned_docs, labels, class_importances, class_to_unlearn):
    emb, cols = [], []
    for i, doc in enumerate(cleaned_docs):
        vec = np.zeros(len(next(iter(class_importances.values()))))
        for w in set(doc):
            if w in class_importances:
                vec += np.array(class_importances[w])
        vec /= max(len(vec), 1)
        emb.append(vec)
        cols.append("red" if labels[i] == class_to_unlearn else "blue")
    emb = np.array(emb)
    proj = TSNE(n_components=2, init="random", random_state=42, n_jobs=1).fit_transform(emb)
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=cols, alpha=0.6)
    plt.title(f"t-SNE (class {class_to_unlearn} in red)")
    plt.show()

def membership_inference_attack(classify_document, train_docs, test_docs, train_labels, test_labels, cls):
    tr_idx = [i for i, l in enumerate(train_labels) if l == cls]
    te_idx = [i for i, l in enumerate(test_labels) if l == cls]
    n = len(te_idx)
    sample_tr = random.sample(tr_idx, n)
    X, y = [], []
    for i in sample_tr + te_idx:
        _, p = classify_document((train_docs if i in sample_tr else test_docs)[i % n])
        margin = sorted(p, reverse=True)[0] - sorted(p, reverse=True)[1]
        label = 1 if i in sample_tr else 0
        X.append([max(p), margin, -sum(pi * math.log(pi + 1e-12) for pi in p)])
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, random_state=42)
    atk = LogisticRegression(class_weight="balanced").fit(Xa, ya)
    auc = roc_auc_score(yt, atk.predict_proba(Xt)[:, 1])
    return auc


# ──────────────────────────────────────────────────────────────────────────────
# 6) Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_20newsgroup_dataset():
    return fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))

# ──────────────────────────────────────────────────────────────────────────────
# 7) Main workflow
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # a) load & split
    dataset = load_20newsgroup_dataset()
    documents, labels = dataset.data, dataset.target
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(documents, labels))
    train_docs = [documents[i] for i in train_idx]
    test_docs = [documents[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    # b) mine per-class keywords via LLM
    class_names = fetch_20newsgroups(subset="all").target_names
    keywords_by_class = {}
    for cid in range(len(set(labels))):
        cls_docs = [train_docs[i] for i, l in enumerate(train_labels) if l == cid]
        keywords_by_class[cid] = set(get_class_keywords(cls_docs))
    top_words = sorted(set().union(*keywords_by_class.values()))
    print(f"{len(top_words)} LLM keywords gathered.")

    # c) build cleaned docs & IDF
    cleaned_docs = []
    for doc in train_docs:
        words = clean_data(doc)
        cleaned_docs.append(words)

    idf = {
        w: math.log(len(cleaned_docs) / (1 + sum(w in doc for doc in cleaned_docs)))
        for w in top_words
    }

    # d) graph & class importances
    graph_of_doc = build_word_document_graph(cleaned_docs, train_labels, top_words)
    class_importances = create_class_importance_nodes(graph_of_doc, train_labels)

    # e) local embeddings of top_words
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embed_model.max_seq_length = 512
    word_embeddings = {}
    for w in top_words:
        emb = embed_model.encode(w, convert_to_numpy=True, normalize_embeddings=True)
        norm = np.linalg.norm(emb)
        word_embeddings[w] = (emb / norm) if norm > 0 else emb

    # f) assemble classifier
    classify_document = train_classifier(class_importances, word_embeddings, embed_model, idf)

    # g) evaluate pre-unlearning
    test_pred = [classify_document(d)[0] for d in test_docs]
    print("Accuracy before unlearning:", accuracy_score(test_labels, test_pred))

    # h) visualize & MIA
    cls_to_unlearn = random.randint(0, max(train_labels))
    tsne_visualization(test_docs, test_labels, class_importances, cls_to_unlearn)
    print("MIA AUC before:", membership_inference_attack(
        classify_document, train_docs, test_docs, train_labels, test_labels, cls_to_unlearn))

    # i) unlearn & re-evaluate
    zero_class_importance(class_importances, cls_to_unlearn)
    classify_document = train_classifier(class_importances, word_embeddings, embed_model, idf)
    print("Retrained classifier after unlearning")

    test_pred = [classify_document(d)[0] for d in test_docs]
    print(f"Accuracy after unlearning class {cls_to_unlearn}:",
          accuracy_score(test_labels, test_pred))

    docs_no, labels_no = zip(*[
        (d, l) for d, l in zip(test_docs, test_labels) if l != cls_to_unlearn
    ])
    print("Post-Unlearning Acc (without that class):",
          accuracy_score(labels_no, [classify_document(d)[0] for d in docs_no]))

    tsne_visualization(test_docs, test_labels, class_importances, cls_to_unlearn)
    print("MIA AUC after:", membership_inference_attack(
        classify_document, train_docs, test_docs, train_labels, test_labels, cls_to_unlearn))


if __name__ == "__main__":
    main()