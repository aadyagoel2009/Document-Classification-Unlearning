import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from datasets import load_dataset
import re
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import fetch_20newsgroups

# ------------------------
# Text cleaning
# ------------------------
def load_20newsgroup_dataset():
    """Load and return the 20 Newsgroups dataset."""
    return fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

def char_entropy(word):
    counts = Counter(word)
    total = len(word)
    return -sum((c/total)*math.log((c/total)+1e-12) for c in counts.values())


def clean_data(text):
    text = re.sub(r"\W+", " ", text.lower())
    words = [w for w in text.split() if w.isalpha() and len(w)>=2]
    # filter gibberish
    return [w for w in words if 0.5 <= char_entropy(w) <= 5.0]

# ------------------------
# TF-IDF + Entropy filtering for word selection
# ------------------------
def compute_tfidf_scores(docs, top_words = 60000):
    df = defaultdict(int)
    tfs = []
    cleaned_docs = []
    for doc in docs:
        words = clean_data(doc)
        cleaned_docs.append(words)

        #counts the number of times the word w appears in a given document
        tf = Counter(words)
        tfs.append(tf)

        #counts the number of documents in which the word w appears 
        for w in set(words): df[w] += 1

    print("all words added")
    N = len(docs)
    idf = {w: math.log(N/(1+df[w])) for w in df}
    # compute avg tf-idf per word
    print("about to compute")
    sum_tfidf = defaultdict(float)
    for tf in tfs:
        for w, count in tf.items():
            sum_tfidf[w] += count * idf[w]

    print("computed")
    # now average in O(V)
    avg_tfidf = {w: sum_tfidf[w] / N for w in sum_tfidf}
    top_words_list = sorted(avg_tfidf, key=avg_tfidf.get, reverse=True)[:top_words]
    return top_words_list, cleaned_docs, idf, tfs

# ------------------------
# Build word-document graph
# ------------------------
def build_word_document_graph(cleaned_docs, labels, top_words, tfs, idf):
    G = nx.Graph()
    for i, doc in enumerate(cleaned_docs):
        node = f"doc_{i}"
        G.add_node(node, label=labels[i])

        tf = tfs[i]    
        for w in set(doc):

            if w in top_words:
                G.add_node(w)
                wgt = tf[w] * idf.get(w, 0.0)
                # add a weighted edge instead of unweighted
                G.add_edge(node, w, weight=wgt)
    return G


# ------------------------
# Class importance
# ------------------------
def apply_laplace_smoothing(counts, alpha, k):
    sm = [c+alpha for c in counts]
    s = sum(sm)
    return [v/s for v in sm]

def create_class_importance_nodes(G, labels, alpha=0):
    num_classes = len(set(labels))
    imp = defaultdict(lambda: [0]*num_classes)
    for node in G.nodes():
        if node.startswith("doc_"):
            idx = int(node.split('_')[1])
            cls = labels[idx]

            for neigh in G.neighbors(node):
                wgt = G[node][neigh].get("weight", 1.0)
                imp[neigh][cls] += wgt

    for w, cnts in imp.items(): imp[w] = apply_laplace_smoothing(cnts, alpha, num_classes)
    return imp

# ------------------------
# Classifier & Unlearning
# ------------------------
def train_classifier(imp, idf):
    def softmax(logits):
        shifted = logits - np.max(logits)
        exp_shifted = np.exp(shifted)
        return exp_shifted / exp_shifted.sum()

    def classify(doc):
        tf = Counter(clean_data(doc))
        vec = np.zeros(len(next(iter(imp.values()))))
        
        for w, count in tf.items():
            if w in imp:
                weight = count * idf.get(w, 0.0)
                vec += weight * np.array(imp[w])
        if vec.sum()== 0: return 0, vec.tolist()
        
        probs = softmax(vec)
        u = random.random()                  # uniform in [0,1)
        cumulative_probs = np.cumsum(probs)        # e.g. [0.02, 0.17, 0.82, 1.00]
        pred = int(np.searchsorted(cumulative_probs, u))
        return pred, probs
    
    return classify

def zero_class_importance(importances, cls):
    for w, arr in importances.items():
        arr[cls] = 0
        s = sum(arr)
        importances[w] = [c/s if s else 0 for c in arr]
    return importances

def tsne_visualization(cleaned_docs, labels, class_importances, class_to_unlearn):
    emb, cols = [], []
    for i, doc in enumerate(cleaned_docs):
        vec = np.zeros(len(next(iter(class_importances.values()))))
        for w in set(doc):
            if w in class_importances:
                vec += np.array(class_importances[w])
        vec /= max(len(vec),1)
        emb.append(vec)
        cols.append('red' if labels[i]==class_to_unlearn else 'blue')
    emb = np.array(emb)
    proj = TSNE(n_components=2, random_state=42).fit_transform(emb)
    plt.figure(figsize=(8,6))
    plt.scatter(proj[:,0], proj[:,1], c=cols, alpha=0.6)
    plt.title(f"t-SNE (class {class_to_unlearn} in red)")
    plt.show()

def membership_inference_attack(classify_document, train_docs, test_docs, train_labels, test_labels, cls):
    # balanced sampling
    tr_idx = [i for i,l in enumerate(train_labels) if l==cls]
    te_idx = [i for i,l in enumerate(test_labels ) if l==cls]
    n = len(te_idx)
    sample_tr = random.sample(tr_idx, n)
    X, y = [], []
    for i in sample_tr:
        _, p = classify_document(train_docs[i])
        margin = sorted(p, reverse=True)[0] - sorted(p, reverse=True)[1]
        X.append([max(p), margin, -sum(pi*math.log(pi+1e-12) for pi in p)]); y.append(1)
    for i in te_idx:
        _, p = classify_document(test_docs[i])
        margin = sorted(p, reverse=True)[0] - sorted(p, reverse=True)[1]
        X.append([max(p), margin, -sum(pi*math.log(pi+1e-12) for pi in p)]); y.append(0)
    X = np.array(X); y = np.array(y)
    Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, random_state=42)
    atk = LogisticRegression(class_weight='balanced').fit(Xa, ya)
    auc = roc_auc_score(yt, atk.predict_proba(Xt)[:,1])
    return auc


# ------------------------
# Main workflow with DBpedia
# ------------------------
def main(train_docs, train_labels, test_docs, test_labels):
    # word selection
    top_words, cleaned_docs, idf, tfs = compute_tfidf_scores(train_docs, top_words = 60000)
    print("words chosen")

    # graph & importance
    graph_of_doc = build_word_document_graph(cleaned_docs, train_labels, top_words, tfs, idf)
    print("graph built")
    class_importances = create_class_importance_nodes(graph_of_doc, train_labels)
    print("importances created")
    classify_document = train_classifier(class_importances, idf)
    print("classifier trained")

    # evaluate before unlearning
    test_pred = [classify_document(d)[0] for d in test_docs]
    acc_before = accuracy_score(test_labels, test_pred)

    # unlearning
    class_to_unlearn = random.randint(0, max(train_labels))
    #tsne_visualization(test_docs, test_labels, class_importances, class_to_unlearn)
    mia_before = membership_inference_attack(classify_document, 
        train_docs, test_docs, train_labels, test_labels, class_to_unlearn)

    # zero class importance
    zero_class_importance(class_importances, class_to_unlearn)
    classify_document = train_classifier(class_importances, idf)
    print("retrain classifier")

    test_pred = [classify_document(d)[0] for d in test_docs]
    print(f"Accuracy after unlearning class {class_to_unlearn}:", accuracy_score(test_labels, test_pred))
    docs_no, labels_no = zip(*[(d, l) for d, l in zip(test_docs, test_labels) if l != class_to_unlearn])
    acc_after = accuracy_score(labels_no, [classify_document(d)[0] for d in docs_no])

    #tsne_visualization(test_docs, test_labels, class_importances, class_to_unlearn)
    mia_after = membership_inference_attack(classify_document, 
        train_docs, test_docs, train_labels, test_labels, class_to_unlearn)
    
    print(f"  • Acc  before/after unlearning : {acc_before:.4f} → {acc_after:.4f}")
    print(f"  • MIA AUC before/after        : {mia_before:.4f} → {mia_after:.4f}")
    return acc_before, acc_after, mia_before, mia_after


def run_kfold(k: int = 10, random_state: int = 42):
    ds = load_20newsgroup_dataset()
    all_docs = ds.data
    all_labels = ds.target

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    acc_before, acc_after, mia_before, mia_after = [], [], [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_docs, all_labels), 1):
        print(f"\n===== Fold {fold}/{k} =====")
        train_docs   = [all_docs[i]   for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        test_docs    = [all_docs[i]   for i in val_idx]
        test_labels  = [all_labels[i] for i in val_idx]

        a1, a2, m1, m2 = main(train_docs, train_labels, test_docs, test_labels)
        acc_before.append(a1); acc_after.append(a2)
        mia_before.append(m1); mia_after.append(m2)

    print("\n================  Cross-validated summary  ================")
    print(f"Mean accuracy  before unlearning: {np.mean(acc_before):.4f}")
    print(f"Mean accuracy   after unlearning: {np.mean(acc_after ):.4f}")
    print(f"Mean MIA AUC   before unlearning: {np.mean(mia_before):.4f}")
    print(f"Mean MIA AUC    after unlearning: {np.mean(mia_after ):.4f}")

if __name__ == '__main__':
    run_kfold(k=10) 
