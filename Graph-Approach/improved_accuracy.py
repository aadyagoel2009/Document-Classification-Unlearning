import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
import re
import math
import random

# ------------------------
# Stop words
# ------------------------
stop_words = set([
    "i","me","my","myself","we","our","ours","ourselves","you","your",
    "yours","yourself","yourselves","he","him","his","himself","she","her",
    "hers","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","s","t","can","will","just","don","should","now","us",
    "much","get","well","would","may","could","however","without","never"
])

# ------------------------
# Text cleaning and entropy filter
# ------------------------
def char_entropy(word):
    counts = Counter(word)
    total = len(word)
    return -sum((c/total)*math.log((c/total)+1e-12) for c in counts.values())

def clean_data(text):
    text = re.sub(r"\W+", " ", text.lower())
    words = [w for w in text.split() if w.isalpha() and w not in stop_words and len(w)>=3]
    # Filter out gibberish by entropy
    return [w for w in words if 1.0 <= char_entropy(w) <= 4.0]

# ------------------------
# TF-IDF computation
# ------------------------
def compute_tfidf_scores(docs):
    df = defaultdict(int)
    tfs = []
    for doc in docs:
        words = clean_data(doc)
        tf = Counter(words)
        tfs.append(tf)
        for w in set(words): df[w]+=1
    N = len(docs)
    idf = {w: math.log(N/(1+df[w])) for w in df}
    return tfs, idf

# ------------------------
# Balanced class-aware word selection
# ------------------------
def select_top_words_by_class(docs, labels, tfs, idf, per_class=3000):
    classwise = defaultdict(lambda: Counter())
    for tf, label in zip(tfs, labels):
        for w, cnt in tf.items():
            classwise[label][w] += cnt * idf.get(w,0)
    # select top per class
    selected = set()
    for cls, counter in classwise.items():
        for w,_ in counter.most_common(per_class): selected.add(w)
    return selected

# ------------------------
# Build word-document graph
# ------------------------
def build_graph(docs, labels, top_words):
    G = nx.Graph()
    for i, doc in enumerate(docs):
        node = f"doc_{i}"
        G.add_node(node, label=labels[i])
        words = set(clean_data(doc))
        for w in words:
            if w in top_words:
                G.add_node(w)
                G.add_edge(node, w)
    return G

# ------------------------
# Class importance
# ------------------------
def apply_laplace_smoothing(counts, alpha, k):
    sm = [c+alpha for c in counts]
    s = sum(sm)
    return [v/s for v in sm]

def create_class_importance(G, labels, alpha=2):
    num_classes = len(set(labels))
    imp = defaultdict(lambda: [0]*num_classes)
    for node in G.nodes():
        if node.startswith("doc_"):
            idx = int(node.split("_")[1])
            cls = labels[idx]
            for neigh in G.neighbors(node):
                imp[neigh][cls] += 1
    for w, cnts in imp.items(): imp[w] = apply_laplace_smoothing(cnts, alpha, num_classes)
    return imp

# ------------------------
# Classifier & Unlearning
# ------------------------
def train_classifier(imp):
    def classify(doc):
        v = np.zeros(len(next(iter(imp.values()))))
        for w in clean_data(doc):
            if w in imp: v += np.array(imp[w])
        if v.sum()==0: return 0, v.tolist()
        probs = v/np.sum(v)
        return int(np.argmax(probs)), probs.tolist()
    return classify

def zero_class_importance(imp, cls):
    for w, arr in imp.items():
        arr[cls]=0
        s=sum(arr)
        imp[w]=[a/s if s else 0 for a in arr]
    return imp

# ------------------------
# Main
# ------------------------
def main():
    data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
    docs, labels = data.data, data.target
    # TF-IDF + entropy
    tfs, idf = compute_tfidf_scores(docs)
    # Balanced selection
    top_words = select_top_words_by_class(docs, labels, tfs, idf, per_class=3000)
    # Graph + importance
    G = build_graph(docs, labels, top_words)
    imp = create_class_importance(G, labels)
    clf = train_classifier(imp)
    # Evaluate before
    preds = [clf(d)[0] for d in docs]
    print("Accuracy before unlearning:", accuracy_score(labels, preds))  # expect ~0.73
    # Unlearning
    cls = random.randint(0, max(labels))
    imp = zero_class_importance(imp, cls)
    clf = train_classifier(imp)
    preds2 = [clf(d)[0] for d in docs]
    print(f"Accuracy after unlearning class {cls}:", accuracy_score(labels, preds2))

if __name__=='__main__':
    main()
