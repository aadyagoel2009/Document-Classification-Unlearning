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
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, roc_auc_score

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

def load_20newsgroup_dataset():
    """Load and return the 20 Newsgroups dataset."""
    return fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# ------------------------
# Text cleaning
# ------------------------
def char_entropy(word):
    counts = Counter(word)
    total = len(word)
    return -sum((c/total)*math.log((c/total)+1e-12) for c in counts.values())


def clean_data(text):
    text = re.sub(r"\W+", " ", text.lower())
    words = [w for w in text.split() if w.isalpha() and w not in stop_words and len(w)>=3]
    # filter gibberish
    return [w for w in words if 1.0 <= char_entropy(w) <= 4.0]

# ------------------------
# TF-IDF + Entropy filtering for word selection
# ------------------------
def compute_tfidf_scores(docs):
    df = defaultdict(int)
    tfs = []
    for doc in docs:
        words = clean_data(doc)
        tf = Counter(words)
        tfs.append(tf)
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
    return sorted(avg_tfidf, key=avg_tfidf.get, reverse=True)[:60000]

# ------------------------
# Build word-document graph
# ------------------------
def build_word_document_graph(docs, labels, top_words):
    G = nx.Graph()
    for i, doc in enumerate(docs):
        node = f"doc_{i}"
        G.add_node(node, label=labels[i])
        for w in set(clean_data(doc)):
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

def create_class_importance_nodes(G, labels, alpha=0):
    num_classes = len(set(labels))
    imp = defaultdict(lambda: [0]*num_classes)
    for node in G.nodes():
        if node.startswith("doc_"):
            idx = int(node.split('_')[1])
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
        vec = np.zeros(len(next(iter(imp.values()))))
        for w in clean_data(doc):
            if w in imp: vec += np.array(imp[w])
        if vec.sum()==0: return 0, vec.tolist()
        probs = vec/vec.sum()
        return int(np.argmax(probs)), probs.tolist()
    return classify

def zero_class_importance(importances, cls):
    for w, arr in importances.items():
        arr[cls] = 0
        s = sum(arr)
        importances[w] = [c/s if s else 0 for c in arr]
    return importances

def tsne_visualization(docs, labels, class_importances, class_to_unlearn):
    emb, cols = [], []
    for i, doc in enumerate(docs):
        vec = np.zeros(len(next(iter(class_importances.values()))))
        for w in set(clean_data(doc)):
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
def main():
    dataset = load_20newsgroup_dataset()
    documents = dataset.data
    labels = dataset.target

    split_index = int(0.8 * len(documents))
    train_docs = documents[:split_index]
    test_docs = documents[split_index:]
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]
    print("data partitioned")

    # word selection
    top_words = compute_tfidf_scores(train_docs)
    print("words chosen")

    # graph & importance
    graph_of_doc = build_word_document_graph(train_docs, train_labels, top_words)
    print("graph built")
    class_importances = create_class_importance_nodes(graph_of_doc, train_labels)
    print("importances created")
    classify_document = train_classifier(class_importances)
    print("classifier trained")

    # evaluate before unlearning
    test_pred = [classify_document(d)[0] for d in test_docs]
    print("Accuracy before unlearning:", accuracy_score(test_labels, test_pred))

    # unlearning
    class_to_unlearn = random.randint(0, max(train_labels))
    tsne_visualization(test_docs, test_labels, class_importances, class_to_unlearn)
    print("MIA AUC before:", membership_inference_attack(classify_document, 
        train_docs, test_docs, train_labels, test_labels, class_to_unlearn))


    # zero class importance
    zero_class_importance(class_importances, class_to_unlearn)
    classify_document = train_classifier(class_importances)
    print("retrain classifier")

    test_pred = [classify_document(d)[0] for d in test_docs]
    print(f"Accuracy after unlearning class {class_to_unlearn}:", accuracy_score(test_labels, test_pred))
    docs_no, labels_no = zip(*[(d, l) for d, l in zip(test_docs, test_labels) if l != class_to_unlearn])
    print(f"Post-Unlearning Acc (without class {class_to_unlearn}):",
      accuracy_score(labels_no, [classify_document(d)[0] for d in docs_no]))
    
    tsne_visualization(test_docs, test_labels, class_importances, class_to_unlearn)
    print("MIA AUC after:", membership_inference_attack(classify_document, 
        train_docs, test_docs, train_labels, test_labels, class_to_unlearn))


if __name__ == '__main__':
    main()
