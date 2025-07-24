
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

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
from sklearn.datasets import fetch_20newsgroups
import os, json
from functools import lru_cache
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import StratifiedShuffleSplit


# ------------------------
# Word Selection
# ------------------------

#Hyperparameters 
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KEYWORDS_PER_CLASS = 4000                     # words returned per class
ENT_MIN, ENT_MAX   = 1.0, 4.0                # entropy gate 

kw_model = KeyBERT(EMB_MODEL)

def char_entropy(word):
    counts = Counter(word)
    total = len(word)
    return -sum((c/total)*math.log((c/total)+1e-12) for c in counts.values())

def clean_data(text):
    text = re.sub(r"\W+", " ", text.lower())
    words = [w for w in text.split() if w.isalpha() and len(w)>=2]
    return [w for w in words if 0.5 <= char_entropy(w) <= 5.0]

def build_keywords_per_class(train_docs, train_labels):
    """Return the union of KEYWORDS_PER_CLASS discriminative tokens per class."""
    class_names  = fetch_20newsgroups(subset="all").target_names
    keywords_by_class = {}

    for cls, name in enumerate(class_names):
        # 1 ) gather training docs for this class
        freq = Counter()
        class_ids = [d for d, l in enumerate(train_labels) if l == cls]
        class_docs = [train_docs[i] for i in class_ids]

        for doc in class_docs:
            # 2 ) KeyBERT: return (keyword, score) tuples
            kw_scores = kw_model.extract_keywords(
                doc,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=100, 
                use_mmr=True,   
                diversity = 0.7                     
            )
            freq.update(w.lower() for w, _ in kw_scores if ENT_MIN <= char_entropy(w) <= ENT_MAX and w.isascii())
            print("doc")
        
        # ensure exactly KEYWORDS_PER_CLASS per class
        keywords_by_class[cls] = set([w for w, _ in freq.most_common(KEYWORDS_PER_CLASS)])
        print(cls)

    top_words = sorted(set().union(*keywords_by_class.values()))
    return top_words

def build_doc_kws(doc, k=20):
    """
    Return the top-k KeyBERT keywords for a single document.
    """
    kws = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=k,
        use_mmr=True,
        diversity=0.7
    )
    # just the words, filtered by your entropy gate
    return [
        w.lower() for w,_ in kws
        if ENT_MIN <= char_entropy(w) <= ENT_MAX and w.isascii()
    ]

# ------------------------
# Build word-document graph
# ------------------------
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
def train_classifier(imp, word_embeddings, embed_model, idf):
    def clean_data(text):
        text = re.sub(r"\W+", " ", text.lower())
        words = [w for w in text.split() if w.isalpha() and len(w)>=3]
        return [w for w in words if 1.0 <= char_entropy(w) <= 4.0]
    
    def softmax(logits):
        shifted = logits - np.max(logits)
        exp_shifted = np.exp(shifted)
        return exp_shifted / exp_shifted.sum()
    
    def classify(doc):
        raw = embed_model.encode(
            doc, convert_to_numpy=True, normalize_embeddings=True
        )
        norm = np.linalg.norm(raw)
        doc_emb = raw/norm if norm>0 else raw

        tf = Counter(clean_data(doc))
        vec = np.zeros(len(next(iter(imp.values()))))

        for w, count in tf.items():
            if w in imp:
                sim = float(np.dot(doc_emb, word_embeddings[w]))
                sim = max(sim, 0.0)
            
                tf_log    = 1 + math.log(count)
                weight = sim * tf_log * idf[w]
                vec += weight * np.array(imp[w])

        for w in build_doc_kws(doc, k=5):
            if w in imp:
                vec += 1.0 * np.array(imp[w])

        # fallback if no words matched
        if vec.sum() == 0:
            return 0, vec.tolist()

        probs = softmax(vec)
        u = random.random()        
        cumulative_probs = np.cumsum(probs)        
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
    proj = TSNE(n_components=2, init="random", random_state=42, n_jobs=1).fit_transform(emb)
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

def load_20newsgroup_dataset():
    """Load and return the 20 Newsgroups dataset."""
    return fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# ------------------------
# Main workflow with 20newsgroup
# ------------------------
def main():
    dataset = load_20newsgroup_dataset()
    documents = dataset.data
    labels = dataset.target

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    print("startified")
    train_idx, test_idx = next(sss.split(documents, labels))


    train_docs   = [documents[i] for i in train_idx]
    test_docs    = [documents[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels  = labels[test_idx]
    "split"

    # word selection
    top_words = build_keywords_per_class(train_docs, train_labels)
    print(f"{len(top_words)} LLM keywords gathered.")

    cleaned_docs = []
    for doc in train_docs:
        words = clean_data(doc)
        cleaned_docs.append(words)

    idf = {
        w: math.log(len(cleaned_docs) / (1 + sum(w in doc for doc in cleaned_docs)))
        for w in top_words
    }

    # graph & importance
    graph_of_doc = build_word_document_graph(cleaned_docs, train_labels, top_words)
    class_importances = create_class_importance_nodes(graph_of_doc, train_labels)

    embed_model = SentenceTransformer(EMB_MODEL)
    embed_model.max_seq_length = 512
    word_embeddings = {}
    for w in top_words:
        emb = embed_model.encode(w, convert_to_numpy=True)    # -> np.ndarray
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        word_embeddings[w] = emb


    classify_document = train_classifier(class_importances, word_embeddings, embed_model, idf)

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
    classify_document = train_classifier(class_importances, word_embeddings, embed_model, idf)
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
