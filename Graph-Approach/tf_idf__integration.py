import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from math import log
import re
import random
import math

# ------------------------
#  Custom stop words list
# ------------------------
stop_words = set([
    "i","me","my","myself","we","our","ours","ourselves","you","your",
    "yours","yourself","yourselves","he","him","his","himself","she",
    "her","hers","it","its","itself","they","them","their","theirs",
    "themselves","what","which","who","whom","this","that",
    "these","those","am","is","are","was","were","be","been","being",
    "have","has","had","having","do","does","did","doing","a","an",
    "the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into","through",
    "during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then",
    "once","here","there","when","where","why","how","all","any",
    "both","each","few","more","most","other","some","such","no",
    "nor","not","only","own","same","so","than","too","very","s",
    "t","can","will","just","don","should","now","us","much","get","well",
    "would","may","could","however","without","never"
])

# ------------------------
#  Text preprocessing
# ------------------------

def clean_data(text):
    text = re.sub(r"\W+", " ", text.lower())
    return [w for w in text.split() if w.isalpha() and w not in stop_words]


def clean_data_ngrams(text, n=3):
    words = clean_data(text)
    ngrams = ["_".join(words[i:i+n]) for i in range(len(words)-n+1)]
    return words + ngrams

# ------------------------
#  Weighted bipartite graph (TF-IDF)
# ------------------------

def compute_document_frequency(docs):
    df = defaultdict(int)
    for doc in docs:
        for w in set(clean_data(doc)):
            df[w] += 1
    return df


def build_weighted_graph(docs, labels, top_terms):
    df = compute_document_frequency(docs)
    N = len(docs)
    G = nx.Graph()
    for i, doc in enumerate(docs):
        doc_node = f"doc_{i}"
        G.add_node(doc_node, label=labels[i])
        tokens = clean_data_ngrams(doc, n=2)
        tf = Counter(tokens)
        for term in top_terms:
            if term in tf:
                idf = log(N / (1 + df.get(term, 0)))
                weight = tf[term] * idf
                G.add_node(term)
                G.add_edge(doc_node, term, weight=weight)
    return G

# ------------------------
#  Centrality-based features
# ------------------------

def compute_doc_centrality_features(G):
    deg_cent  = nx.degree_centrality(G)
    # try power‐iteration first, then fallback
    try:
        eig_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=500, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        eig_cent = nx.eigenvector_centrality_numpy(G, weight='weight')
    betw_cent = nx.betweenness_centrality(G, weight='weight', k=100, seed=42)


    doc_feats = {}
    for node in G.nodes():
        if node.startswith("doc_"):
            neighs = list(G.neighbors(node))
            if neighs:
                doc_feats[node] = np.array([
                    np.mean([deg_cent[w]   for w in neighs]),
                    np.mean([eig_cent[w]   for w in neighs]),
                    np.mean([betw_cent[w]  for w in neighs])
                ])
            else:
                doc_feats[node] = np.zeros(3)
    return doc_feats

# ------------------------
#  Graph-based unlearning definitions
# ------------------------

def create_word_graph(documents, allowed_words=None):
    g = nx.DiGraph()
    for doc in documents:
        words = words = clean_data_ngrams(doc, n=3)
        counts = Counter(words)
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]

            # SKIP any words outside our top vocabulary
            if allowed_words is not None and (w1 not in allowed_words or w2 not in allowed_words):
                continue

            # (rest is your same logic)
            if counts[w1] > 1:
                if w1 not in g:
                    g.add_node(w1, count=0)
                g.nodes[w1]['count'] += 1
            if w1 not in g:
                g.add_node(w1, count=0)
            if w2 not in g:
                g.add_node(w2, count=0)
            g.add_edge(w1, w2)
    return g


def run_pagerank(word_graph, n):
    try:
        # limit to 100 iterations and accept a looser tolerance
        pr = nx.pagerank(
            word_graph,
            weight='count',
            max_iter=100,
            tol=1e-4
        )
    except nx.PowerIterationFailedConvergence:
        # fallback: solve via the NumPy linear‐algebra backend
        pr = nx.pagerank_numpy(word_graph, weight='count')

    # return top-n words by score
    return sorted(pr, key=pr.get, reverse=True)[:n]


def build_word_document_graph(docs, labels, top_words):
    G = nx.Graph()
    for idx, doc in enumerate(docs):
        node = f"doc_{idx}"
        G.add_node(node, label=labels[idx])
        words = set(clean_data(doc))
        for w in top_words:
            if w in words:
                if w not in G:
                    G.add_node(w)
                G.add_edge(node, w)
    return G

# ------------------------
#  Unlearning utilities
# ------------------------

def apply_laplace_smoothing(counts, alpha, k):
    smoothed = [c+alpha for c in counts]
    total = sum(smoothed)
    return [c/total for c in smoothed]


def create_class_importance_nodes(graph_of_docs, labels, alpha=2):
    num_classes = len(set(labels))
    imp = defaultdict(lambda: [0]*num_classes)
    for node in graph_of_docs.nodes():
        if node.startswith("doc_"):
            idx = int(node.split("_")[1])
            cls = labels[idx]
            for neigh in graph_of_docs.neighbors(node):
                imp[neigh][cls] += 1
    for w, cnts in imp.items():
        imp[w] = apply_laplace_smoothing(cnts, alpha, num_classes)
    return imp


def zero_class_importance(imp, cls):
    for w, arr in imp.items():
        arr[cls] = 0
        s = sum(arr)
        imp[w] = [c/s if s else 0 for c in arr]
    return imp


def remove_high_importance_words(graph_of_docs, imp, cls, threshold):
    to_remove = [w for w, arr in imp.items() if arr[cls]>=threshold]
    graph_of_docs.remove_nodes_from(to_remove)
    print(f"Removed {len(to_remove)} nodes for class {cls}")
    return graph_of_docs

# ------------------------
#  Classifier & MIA attack
# ------------------------

def train_classifier(class_imp, top_words):
    def classify(doc):
        words = clean_data_ngrams(doc, n=3)
        cnts = Counter(words)
        raw = [0]*len(next(iter(class_imp.values())))
        tot = 0
        for w,c in cnts.items():
            if w in top_words and w in class_imp:
                for i, val in enumerate(class_imp[w]):
                    raw[i]+=val*c
                tot+=c
        if tot:
            probs=[r/tot for r in raw]
        else:
            probs=[1/len(raw)]*len(raw)
        return int(np.argmax(probs)), probs
    return classify


def compute_entropy(probs):
    return -sum(p*np.log(p+1e-12) for p in probs)

def tsne_visualization(docs, labels, class_imp, cls):
    emb, cols=[],[]
    for i, doc in enumerate(docs):
        vec=np.zeros(len(next(iter(class_imp.values()))))
        for w in set(clean_data(doc)):
            if w in class_imp:
                vec+=np.array(class_imp[w])
        vec/=max(len(set(clean_data(doc))),1)
        emb.append(vec)
        cols.append('red' if labels[i]==cls else 'blue')
    emb=np.array(emb)
    res=TSNE(n_components=2, random_state=42).fit_transform(emb)
    plt.scatter(res[:,0],res[:,1],c=cols,alpha=0.6)
    plt.title(f"t-SNE class {cls}")
    plt.show()

# ------------------------
#  Main workflow
# ------------------------

def main():
    data=fetch_20newsgroups(subset='all',remove=('headers','footers','quotes'))
    docs, labels=data.data,data.target
    N=len(docs)
    split=int(0.8*N)
    train_docs, test_docs=docs[:split],docs[split:]
    y_train, y_test=labels[:split],labels[split:]

    # Part 1: centrality features
    all_terms=[]
    for d in train_docs: all_terms.extend(clean_data_ngrams(d))
    top_terms=[t for t,_ in Counter(all_terms).most_common(10000)]
    Gw=build_weighted_graph(train_docs,y_train,top_terms)
    feats=compute_doc_centrality_features(Gw)
    X=np.array([feats.get(f"doc_{i}",np.zeros(3)) for i in range(split)])
    X_test=np.array([feats.get(f"doc_{i}",np.zeros(3)) for i in range(split,N)])
    clf=LogisticRegression(max_iter=1000).fit(X,y_train)
    print("Centrality Acc:",accuracy_score(y_test,clf.predict(X_test)))

    cls=random.randint(0,max(labels))

    # original graph-of-words
    wg=create_word_graph(train_docs)
    top_w=run_pagerank(wg,120000)
    #print([w for w in top_w if "_" not in w])

    doc_g=build_word_document_graph(train_docs,y_train,top_w)
    imp=create_class_importance_nodes(doc_g,y_train)
    clf_fn=train_classifier(imp,top_w)
    
    print("Baseline Acc:",accuracy_score(y_test,[clf_fn(d)[0] for d in test_docs]))

    # MIA before
    train_idx=[i for i,l in enumerate(y_train) if l==cls]
    test_idx =[i for i,l in enumerate(y_test) if l==cls]
    n=len(test_idx)
    bal_idx=random.sample(train_idx,n)
    Xr,yr=[],[]
    
    for i in bal_idx:
        _,pr=clf_fn(train_docs[i]);Xr.append([max(pr),sorted(pr,reverse=True)[0]-sorted(pr,reverse=True)[1],compute_entropy(pr)]);yr.append(1)
    
    for i in test_idx:
        _,pr=clf_fn(test_docs[i]);Xr.append([max(pr),sorted(pr,reverse=True)[0]-sorted(pr,reverse=True)[1],compute_entropy(pr)]);yr.append(0)
    
    Xr,yr=np.array(Xr),np.array(yr)
    Xa,Xt,ya,yt=train_test_split(Xr,yr,test_size=0.3,random_state=42)
    atk=LogisticRegression(class_weight='balanced').fit(Xa,ya)
    print("MIA AUC before:",roc_auc_score(yt,atk.predict_proba(Xt)[:,1]))

    tsne_visualization(test_docs, y_test, imp, cls)

    # UNLEARNING
    doc_g=remove_high_importance_words(doc_g,imp,cls,threshold=0.05)
    imp=zero_class_importance(imp,cls)
    clf_fn=train_classifier(imp,top_w)

    # Accuracy after unlearning — with and without class cls
    preds_after = [clf_fn(d)[0] for d in test_docs]
    acc_with_cls = accuracy_score(y_test, preds_after)
    filtered_test = [(d, l) for d, l in zip(test_docs, y_test) if l != cls]
    if filtered_test:
        preds_filtered = [clf_fn(d)[0] for d, _ in filtered_test]
        labels_filtered = [l for _, l in filtered_test]
        acc_without_cls = accuracy_score(labels_filtered, preds_filtered)
    else:
        acc_without_cls = "N/A (class not present)"

    print(f"Post-Unlearning Acc (with class {cls}):", acc_with_cls)
    print(f"Post-Unlearning Acc (without class {cls}):", acc_without_cls)
    
    # MIA after
    Xr2,yr2=[],[]
    for i in bal_idx:
        _,pr=clf_fn(train_docs[i]);Xr2.append([max(pr),sorted(pr,reverse=True)[0]-sorted(pr,reverse=True)[1],compute_entropy(pr)]);yr2.append(1)
    
    for i in test_idx:
        _,pr=clf_fn(test_docs[i]);Xr2.append([max(pr),sorted(pr,reverse=True)[0]-sorted(pr,reverse=True)[1],compute_entropy(pr)]);yr2.append(0)
    
    Xa2,Xt2,ya2,yt2=train_test_split(np.array(Xr2),np.array(yr2),test_size=0.3,random_state=42)
    atk2=LogisticRegression(class_weight='balanced').fit(Xa2,ya2)
    print("MIA AUC after:",roc_auc_score(yt2,atk2.predict_proba(Xt2)[:,1]))

    tsne_visualization(test_docs,y_test,imp,cls)

if __name__=='__main__':
    main()