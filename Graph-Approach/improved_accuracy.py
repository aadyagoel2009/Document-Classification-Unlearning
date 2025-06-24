import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re
import random
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from scipy.stats import entropy

# Define custom stop words
stop_words = set([
    # … your full stop_words list …
])

def load_20newsgroup_dataset():
    """Load and return the 20 Newsgroups dataset."""
    return fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

def clean_data(text):
    """Clean and tokenize text data, removing stop words."""
    text = re.sub(r'\W+', ' ', text.lower())
    words = [word for word in text.split() if word not in stop_words and word.isalpha()]
    return words

def create_word_graph(documents):
    """Build a word graph where nodes are words and edges connect consecutive words."""
    word_graph = nx.DiGraph()
    for doc in documents:
        words = clean_data(doc)
        word_counts = defaultdict(int)
        for w in words:
            word_counts[w] += 1
        for w1, w2 in zip(words, words[1:]):
            if word_counts[w1] > 1:
                if w1 not in word_graph:
                    word_graph.add_node(w1, count=0, neighbors=defaultdict(int))
                word_graph.nodes[w1]['count'] += 1
            if w1 not in word_graph:
                word_graph.add_node(w1, count=0, neighbors=defaultdict(int))
            if w2 not in word_graph:
                word_graph.add_node(w2, count=0, neighbors=defaultdict(int))
            word_graph.nodes[w1]['neighbors'][w2] += 1
            word_graph.add_edge(w1, w2)
    return word_graph

def run_pagerank(word_graph, n):
    """Apply PageRank to the word graph and return the top `n` words."""
    page_rank_scores = nx.pagerank(word_graph, weight='count')
    top_words = sorted(page_rank_scores, key=page_rank_scores.get, reverse=True)[:n]
    return top_words 

def build_word_document_graph(documents, labels, top_words):
    """Build a graph_of_docs where documents are connected to the top words they contain."""
    graph_of_docs = nx.Graph()
    for idx, doc in enumerate(documents):
        words = set(clean_data(doc))
        doc_node = f"doc_{idx}"
        graph_of_docs.add_node(doc_node, label=labels[idx])
        for word in top_words:
            if word in words:
                if word not in graph_of_docs:
                    graph_of_docs.add_node(word)
                graph_of_docs.add_edge(doc_node, word)
    return graph_of_docs

def apply_laplace_smoothing(class_counts, alpha, num_classe):
    """Apply Laplace smoothing to an array of class counts."""
    smoothed_counts = [count + alpha for count in class_counts]
    total = sum(smoothed_counts)
    return [count / total for count in smoothed_counts]

def create_class_importance_nodes(graph_of_docs, labels, alpha=2):
    """Calculate class importance scores for each word node using Laplace smoothing."""
    num_classes = len(set(labels))
    class_importance = defaultdict(lambda: [0] * num_classes)
    for node in graph_of_docs:
        if node.startswith('doc_'):
            doc_index = int(node.split('_')[1])
            label = labels[doc_index]
            for neighbor in graph_of_docs.neighbors(node):
                class_importance[neighbor][label] += 1
    for word, counts in class_importance.items():
        class_importance[word] = apply_laplace_smoothing(counts, alpha, num_classes)
    return class_importance

def zero_class_importance(class_importances, class_to_zero):
    """Set the class importance scores for the given class to zero and re-average the scores."""
    for word, importance_array in class_importances.items():
        importance_array[class_to_zero] = 0
        total = sum(importance_array)
        if total > 0:
            class_importances[word] = [count / total for count in importance_array]
        else:
            class_importances[word] = [0] * len(importance_array)
    return class_importances

def train_classifier(class_importance, top_words):
    """Create a custom classifier using the normalized class importance scores."""
    def classify_document(document):
        words = clean_data(document)
        word_counts = Counter(words)
        class_scores = [0] * 20
        total_weight = 0
        for word, count in word_counts.items():
            if word in top_words and word in class_importance:
                for cls in range(20):
                    class_scores[cls] += class_importance[word][cls] * count
                total_weight += count
        if total_weight > 0:
            class_scores = [score / total_weight for score in class_scores]
        return class_scores.index(max(class_scores))
    return classify_document

def remove_high_importance_words(graph_of_docs, class_importance, unlearned_class, threshold):
    nodes_to_remove = []
    for word, importance_array in class_importance.items():
        if importance_array[unlearned_class] >= threshold:
            nodes_to_remove.append(word)
    graph_of_docs.remove_nodes_from(nodes_to_remove)
    print(f"Removed {len(nodes_to_remove)} nodes with high importance for class {unlearned_class}.")
    return graph_of_docs, nodes_to_remove

def tsne_visualization(documents, labels, class_importance, class_to_zero):
    embeddings, color_map = [], []
    for idx, doc in enumerate(documents):
        words = set(clean_data(doc))
        vec = np.zeros(20)
        for w in words:
            if w in class_importance:
                vec += np.array(class_importance[w])
        vec /= len(words) if words else 1
        embeddings.append(vec)
        color_map.append('red' if labels[idx]==class_to_zero else 'blue')
    emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(np.array(embeddings))
    plt.figure(figsize=(10,7))
    plt.scatter(emb[:,0], emb[:,1], c=color_map, alpha=0.6)
    plt.title(f"t-SNE Visualization: Class {class_to_zero} in Red"); plt.show()

def main():
    dataset = load_20newsgroup_dataset()
    documents, labels = dataset.data, dataset.target
    split = int(0.8*len(documents))
    train_dataset = documents[:split]
    test_dataset  = documents[split:]
    y_train = labels[:split]
    y_test  = labels[split:]

    # — new TF-IDF setup —
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words)
    tfidf_train = vectorizer.fit_transform(train_dataset)
    tfidf_test  = vectorizer.transform(test_dataset)

    class_to_zero = random.randint(0,19)
    print(f"Unlearned Class: {class_to_zero}")

    # existing pipeline
    word_graph = create_word_graph(train_dataset)
    top_words  = run_pagerank(word_graph, n=80000)
    graph_of_docs = build_word_document_graph(train_dataset, y_train, top_words)
    class_importance = create_class_importance_nodes(graph_of_docs, labels)

    classify_document = train_classifier(class_importance, top_words)
    y_pred = [classify_document(doc) for doc in test_dataset]
    accuracy_all = accuracy_score(y_test, y_pred)
    print(f"Accuracy on entire test data before unlearning: {accuracy_all*100:.2f}%")

    tsne_visualization(test_dataset, y_test, class_importance, class_to_zero)

    graph_of_docs, removed_words = remove_high_importance_words(
        graph_of_docs, class_importance, class_to_zero, threshold=0.05
    )
    class_importance = zero_class_importance(class_importance, class_to_zero)
    classify_document = train_classifier(class_importance, top_words)

    y_pred = [classify_document(doc) for doc in test_dataset]
    accuracy_all = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test data after unlearning {class_to_zero}: {accuracy_all*100:.2f}%")

    valid_idx = [i for i,l in enumerate(y_test) if l!=class_to_zero]
    accuracy_excl = accuracy_score(
        np.array(y_test)[valid_idx],
        np.array(y_pred)[valid_idx]
    )
    print(f"Accuracy excluding class {class_to_zero}: {accuracy_excl*100:.2f}%")

    tsne_visualization(test_dataset, y_test, class_importance, class_to_zero)



if __name__ == "__main__":
    main()