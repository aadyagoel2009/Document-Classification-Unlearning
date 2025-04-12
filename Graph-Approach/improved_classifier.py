import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ks_2samp

# For the improved classifier (a shallow MLP)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --------------------------
# Graph-Based and Unlearning Functions (Stage 1)
# --------------------------

stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "it", "its", "itself", "they", "them", "their", "theirs", 
    "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", 
    "once", "here", "there", "when", "where", "why", "how", "all", "any", 
    "both", "each", "few", "more", "most", "other", "some", "such", "no", 
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
    "t", "can", "will", "just", "don", "should", "now", "us", "much", "get", "well",
    "would", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "10", "a", "b",
    "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
    "r", "s", "t", "u", "v", "w", "x", "y", "z", "may", "didn", "ll", "ls", "got", "ah",
    "ve", "ca", "db", "also", "like", "ax", "even", "could", "re", "however",
    "without", "doesn", "going", "never", "mr", "de", "bit", "put", "let"
])

def load_20newsgroup_dataset():
    return fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

def clean_data(text):
    text = re.sub(r'\W+', ' ', text.lower())
    words = [word for word in text.split() if word not in stop_words and word.isalpha()]
    return words

def create_word_graph(documents):
    word_graph = nx.DiGraph()
    for doc in documents:
        words = clean_data(doc)
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            if word_counts[word1] > 1:
                if word1 not in word_graph:
                    word_graph.add_node(word1, count=0, neighbors=defaultdict(int))
                word_graph.nodes[word1]['count'] += 1
            if word1 not in word_graph:
                word_graph.add_node(word1, count=0, neighbors=defaultdict(int))
            if word2 not in word_graph:
                word_graph.add_node(word2, count=0, neighbors=defaultdict(int))
            word_graph.nodes[word1]['neighbors'][word2] += 1
            word_graph.add_edge(word1, word2)
    return word_graph

def run_pagerank(word_graph, n):
    page_rank_scores = nx.pagerank(word_graph, weight='count')
    top_words = sorted(page_rank_scores, key=page_rank_scores.get, reverse=True)[:n]
    return top_words 

def build_word_document_graph(documents, labels, top_words):
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

def create_class_importance_nodes(graph_of_docs, labels, alpha=2):
    num_classes = len(set(labels))
    class_importance = defaultdict(lambda: [0] * num_classes)
    for node in graph_of_docs:
        if 'doc_' in node:
            doc_index = int(node.split('_')[1])
            label = labels[doc_index]
            for neighbor in graph_of_docs.neighbors(node):
                class_importance[neighbor][label] += 1
    for word, counts in class_importance.items():
        class_importance[word] = apply_laplace_smoothing(counts, alpha, num_classes)
    return class_importance

def apply_laplace_smoothing(class_counts, alpha, num_classes):
    smoothed_counts = [count + alpha for count in class_counts]
    total = sum(smoothed_counts)
    return [count / total for count in smoothed_counts]

def zero_class_importance(class_importances, class_to_zero):
    for word, importance_array in class_importances.items():
        importance_array[class_to_zero] = 0
        total = sum(importance_array)
        if total > 0:
            class_importances[word] = [count / total for count in importance_array]
        else:
            class_importances[word] = [0] * len(importance_array)
    return class_importances

def train_classifier(class_importance, top_words):
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
        predicted_class = class_scores.index(max(class_scores))
        return predicted_class
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
    embeddings = []
    color_map = []
    for idx, doc in enumerate(documents):
        words = set(clean_data(doc))
        doc_vector = np.zeros(20)
        for word in words:
            if word in class_importance:
                doc_vector += np.array(class_importance[word])
        doc_vector /= len(words) if len(words) > 0 else 1
        embeddings.append(doc_vector)
        color_map.append('red' if labels[idx] == class_to_zero else 'blue')
    embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=color_map, alpha=0.6)
    plt.title(f"t-SNE Visualization: Class {class_to_zero} in Red")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

def compute_doc_representation(documents, top_words, class_importance):
    representations = []
    for doc in documents:
        words = set(clean_data(doc))
        doc_vector = np.zeros(20)
        for word in words:
            if word in top_words and word in class_importance:
                doc_vector += np.array(class_importance[word])
        if np.sum(doc_vector) > 0:
            doc_vector = doc_vector / np.sum(doc_vector)
        representations.append(doc_vector)
    return np.array(representations)

# --------------------------
# Improved Classifier (Stage 2)
# --------------------------
# This is a shallow MLP that learns a non-linear combination of the 20D keyword importance features.

class ImprovedClassifier(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=20):
        super(ImprovedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        return probs

def improved_loss(outputs, true_labels, unlearned_class, lambda_align=0.01):
    ce_loss = F.cross_entropy(outputs, true_labels)
    unlearned_probs = outputs[:, unlearned_class]
    target = torch.mean(unlearned_probs)
    alignment_loss = torch.mean((unlearned_probs - target) ** 2)
    return ce_loss + lambda_align * alignment_loss

# --------------------------
# Main Function: Two-Stage Training with Improved Classifier
# --------------------------

def main():
    # Stage 1: Graph-Based Unlearning & Classification
    dataset = load_20newsgroup_dataset()
    documents = dataset.data
    labels = dataset.target

    split_index = int(0.8 * len(documents))
    train_dataset = documents[:split_index]
    test_dataset = documents[split_index:]
    y_train_labels = labels[:split_index]
    y_test_labels = labels[split_index:]
    
    # Choose a random class to unlearn
    class_to_zero = random.randint(0, 19)
    print(f"Unlearned Class: {class_to_zero}")

    # (Optional) Fit TF-IDF for efficiency (not used further here)
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(train_dataset)

    # Build word graph and compute keyword importance
    word_graph = create_word_graph(train_dataset)
    top_words = run_pagerank(word_graph, n=80000)

    # Build word-document graph and compute class importance vectors
    graph_of_docs = build_word_document_graph(train_dataset, y_train_labels, top_words)
    class_importance = create_class_importance_nodes(graph_of_docs, labels)
    
    # Evaluate the initial graph-based classifier
    classify_document = train_classifier(class_importance, top_words)
    y_pred = [classify_document(doc) for doc in test_dataset]
    base_accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"Base Graph-Based Classifier Accuracy (before unlearning): {base_accuracy * 100:.2f}%")
    
    # Unlearning: Remove high-importance words and zero out the unlearned class
    graph_of_docs, removed_words = remove_high_importance_words(graph_of_docs, class_importance, class_to_zero, threshold=0.95)
    class_importance = zero_class_importance(class_importance, class_to_zero)

    # Re-evaluate graph-based classifier after unlearning
    classify_document = train_classifier(class_importance, top_words)
    y_pred = [classify_document(doc) for doc in test_dataset]
    post_unlearn_accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"Graph-Based Classifier Accuracy after Unlearning Class {class_to_zero}: {post_unlearn_accuracy * 100:.2f}%")

    # Stage 2: Train the Improved Classifier on 20D Representations
    train_reps = compute_doc_representation(train_dataset, top_words, class_importance)
    test_reps = compute_doc_representation(test_dataset, top_words, class_importance)

    # Convert representations and labels to PyTorch tensors
    X_train = torch.tensor(train_reps, dtype=torch.float32)
    X_test = torch.tensor(test_reps, dtype=torch.float32)
    y_train = torch.tensor(y_train_labels, dtype=torch.long)
    y_test = torch.tensor(y_test_labels, dtype=torch.long)

    # Define the improved classifier and optimizer
    model = ImprovedClassifier(input_dim=20, hidden_dim=64, num_classes=20)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    lambda_align = 0.01  # small weight to preserve accuracy

    # Train the improved classifier
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = improved_loss(outputs, y_train, unlearned_class=class_to_zero, lambda_align=lambda_align)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Evaluate the improved classifier on the test set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted_labels = torch.argmax(outputs, dim=1)
        improved_accuracy = (predicted_labels == y_test).float().mean().item()
    print(f"\nImproved Classifier Accuracy on test data: {improved_accuracy * 100:.2f}%")

    # Perform KS test on the improved classifier's outputs
    unlearned_probs = outputs[:, class_to_zero].cpu().numpy()
    overall_probs = outputs.cpu().numpy().flatten()
    stat, p_value = ks_2samp(unlearned_probs, overall_probs)
    print(f"\nKS test after improved classifier: Statistic: {stat:.4f}, p-value: {p_value:.4f}")

    # Print summary statistics
    print("\nAfter improved classifier, unlearned class probability statistics:")
    print(f"Mean: {np.mean(unlearned_probs):.4f}, Std: {np.std(unlearned_probs):.4f}")
    print("Overall probability statistics:")
    print(f"Mean: {np.mean(overall_probs):.4f}, Std: {np.std(overall_probs):.4f}")

if __name__ == "__main__":
    main()