import networkx as nx
import numpy as np
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re
import random
from scipy.spatial.distance import euclidean

# Define custom stop words
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
        
        # Count word occurrences within the document
        for word in words:
            word_counts[word] += 1
        
        # Update nodes and connections
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            
            # Only increment count if word appears multiple times
            if word_counts[word1] > 1:
                if word1 not in word_graph:
                    word_graph.add_node(word1, count=0, neighbors=defaultdict(int))
                word_graph.nodes[word1]['count'] += 1
            
            # Connect consecutive words
            if word1 not in word_graph:
                word_graph.add_node(word1, count=0, neighbors=defaultdict(int))
            if word2 not in word_graph:
                word_graph.add_node(word2, count=0, neighbors=defaultdict(int))
            
            word_graph.nodes[word1]['neighbors'][word2] += 1
            word_graph.add_edge(word1, word2)
    return word_graph

def run_pagerank(word_graph, n):
    """Apply PageRank to the word graph and return the top `n` words."""
    page_rank_scores = nx.pagerank(word_graph, weight='count')
    top_words = sorted(page_rank_scores, key=page_rank_scores.get, reverse=True)[:n]
    #print(top_words)
    return top_words 

def build_word_document_graph(documents, labels, top_words):
    """Build a graph_of_docs where documents are connected to the top words they contain."""
    graph_of_docs = nx.Graph()
    for idx, doc in enumerate(documents):
        words = set(clean_data(doc))
        doc_node = f"doc_{idx}"
        graph_of_docs.add_node(doc_node, label=labels[idx])
        
        # Connect the document to relevant top word nodes
        for word in top_words:
            if word in words:
                if word not in graph_of_docs:
                    graph_of_docs.add_node(word)
                graph_of_docs.add_edge(doc_node, word)
    
    return graph_of_docs


def create_class_importance_nodes(graph_of_docs, labels):
    """Calculate class importance scores for each word node, with an array for each class."""
    class_importance = defaultdict(lambda: [0] * 20)  # Assuming there are 20 classes

    for node in graph_of_docs:
        # Process only document nodes
        if 'doc_' in node:
            doc_index = int(node.split('_')[1])
            label = labels[doc_index]  # Get the class label for the document
            
            # Update class importance for each word connected to this document
            for neighbor in graph_of_docs.neighbors(node):
                class_importance[neighbor][label] += 1
    
    for word, importance_array in class_importance.items():
        total = sum(importance_array)
        if total > 0:
            class_importance[word] = [count / total for count in importance_array]
    return class_importance

def zero_class_importance(class_importances, class_to_zero):
    """Set the class importance scores for the given class to zero and re-average the scores."""
    for word, importance_array in class_importances.items():
        # Set the class importance for the specified class to zero
        importance_array[class_to_zero] = 0
        
        # Calculate the new sum of the importance scores for the remaining classes
        total = sum(importance_array)
        if total > 0:
            # Normalize the remaining scores
            class_importances[word] = [count / total for count in importance_array]
        else:
            # If no importance remains, reset all to zero
            class_importances[word] = [0] * len(importance_array)

    return class_importances


def train_classifier(class_importance, top_words):
    """Create a custom classifier using the normalized class importance scores."""

    def classify_document(document):
        """Classify a document by averaging class importance scores of words it contains."""
        words = set(clean_data(document))
        class_scores = [0] * 20  # Initialize scores for each class
        relevant_words = 0

        for word in words:
            if word in top_words and word in class_importance:
                for cls in range(20):
                    class_scores[cls] += class_importance[word][cls]
                relevant_words += 1

        if relevant_words > 0:
            class_scores = [score / relevant_words for score in class_scores]
        
        predicted_class = class_scores.index(max(class_scores))
        return predicted_class

    return classify_document

def compute_centroids(class_importance, documents, labels, top_words):
    """Compute centroids for each class and the entire dataset."""
    class_vectors = defaultdict(list)  # Store vectors for each class
    dataset_vectors = []  # Store all document vectors for entire dataset

    for idx, doc in enumerate(documents):
        words = set(clean_data(doc))
        doc_vector = np.zeros(20)  # Feature vector of 20 classes

        for word in words:
            if word in top_words and word in class_importance:
                doc_vector += np.array(class_importance[word])

        if np.sum(doc_vector) > 0:
            doc_vector /= np.sum(doc_vector)  # Normalize the vector

        class_vectors[labels[idx]].append(doc_vector)
        dataset_vectors.append(doc_vector)

    # Compute centroids
    class_centroids = {cls: np.mean(vectors, axis=0) for cls, vectors in class_vectors.items()}
    dataset_centroid = np.mean(dataset_vectors, axis=0)

    # Compute average distances from centroids
    class_avg_distances = {}
    for cls, vectors in class_vectors.items():
        distances = np.linalg.norm(np.array(vectors) - class_centroids[cls], axis=1)
        class_avg_distances[cls] = np.mean(distances)

    dataset_distances = np.linalg.norm(np.array(dataset_vectors) - dataset_centroid, axis=1)
    dataset_avg_distance = np.mean(dataset_distances)

    return {
        "class_centroids": class_centroids,
        "dataset_centroid": dataset_centroid,
        "class_avg_distances": class_avg_distances,
        "dataset_avg_distance": dataset_avg_distance
    }

def tsne_visualization(documents, labels, class_importance, class_to_zero):
    """Visualize document embeddings using t-SNE."""
    embeddings = []
    color_map = []  # To distinguish unlearned and other classes
    for idx, doc in enumerate(documents):
        words = set(clean_data(doc))
        doc_vector = np.zeros(20)  # 20 classes

        for word in words:
            if word in class_importance:
                doc_vector += np.array(class_importance[word])
        
        doc_vector /= len(words) if len(words) > 0 else 1  # Normalize by the number of words
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

def main():
    # Load dataset
    dataset = load_20newsgroup_dataset()
    documents = dataset.data
    labels = dataset.target

    split_index = int(0.8 * len(documents))
    train_dataset = documents[:split_index]
    test_dataset = documents[split_index:]
    y_train_labels = labels[:split_index]
    y_test_labels = labels[split_index:]

    # Create word graph
    word_graph = create_word_graph(train_dataset)

    # Run PageRank to get top words
    top_words = run_pagerank(word_graph, n=80000)

    # Build word-document graph
    graph_of_docs = build_word_document_graph(train_dataset, y_train_labels, top_words)

    # Create class importance nodes
    class_importance = create_class_importance_nodes(graph_of_docs, labels)


    classify_document = train_classifier(class_importance, top_words)

    # Predict and evaluate on the test set
    y_pred = [classify_document(doc) for doc in test_dataset]
    accuracy_all = accuracy_score(y_test_labels, y_pred)
    print(f"Accuracy on entire test data before unlearning: {accuracy_all * 100:.2f}%")

    centroids_before = compute_centroids(class_importance, test_dataset, y_test_labels, top_words)
    #print("Class Centroids Before Unlearning:", centroids_before["class_centroids"])
    #print("Dataset Centroid Before Unlearning:", centroids_before["dataset_centroid"])

    #Unlearning process
    class_to_zero = random.randint(0, 19)
    
    #Visualization before unlearning
    tsne_visualization(test_dataset, y_test_labels, class_importance, class_to_zero)

    print(f"Unlearned Class: {class_to_zero}")

    # Adjust class importance scores
    class_importance = zero_class_importance(class_importance, class_to_zero)

    # Train custom classifier with adjusted class importance scores
    classify_document = train_classifier(class_importance, top_words)

    # Predict and evaluate on the test set
    y_pred = [classify_document(doc) for doc in test_dataset]
    accuracy_all = accuracy_score(y_test_labels, y_pred)
    print(f"Accuracy on test data after unlearning {class_to_zero}: {accuracy_all * 100:.2f}%")

    # Accuracy excluding the unlearned class
    valid_indices = [i for i, label in enumerate(y_test_labels) if label != class_to_zero]
    y_test_filtered = [y_test_labels[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]
    accuracy_excl = accuracy_score(y_test_filtered, y_pred_filtered)

    print(f"Accuracy excluding class {class_to_zero}: {accuracy_excl * 100:.2f}%")

    centroids_after = compute_centroids(class_importance, test_dataset, y_test_labels, top_words)
    #print("Class Centroids After Unlearning:", centroids_after["class_centroids"])
    #print("Dataset Centroid After Unlearning:", centroids_after["dataset_centroid"])

    distance_before = centroids_before["class_avg_distances"].get(class_to_zero, None)
    print(f"Average Distance of Unlearned Class {class_to_zero} Before Unlearning: {distance_before:.4f}")
    
    distance_after = centroids_after["class_avg_distances"].get(class_to_zero, None)
    print(f"Average Distance of Unlearned Class {class_to_zero} After Unlearning: {distance_after:.4f}")

    #Visualization after unlearning 
    tsne_visualization(test_dataset, y_test_labels, class_importance, class_to_zero)


# Run the main function
if __name__ == "__main__":
    main()