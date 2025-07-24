import os
import time
import re
import math
import random
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.manifold import TSNE

from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# TogetherAI client setup
# ──────────────────────────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),         
    base_url="https://api.deepseek.com/v1",      
)

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
ENT_MIN, ENT_MAX = 1.0, 4.0    # for clean_data()
DOC_KWS_K = 10                 # top-k per document
CLASS_KWS_DOC_K = 10           # boost from doc KWs in classifier

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
    return [w for w in words if ENT_MIN <= char_entropy(w) <= ENT_MAX]

# ──────────────────────────────────────────────────────────────────────────────
# Calls to DeepSeek LLM 
# ──────────────────────────────────────────────────────────────────────────────
def build_doc_kws_with_scores(doc: str, k: int = DOC_KWS_K) -> list[tuple[str, float]]:
    instruction = (
        f"Identify the {k} most distinctive, generalizable keywords or short phrases that capture the core themes of this document "
        "in a way that would help categorize *other* similar texts. "
        "Focus on topic or concept terms. Exclude overly document-specific proper names."
        "Output exactly one `term:score` per line, where `score` is a decimal between 0.0 and 1.0. No additional text or formatting."
    )
    messages = [
            {"role": "system", "content": "Output only lines of `term:score`."},
            {"role": "user",   "content": doc + "\n\n" + instruction}
        ]

    resp = client.chat.completions.create(
        model="deepseek-chat",  
        messages=messages,
        stream = False       
    )

    text = resp.choices[0].message.content.strip()
    lines = [l.strip() for l in text.splitlines() if ":" in l]
    terms = []
    for line in lines:
        term, scr = line.split(":", 1)
        try:
            score = float(scr.strip())
            terms.append((term.strip(), score))
        except ValueError:
            continue
    return terms 

def score_terms_in_doc(doc: str, terms: list[str]) -> list[tuple[str, float]]:
    doc
    term_list = "\n".join(terms)
    instruction = (
        f"Below is a document.  Score each of these terms for how well they "
        f"describe that document on a 0.000–1.000 scale.  Output exactly one `term:score` "
        f"per line, no extra text.\n\n"
        f"Document:\n{doc}\n\nTerms:\n{term_list}"
    )

    messages = [
            {"role": "system", "content": "Output only lines of `term:score`."},
            {"role": "user",   "content": doc + "\n\n" + instruction}
        ]

    resp = client.chat.completions.create(
        model="deepseek-chat",  
        messages=messages,
        stream = False       
    )
    lines = [l.strip() for l in resp.choices[0].message.content.splitlines() if ":" in l]
    scores = []
    for line in lines:
        term, scr = line.split(":",1)
        try:
            scores.append((term.strip(), float(scr)))
        except ValueError:
            continue
    return scores

# ──────────────────────────────────────────────────────────────────────────────
# Build class-importance vectors
# ──────────────────────────────────────────────────────────────────────────────
def compute_class_importances(docs: list[str], labels: list[int], k: int = DOC_KWS_K) -> dict[str, list[float]]:
    num_classes = len(set(labels))
    class_imp = defaultdict(lambda: [0.0] * num_classes)

    for doc, cls in zip(docs, labels):
        terms = build_doc_kws_with_scores(doc, k=k)
        for term, score in terms:
            class_imp[term][cls] += score
        print(terms)

    # normalize each term’s vector
    for term, vec in class_imp.items():
        total = sum(vec)
        if total > 0:
            class_imp[term] = [v / total for v in vec]

    return class_imp

# ──────────────────────────────────────────────────────────────────────────────
# Classifier & unlearning
# ──────────────────────────────────────────────────────────────────────────────
def train_classifier(class_imp: dict[str, list[float]]):
    
    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def classify(doc: str) -> tuple[int, np.ndarray]:
        # embed
        tokens = clean_data(doc)
        candidates = list({w for w in tokens if w in class_imp})
        num_classes = len(next(iter(class_imp.values())))

        if not candidates:
            return 0, np.zeros(num_classes)

        scored = score_terms_in_doc(doc, candidates)

        vec = np.zeros(num_classes)
        for term, score in scored:
            vec += score * np.array(class_imp[term])

        probs = softmax(vec)
        return int(probs.argmax()), probs

    return classify

# ──────────────────────────────────────────────────────────────────────────────
# Membership Inference Attack
# ──────────────────────────────────────────────────────────────────────────────
def membership_inference_attack(classifier, train_docs, test_docs, train_labels, test_labels, target_class: int) -> float:
    # collect features
    X, y = [], []

    # in-class (label=1)
    for i, lbl in enumerate(train_labels):
        if lbl == target_class:
            _, p = classifier(train_docs[i])
            margin = sorted(p)[-1] - sorted(p)[-2]
            X.append([p.max(), margin, -sum(pi * math.log(pi + 1e-12) for pi in p)])
            y.append(1)

    # out-of-class test (label=0)
    for j, lbl in enumerate(test_labels):
        if lbl == target_class:
            _, p = classifier(test_docs[j])
            margin = sorted(p)[-1] - sorted(p)[-2]
            X.append([p.max(), margin, -sum(pi * math.log(pi + 1e-12) for pi in p)])
            y.append(0)

    Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, random_state=42)
    atk = LogisticRegression(class_weight="balanced").fit(Xa, ya)
    return roc_auc_score(yt, atk.predict_proba(Xt)[:, 1])

# ──────────────────────────────────────────────────────────────────────────────
# t-SNE Visualization
# ──────────────────────────────────────────────────────────────────────────────
def tsne_visualization(docs: list[str], labels: list[int], class_imp: dict[str, list[float]], cls_to_unlearn: int):
    points, colors = [], []
    for doc, lbl in zip(docs, labels):
        vec = np.zeros(len(next(iter(class_imp.values()))))
        for w in set(clean_data(doc)):
            if w in class_imp:
                vec += np.array(class_imp[w])
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
    # load & split
    dataset = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    documents = dataset.data
    labels = dataset.target

    split_index = int(0.8 * len(documents))
    train_docs = documents[:split_index]
    test_docs = documents[split_index:]
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]

    # compute class importances
    class_imp = compute_class_importances(train_docs, train_labels, k=DOC_KWS_K)
    terms = list(class_imp.keys())

    # train & evaluate
    classify = train_classifier(class_imp)
    preds = [classify(d)[0] for d in test_docs]
    print("Accuracy before unlearning:", accuracy_score(test_labels, preds))

    # MIA before unlearning
    cls = random.randint(0, max(train_labels))
    tsne_visualization(test_docs, test_labels, class_imp, cls)
    print("MIA AUC before unlearning:", membership_inference_attack(
        classify, train_docs, test_docs, train_labels, test_labels, cls
    ))

    # unlearn one class
    for term, vec in class_imp.items():
        vec[cls] = 0.0
        s = sum(vec)
        class_imp[term] = [v/s for v in vec] if s > 0 else [0.0]*len(vec)

    # re-train & re-evaluate
    classify = train_classifier(class_imp)
    preds2 = [classify(d)[0] for d in test_docs]
    print(f"Accuracy after unlearning class {cls}:", accuracy_score(test_labels, preds2))

    # accuracy on remaining classes
    docs_no  = [d for d,l in zip(test_docs, test_labels) if l != cls]
    labels_no = [l for l in test_labels if l != cls]
    preds3    = [classify(d)[0] for d in docs_no]
    print("Post-unlearning (without that class) acc:", accuracy_score(labels_no, preds3))

    # visualize & MIA again
    tsne_visualization(test_docs, test_labels, class_imp, cls)
    print("MIA AUC after unlearning:", membership_inference_attack(
        classify, train_docs, test_docs, train_labels, test_labels, cls
    ))

if __name__ == "__main__":
    main()