# -- coding: utf-8 --
"""
<<<<<<< Updated upstream
Created on Sat Apr 20 11:16:55 2024
@author: hp
=======
Created on Fri Apr 26 16:10:22 2024

@author: DELL
>>>>>>> Stashed changes
"""

import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
from collections import Counter
import itertools
=======
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix

# Read data from CSV files

# Preprocessing function
# def preprocess(text):
#     # Check if text is not NaN
#     if isinstance(text, str):
#         # Tokenization
#         tokens = re.findall(r'\b\w+\b', text.lower())
#         # Stop-word removal and stemming can be added here if needed
#         return " ".join(tokens)
#     else:
#         return "doctor"

# # Preprocess text data

# # Make a Directed Graph according to the paper
# def make_graph(string):
#     # Split the string into words
#     chunks = string.split()
#     # Create a directed graph
#     G = nx.DiGraph()
#     # Add nodes for each unique word
#     for chunk in set(chunks):
#         G.add_node(chunk)
#     # Add edges between adjacent words
#     for i in range(len(chunks) - 1):
#         G.add_edge(chunks[i], chunks[i + 1])
#     return G

# # Calculate graph distance
# def graph_distance(graph1, graph2):
#     edges1 = set(graph1.edges())
#     edges2 = set(graph2.edges())
#     common = edges1.intersection(edges2)
#     mcs_graph = nx.Graph(list(common))
#     return -len(mcs_graph.edges())

class GraphKNN:
    def _init_(self, k:int):
        self.k = k
        self.train_graphs = []
        self.train_labels = []
    
    def fit(self, train_graphs, train_labels):
        self.train_graphs = train_graphs
        self.train_labels = train_labels
    
    def predict(self, graph):
        distances = []
        for train_graph in self.train_graphs:
            distance = graph_distance(graph, train_graph)
            distances.append(distance)
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        nearest_labels = [self.train_labels[i] for i in nearest_indices]
        prediction = max(set(nearest_labels), key=nearest_labels.count)
        return prediction


# Plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
import os

# Function to read text files from a directory
def read_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
    return texts

# Function to read a single text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text
>>>>>>> Stashed changes

# Preprocessing function
def preprocess(text):
    if isinstance(text, str):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return " ".join(tokens)
    else:
        return "doctor"

# Make a Directed Graph according to the paper
def make_graph(string):
    chunks = string.split()
    G = nx.DiGraph()
    for chunk in set(chunks):
        G.add_node(chunk)
    for i in range(len(chunks) - 1):
        G.add_edge(chunks[i], chunks[i + 1])
    return G

# Calculate graph distance
def graph_distance(graph1, graph2):
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    common = edges1.intersection(edges2)
    mcs_graph = nx.Graph(list(common))
    return -len(mcs_graph.edges())

<<<<<<< Updated upstream
# Extract common subgraphs from a list of graphs
def extract_common_subgraphs(graphs):
    all_edges = [edge for graph in graphs for edge in graph.edges()]
    common_edges = [edge for edge, count in Counter(all_edges).items() if count == len(graphs)]
    common_subgraphs = []
    for edge in common_edges:
        subgraph = nx.DiGraph()
        subgraph.add_edge(edge[0], edge[1])
        common_subgraphs.append(subgraph)
    return common_subgraphs

# Function to plot common subgraphs
def plot_common_subgraphs(common_subgraphs, category):
    for i, subgraph in enumerate(common_subgraphs, 1):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=2000, edge_color='gray', width=2, arrowsize=20)
        plt.title(f"Common Subgraph {i} ({category})")
        plt.show()

# Function to plot directed graph of each text file
def plot_directed_graphs(text_file_paths, category):
    for i, file_path in enumerate(text_file_paths, 1):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        preprocessed_text = preprocess(text)
        graph = build_graph(preprocessed_text)
        plt.figure(figsize=(10,6))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=2000, edge_color='gray', width=2, arrowsize=20)
        plt.title(f"Directed Graph of Text File {i} ({category})")
        plt.show()

# Generate all induced subgraphs of a graph
def generate_induced_subgraphs(graph):
    nodes = graph.nodes()
    for r in range(1, len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            yield graph.subgraph(subset)

# Extract features from a document based on common subgraphs
def extract_features(document_graph, induced_subgraphs):
    document_features = []
    for subgraph in induced_subgraphs:
        is_present = nx.is_isomorphic(subgraph, document_graph)
        document_features.append(1 if is_present else 0)
    return document_features

# Paths to the text files for fashion, education, and health
fashion_file_paths = [
    r'D:\semester 6\gt\text files\preprocessed_fashion1.txt',
    r'D:\semester 6\gt\text files\preprocessed_fashion2.txt',
    # Add more fashion file paths as needed
]

education_file_paths = [
    r'D:\semester 6\gt\text files\preprocessed_education1.txt',
    r'D:\semester 6\gt\text files\preprocessed_education2.txt',
    # Add more education file paths as needed
]

health_file_paths = [
    r'D:\semester 6\gt\text files\preprocessed_health1.txt',
    r'D:\semester 6\gt\text files\preprocessed_health2.txt',
    # Add more health file paths as needed
]

# Plot directed graphs of each text file for fashion, education, and health
plot_directed_graphs(fashion_file_paths, "Fashion")
plot_directed_graphs(education_file_paths, "Education")
plot_directed_graphs(health_file_paths, "Health")

# Read text from text files for fashion, education, and health
fashion_texts = read_text_from_files(fashion_file_paths)
education_texts = read_text_from_files(education_file_paths)
health_texts = read_text_from_files(health_file_paths)

# Extract common subgraphs for fashion, education, and health
fashion_graphs = [build_graph(preprocess(text)) for text in fashion_texts]
education_graphs = [build_graph(preprocess(text)) for text in education_texts]
health_graphs = [build_graph(preprocess(text)) for text in health_texts]

common_fashion_subgraphs = extract_common_subgraphs(fashion_graphs)
common_education_subgraphs = extract_common_subgraphs(education_graphs)
common_health_subgraphs = extract_common_subgraphs(health_graphs)

# Print count of common subgraphs for fashion, education, and health
print(f"Number of common subgraphs for Fashion: {len(common_fashion_subgraphs)}")
print(f"Number of common subgraphs for Education: {len(common_education_subgraphs)}")
print(f"Number of common subgraphs for Health: {len(common_health_subgraphs)}")

# Plot common subgraphs for fashion, education, and health
# plot_common_subgraphs(common_fashion_subgraphs, "Fashion")
# plot_common_subgraphs(common_education_subgraphs, "Education")
# plot_common_subgraphs(common_health_subgraphs, "Health")

# Generate all induced subgraphs for each document graph
fashion_induced_subgraphs = [list(generate_induced_subgraphs(graph)) for graph in fashion_graphs]
education_induced_subgraphs = [list(generate_induced_subgraphs(graph)) for graph in education_graphs]
health_induced_subgraphs = [list(generate_induced_subgraphs(graph)) for graph in health_graphs]

# Extract features for fashion, education, and health documents
fashion_features = [extract_features(graph, subgraphs) for graph, subgraphs in zip(fashion_graphs, fashion_induced_subgraphs)]
education_features = [extract_features(graph, subgraphs) for graph, subgraphs in zip(education_graphs, education_induced_subgraphs)]
health_features = [extract_features(graph, subgraphs) for graph, subgraphs in zip(health_graphs, health_induced_subgraphs)]
=======
# Read training data and create graphs
train_health_dir = "D:/wish/wish/data/health"
train_fashion_dir = "D:/wish/wish/data/fashion"
train_education_dir = "D:/wish/wish//data/education"

train_health_texts = read_text_files(train_health_dir)[:12]
train_fashion_texts = read_text_files(train_fashion_dir)[:12]
train_education_texts = read_text_files(train_education_dir)[:12]

train_texts = train_health_texts + train_fashion_texts + train_education_texts
train_labels = ['health'] * 12 + ['fashion'] * 12 + ['education'] * 12

train_texts_preprocessed = [preprocess(text) for text in train_texts]
train_graphs = [make_graph(text) for text in train_texts_preprocessed]

# Train the model
graph_classifier = GraphKNN(k=3)
graph_classifier.fit(train_graphs, train_labels)

# Read the text file to test
file_to_test = "D:/wish/wish/data/education/preprocessed_education15.txt"
test_text = read_text_file(file_to_test)

# Preprocess the testing text and create a graph
test_text_preprocessed = preprocess(test_text)
test_graph = make_graph(test_text_preprocessed)

# Predict
prediction = graph_classifier.predict(test_graph)
print("Prediction for the given file:", prediction)
# Plot confusion matrix



test_health_dir = "D:/wish/wish/data/health"
test_fashion_dir = "D:/wish/wish/data/fashion"
test_education_dir = "D:/wish/wish//data/education"


# Read training data
train_health_texts = read_text_files(train_health_dir)[:12]
train_fashion_texts = read_text_files(train_fashion_dir)[:12]
train_education_texts = read_text_files(train_education_dir)[:12]

# Concatenate training texts and create labels
train_texts = train_health_texts + train_fashion_texts + train_education_texts
train_labels = ['health'] * 12 + ['fashion'] * 12 + ['education'] * 12

# Read testing data
test_health_texts = read_text_files(test_health_dir)[-3:]
test_fashion_texts = read_text_files(test_fashion_dir)[-3:]
test_education_texts = read_text_files(test_education_dir)[-3:]

# Concatenate testing texts and create labels
test_texts = test_health_texts + test_fashion_texts + test_education_texts
test_labels = ['health'] * 3 + ['fashion'] * 3 + ['education'] * 3

# Preprocess training and testing texts
train_texts_preprocessed = [preprocess(text) for text in train_texts]
test_texts_preprocessed = [preprocess(text) for text in test_texts]

# Create graphs for training and testing data
train_graphs = [make_graph(text) for text in train_texts_preprocessed]
test_graphs = [make_graph(text) for text in test_texts_preprocessed]

# Train the model
graph_classifier = GraphKNN(k=3)
graph_classifier.fit(train_graphs, train_labels)

# Predict
predictions = [graph_classifier.predict(graph) for graph in test_graphs]

# Evaluate
# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
accuracy_percentage = accuracy * 100
print("Accuracy: {:.2f}%".format(accuracy_percentage))

# Calculate F1 Score for each class
f1_scores = f1_score(test_labels, predictions, average=None)
print("F1 Scores:", f1_scores)

# Calculate Jaccard similarity for each class
jaccard = jaccard_score(test_labels, predictions, average=None)
print("Jaccard Similarity:", jaccard)

# Plot confusion matrix
plot_confusion_matrix(test_labels, predictions, classes=['health', 'fashion', 'education'])
>>>>>>> Stashed changes
