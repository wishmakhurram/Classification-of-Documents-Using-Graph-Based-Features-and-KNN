# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:16:55 2024
@author: hp
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import string
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import itertools

# Preprocessing function
def preprocess(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    return stemmed_tokens

# Function to build directed graph
def build_graph(tokens):
    graph = nx.DiGraph()
    for i in range(len(tokens)-1):
        if graph.has_edge(tokens[i], tokens[i+1]):
            graph[tokens[i]][tokens[i+1]]['weight'] += 1
        else:
            graph.add_edge(tokens[i], tokens[i+1], weight=1)
    return graph

# Read text from multiple text files
def read_text_from_files(file_paths):
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

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
