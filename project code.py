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

# Paths to the text files
text_file_paths = [
    r'D:\semester 6\gt\text files\preprocessed_fashion1.txt',
    r'D:\semester 6\gt\text files\preprocessed_fashion2.txt',
    r'D:\semester 6\gt\text files\preprocessed_fashion3.txt',
    r'D:\semester 6\gt\text files\preprocessed_fashion4.txt',
    # r'D:\semester 6\gt\health.txt',
    # r'D:\semester 6\gt\education.txt'
]

# Read text from text files
document_texts = read_text_from_files(text_file_paths)

# Plot separate graphs for each document
for i, text in enumerate(document_texts, 1):
    # Preprocess the text
    preprocessed_text = preprocess(text)

    # Build the directed graph
    graph = build_graph(preprocessed_text)

    # Draw the graph
    plt.figure(figsize=(10,6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=2000, edge_color='gray', width=2, arrowsize=20)
    plt.title(f"Directed Graph of Document {i} Terms")
    plt.show()
