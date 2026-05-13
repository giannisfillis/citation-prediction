# 📄 Citation Prediction in Academic Networks using NLP and Graph Embeddings

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Task-Link%20Prediction-orange.svg" alt="Task">
  <img src="https://img.shields.io/badge/Domain-NLP%20%26%20Graphs-green.svg" alt="Domain">
</p>

## 📌 Overview
This repository contains the code, report, and methodology for the **Citation Prediction (Link Prediction)** Kaggle Challenge. The primary objective of this project is to predict whether a given research paper cites another paper based on a complex citation network of several thousand papers. By analyzing textual data from paper abstracts, authorship overlap, and topological features of the citation graph, we built a robust supervised machine learning pipeline to forecast hidden or future citations.

## 🏛 Academic Context
- **Institution:** University of Ioannina (UOI) - Dept. of Computer Science & Engineering
- **Course:** Natural Language Processing (MYE053) - Spring 2025
- **Team Name:** Neuralholic
- **Author:** Giannis Fillis 

## 📊 Dataset & Preprocessing
The dataset consists of thousands of academic papers. The available core data files include:
- `abstracts.txt`: Contains Paper IDs and their corresponding textual abstracts.
- `authors.txt`: Contains Paper IDs and lists of contributing authors.
- `edgelist.txt`: Directed edges representing known citations (Paper $i$ citing Paper $j$).
- `test.txt`: The test set evaluated for the Kaggle submission.

### Negative Example Generation
Since the provided `edgelist.txt` only contains positive examples (known citations), we generated **negative examples** by randomly sampling pairs of papers $(i, j)$ and ensuring no direct or reverse edge exists between them in the graph, thus preventing model confusion.

### Text Preprocessing
Prior to feature extraction, abstracts underwent a rigorous NLP preprocessing pipeline:
1. Lowercasing and punctuation removal.
2. Tokenization and NLTK stop-word removal.
3. Frequency analysis and removal of the **top 25 most common terms** to eliminate domain-generic noise and highlight topic-specific keywords.

## ⚙️ Methodology & Feature Engineering
To train our models, a rich and highly dimensional feature set was extracted, representing semantic similarity, graph topology, and authorship relations.

### 1. Textual Features (Semantic Similarity)
We extracted dense embeddings to calculate the cosine similarity between the abstracts of citing and cited papers:
- **TF-IDF:** Traditional statistical term-weighting.
- **Word2Vec (Gensim):** Skip-Gram architecture (vector size: 400, window: 5, epochs: 20). Missing abstracts were gracefully imputed using the average embeddings of their network neighbors.
- **Doc2Vec:** For document-level dense representations.
- **Sentence Transformers (SPECTER2):** State-of-the-art `allenai-specter` model specialized for scientific documents.

### 2. Graph/Network Features (Topological Importance)
Utilized `NetworkX` to map the citation network and extract node-level and edge-level metrics:
- **Node2Vec:** Dense topological representations of nodes based on random walks (dimensions: 512).
- **Centrality Measures:** PageRank, Katz Centrality, and Eigenvector Centrality.
- **Degrees:** In-degree of the cited paper ($j$) and Out-degree of the citing paper ($i$).
- **HITS Algorithm:** Hub and Authority scores.
- **Neighborhood Overlap:** Jaccard Similarity based on common predecessors and successors.

### 3. Authorship Overlap
- Computed the **Sørensen–Dice coefficient** between the author lists of paper pairs to capture collaborative dynamics (prioritizing shared authorship significance over unshared).

### 4. Engineered Interaction Features
Using **Mutual Information (MI) Gain**, we discovered and engineered high-impact meta-features (e.g., $Textual\_Similarity \times Out\_Degree\_i$, $In\_Degree\_j \times Out\_Degree\_i$, $In\_Degree\_j \times Hub\_Citing$).

## 🚀 Models & Evaluation
Various machine learning algorithms were trained and evaluated using **Logarithmic Loss (Log Loss)** as the primary evaluation metric (optimized via 5-fold Cross-Validation).

| Model | Feature Set | CV Score (Log Loss) |
|-------|-------------|---------------------|
| **Gaussian Naive Bayes** | node2vec, sent_transf | 0.31508 |
| **Logistic Regression** | Features Selected from RFEC | 0.14760 |
| **SGD Classifier** | Features Selected from RFEC | 0.19081 |
| **Decision Trees** | All Features + Top Engineered | 0.25858 |
| **Random Forest** | All Features + Top Engineered | 0.17985 |
| **MLP Classifier** | All Features + Top Engineered | **0.13983** |
| **XGBoost** | Top 10 Features | **0.14641** |

> 🏆 **Conclusion:** The **MLP Classifier** and **XGBoost** models yielded the most competitive performance. The integration of advanced contextual embeddings (SPECTER2) combined with extensive graph topology features (PageRank, Katz, Node2Vec) significantly boosted predictive accuracy.

## 🏃‍♂️ How to Run
1. **Data Preparation:** Place the provided dataset files (`abstracts.txt`, `authors.txt`, `edgelist.txt`, `test.txt`) into a directory named `data/` in the root folder.
2. **Environment Setup:** Ensure all dependencies are installed using the command provided in the Requirements section.
3. **Execution:** Open the Jupyter Notebook `citation-prediction.ipynb` using JupyterLab or VS Code.
4. **Pipeline:** Run the cells sequentially. The notebook follows a structured path:
    - Text Preprocessing & Cleaning.
    - Feature Engineering (TF-IDF, Word2Vec, SPECTER2, Graph Metrics).
    - Model Training & Cross-Validation.
    - Final Prediction Generation.
5. **Results:** Final prediction outputs will be exported as `.csv` files in the `submissions/` directory, formatted for Kaggle submission.
