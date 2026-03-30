# A-Comparative-Study-of-Lexical-Algorithms-with-Visualizations

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive empirical analysis comparing three fundamental text vectorization techniques for classification tasks: **One-Hot Encoding**, **Bag-of-Words (BoW)**, and **Term Frequency-Inverse Document Frequency (TF-IDF)**, each paired with Logistic Regression.

## 📊 Overview

This project systematically evaluates and compares the performance, efficiency, and resource consumption of different lexical vectorization methods on the 20 Newsgroups dataset. The study provides insights into the trade-offs between classification accuracy, training time, and memory footprint.

### Key Findings

| Technique | Accuracy | Model Training Time | Vectorizer Training Time | Total Time |
|-----------|----------|---------------------|--------------------------|------------|
| **One-Hot Encoding** | 96.14% | 5.23s | 1.35s | 6.58s |
| **Bag-of-Words** | 96.14% | 2.79s | 1.50s | 4.29s |
| **TF-IDF** | **96.54%** | 3.17s | **1.34s** | 4.51s |

**Winner:** TF-IDF provides the best balance of accuracy and efficiency! ✨

## 🎯 Objectives

- **Compare Performance Metrics**: Evaluate accuracy, precision, recall, and F1-score across three vectorization techniques
- **Analyze Computational Efficiency**: Measure model and vectorizer training times
- **Assess Memory Footprint**: Compare memory consumption of models, vectorizers, and transformed data
- **Identify Trade-offs**: Highlight strengths and weaknesses of each approach for informed decision-making

## 🔬 Methodology

### Dataset
- **Source**: 20 Newsgroups dataset
- **Categories**: 4 selected categories (alt.atheism, comp.graphics, rec.autos, sci.med)
- **Total Documents**: 3,752 documents
- **Split**: 80% training, 20% testing

### Preprocessing Pipeline
1. Text cleaning (removal of special characters, URLs, email addresses)
2. Lowercasing
3. Tokenization using NLTK
4. Stopword removal
5. Lemmatization

### Vectorization Techniques

#### 1. One-Hot Encoding
- **Method**: Binary representation of word presence
- **Implementation**: `CountVectorizer(binary=True)`
- **Characteristics**: Simple, captures word presence/absence

#### 2. Bag-of-Words (BoW)
- **Method**: Word frequency counting
- **Implementation**: `CountVectorizer()`
- **Characteristics**: Captures word frequency, ignores word order

#### 3. TF-IDF (Term Frequency-Inverse Document Frequency)
- **Method**: Weighted term importance
- **Implementation**: `TfidfVectorizer()`
- **Characteristics**: Emphasizes distinctive words, reduces common word impact

### Classification Model
- **Algorithm**: Logistic Regression
- **Solver**: lbfgs
- **Max Iterations**: 1000
- **Random State**: 42 (for reproducibility)

## 📈 Results

### Performance Metrics

<table>
<tr>
<th>Metric</th>
<th>One-Hot Encoding</th>
<th>Bag-of-Words</th>
<th>TF-IDF</th>
</tr>
<tr>
<td><b>Accuracy</b></td>
<td>0.9614</td>
<td>0.9614</td>
<td><b>0.9654</b></td>
</tr>
<tr>
<td><b>Precision</b></td>
<td>0.9625</td>
<td>0.9630</td>
<td><b>0.9667</b></td>
</tr>
<tr>
<td><b>Recall</b></td>
<td>0.9614</td>
<td>0.9614</td>
<td><b>0.9654</b></td>
</tr>
<tr>
<td><b>F1-Score</b></td>
<td>0.9614</td>
<td>0.9613</td>
<td><b>0.9651</b></td>
</tr>
</table>

### Training Time Analysis

- **Fastest Vectorizer Training**: TF-IDF (1.34s)
- **Fastest Model Training**: Bag-of-Words (2.79s)
- **Fastest Total Pipeline**: Bag-of-Words (4.29s)

### Memory Footprint

All three techniques showed similar memory consumption:
- **Vectorizer Memory**: ~0.92 MB
- **Training Data (X_train)**: ~3.97 MB
- **Test Data (X_test)**: ~0.90 MB

## 🎨 Visualizations

The notebook includes comprehensive visualizations:

1. **Performance Metrics Comparison**: Side-by-side bar charts for accuracy, precision, recall, and F1-score
2. **Training Time Analysis**: Comparison of model and vectorizer training times
3. **Memory Footprint Visualization**: Grouped bar charts showing memory usage across components
4. **Normalized Heatmap**: Multi-metric comparison with normalized scores
5. **Radar Chart**: Multi-dimensional performance visualization
6. **Stacked Bar Chart**: Total pipeline time breakdown

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.7
scikit-learn >= 1.0
nltk >= 3.6
pandas >= 1.3
matplotlib >= 3.4
seaborn >= 0.11
numpy >= 1.21
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lexical-vectorization-comparison.git
cd lexical-vectorization-comparison
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Usage

#### Running the Notebook

```bash
jupyter notebook A_Comparative_Study_of_Lexical_Vectorization_Techniques_for_Text_Classification.ipynb
```

Or use Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/lexical-vectorization-comparison/blob/main/A_Comparative_Study_of_Lexical_Vectorization_Techniques_for_Text_Classification.ipynb)

#### Running Individual Components

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
categories = ['alt.atheism', 'rec.autos', 'comp.graphics', 'sci.med']
data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data.data)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, data.target, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## 📊 Key Insights

### Strengths and Weaknesses

#### One-Hot Encoding ✓
- ✅ **Strengths**: Simple to understand, captures word presence effectively
- ❌ **Weaknesses**: Slowest model training time (5.23s), ignores word frequency

#### Bag-of-Words ✓
- ✅ **Strengths**: Fastest model training (2.79s), captures word frequency
- ❌ **Weaknesses**: Ignores word order and context, slightly slower vectorizer training

#### TF-IDF ✓✓
- ✅ **Strengths**: Best overall accuracy (96.54%), fastest vectorizer training (1.34s), emphasizes important words
- ❌ **Weaknesses**: Marginally slower model training than BoW

### Recommendations

**For Production Systems:**
- **Use TF-IDF** for the best balance of accuracy, training speed, and interpretability
- Suitable for most text classification tasks

**For Real-time Applications:**
- **Use Bag-of-Words** when model training speed is critical and a 0.4% accuracy drop is acceptable

**For Simple Baseline:**
- **Use One-Hot Encoding** as a quick baseline or when word presence is more important than frequency

## 🔍 Future Work

- [ ] Extend comparison to advanced techniques (Word2Vec, GloVe, BERT embeddings)
- [ ] Test on additional datasets (sentiment analysis, spam detection)
- [ ] Evaluate with different classifiers (SVM, Random Forest, Neural Networks)
- [ ] Analyze impact of vocabulary size on performance
- [ ] Investigate n-gram features (bigrams, trigrams)
- [ ] Add cross-validation for more robust results
- [ ] Implement hyperparameter tuning

## 📚 References

1. Scikit-learn Documentation: [Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
3. 20 Newsgroups Dataset: [Dataset Description](http://qwone.com/~jason/20Newsgroups/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors

- **Subhrajit Mukherjee** -

## 🙏 Acknowledgments

- scikit-learn team for excellent documentation and tools
- NLTK team for natural language processing utilities
- 20 Newsgroups dataset contributors
- The open-source community for inspiration and support

## 📧 Contact

For questions or feedback, please reach out:
- Email: SubhrajitMukherjee04@gmail.com
- LinkedIn: [Your Profile](https://www.linkedin.com/in/subhrajit-mkj/)

---

⭐ If you found this project helpful, please consider giving it a star!
