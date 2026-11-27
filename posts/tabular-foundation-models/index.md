---
draft: true
categories:
- machine learning
- tabular data
- foundation models
date: '2026-11-27'
title: "Tabular Foundation Models: Predictions Without Training"
description: What tabular foundation models like TabPFN are, how they differ from the classic data science workflow, and how they compare to gradient-boosted trees in practice.
---

Most real-world machine learning still happens on tables: rows of customers, transactions, sensor readings, or train delays.
For years, the go-to approach for such tabular data has been gradient-boosted decision trees like XGBoost, LightGBM, and CatBoost.
Deep learning, which transformed text and images, struggled to beat them.

Tabular foundation models (TFMs) are changing that.
In this post, I explain what TFMs are, how they differ from the typical data science workflow, and show hands-on how to use TabPFN and how it compares to gradient-boosted trees on a small example.

:::{.callout-note}
This is the first of two posts. The [second post](TODO) covers how to benchmark tabular models systematically and how ensembling multiple TFMs can improve predictions further.
:::


# The Classic Tabular Workflow

A typical data science project on tabular data looks roughly like this:

1. Clean and preprocess the data (missing values, categorical encoding, scaling).
1. Engineer features.
1. Pick a model, usually a gradient-boosted tree.
1. Tune its hyperparameters with cross-validation.
1. Possibly ensemble several tuned models.

Every new dataset starts this process from scratch.
The model learns only from the data in front of it, and getting good results takes time, compute, and experience.

<!-- TODO: optional figure contrasting the two workflows -->


# What Are Tabular Foundation Models?

A tabular foundation model is a neural network (a transformer) that is pretrained once and then applied to new datasets without any further training.

The key idea behind [TabPFN](https://github.com/PriorLabs/TabPFN), the best-known TFM, is:

- **Pretraining on synthetic data.** TabPFN is trained on millions of artificially generated datasets, sampled from a "prior" over plausible data-generating processes (e.g., random causal graphs). It never sees the real dataset you apply it to during pretraining.
- **In-context learning.** At prediction time, you pass the entire training set *and* the test rows to the model in a single forward pass. The model infers the patterns from the training rows and predicts the test rows directly, similar to how a large language model (LLM) answers based on examples in its prompt.
- **No training or tuning per dataset.** "Fitting" TabPFN just stores the training data. There is no gradient descent and no hyperparameter search on your data.

In other words, the model has learned *how to learn* from tables.
Instead of training a new model for every dataset, you reuse one pretrained model.

## Strengths and Limitations

<!-- TODO: double-check current limits for the latest TabPFN version -->

| | Gradient-boosted trees | Tabular foundation models |
|---|---|---|
| Training per dataset | Yes, plus tuning | None (single forward pass) |
| Small datasets | Prone to overfitting | Very strong |
| Large datasets | Scale well | Limited by context size and memory |
| Prediction speed | Very fast | Slower, ideally needs a GPU |
| Uncertainty estimates | Limited | Built-in predictive distributions |

TFMs shine on small to medium datasets, where they often beat tuned gradient-boosted trees out of the box.
Their main costs are memory and slower predictions, since every prediction processes the full training set.
Also check the license of the model weights before using them commercially.


# Hands-On: TabPFN vs. Gradient-Boosted Trees

Using TabPFN is as simple as using any scikit-learn model:

```bash
pip install tabpfn
```

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = TabPFNClassifier()        # no hyperparameters to tune
clf.fit(X_train, y_train)       # stores the training data, no training
proba = clf.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, proba))
```

That's it. No preprocessing pipeline, no tuning loop.

## Example: Airline Passenger Satisfaction

To see how this compares in practice, I extended my [Kaggle notebook on airline passenger satisfaction](https://www.kaggle.com/code/stefanbschneider/airline-satisfaction-ml-cookbook) to compare TabPFN against XGBoost, LightGBM, and CatBoost.

<!-- TODO: describe dataset (rows, features, target), setup (subsample size, CV, metric), and hardware -->

<!-- TODO: results figure, e.g., accuracy/ROC AUC vs. training-set size per model, plus fit + predict time -->

<!-- TODO: results table -->

<!-- TODO: 2-3 sentences on takeaways, e.g., how TabPFN does with small training sets vs. the full dataset, and the speed trade-off -->


# One Dataset Is Just an Anecdote

A comparison on a single dataset is fun, but it doesn't tell us much.
Results depend on the dataset, the train/test split, how much each model was tuned, and which metric we look at.
To know whether TFMs are really better, and when, we need systematic benchmarks across many datasets.

That's the topic of the [next post](TODO): benchmarking tabular models with TabArena, going beyond independent and identically distributed (IID) data with BeyondArena, and ensembling multiple TFMs.


# Further Reading

- [TabPFN on GitHub](https://github.com/PriorLabs/TabPFN) and [docs](https://docs.priorlabs.ai/)
- Hollmann et al., [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6), Nature, 2025
- [TabArena leaderboard](https://tabarena.ai/)
