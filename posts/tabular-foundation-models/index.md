---
aliases:
- /tabular-foundation-models
draft: true
date: '2026-11-27'
title: "Tabular Foundation Models: Getting Started with TabPFN"
description: What tabular foundation models are, how they differ from the usual data science workflow, and how TabPFN compares to gradient boosting on a simple example.
categories: [tabpfn, python, tabular data, foundation models, machine learning]
---

Most machine learning in practice still happens on tabular data, i.e., tables with rows (samples) and columns (features).
For years, gradient-boosted decision trees like [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), and [CatBoost](https://catboost.ai/) have been the default choice for such data.
Deep learning has been very successful for text and images, but on tables it usually could not beat a well-tuned gradient boosting model.

Tabular foundation models (TFMs) like [TabPFN](https://github.com/PriorLabs/TabPFN) take a different approach.
Over the last months, I looked into TFMs in more detail and found them quite impressive.
In this blog post, I explain what TFMs are and how they differ from the usual data science workflow.
I then show how to use TabPFN in just a few lines of Python and compare it against gradient boosting on a simple example.

:::{.callout-note}
This is the first of two posts. In the [second post](TODO), I look at how to benchmark tabular models systematically and how ensembling multiple TFMs can improve predictions further.
:::


# The Usual Workflow for Tabular Data

When working on a new prediction task with tabular data, the workflow typically looks like this:

1. Clean and preprocess the data, e.g., handle missing values and encode categorical features.
1. Engineer useful features.
1. Select a model, often a gradient boosting model.
1. Tune the model's hyperparameters with cross-validation.
1. Optionally, ensemble multiple tuned models.

For each new dataset, we repeat this process and train a new model from scratch.
This works well but takes time, compute, and experience.

<!-- TODO: optional figure comparing the usual workflow with TFMs -->


# What Are Tabular Foundation Models?

TFMs are neural networks (transformers) that are pre-trained once and then used for new datasets without any further training.
For TabPFN, this works roughly as follows:

* **Pre-training on synthetic data:** TabPFN is trained on millions of artificially generated datasets. These datasets are sampled from a prior over possible data-generating processes, e.g., based on random causal graphs. Hence, the name prior-data fitted network (PFN). Notably, TabPFN does not see any real-world data during pre-training.
* **In-context learning:** To make predictions, we pass the training data together with the test samples to the model in a single forward pass. The model then predicts the test samples based on the patterns in the training data. This is similar to how a large language model (LLM) answers based on examples given in the prompt.
* **No training or tuning:** Calling `fit()` on TabPFN does not train the model. It mainly stores the training data for later predictions. There is no gradient descent and no hyperparameter tuning on our dataset.

So instead of training a new model for each dataset, we use the same pre-trained model for all datasets.

As I understand, TFMs work particularly well on small to medium-sized datasets, where they often beat tuned gradient boosting models without any tuning.
The downside is that predictions are slower and need more memory than with gradient boosting since the model processes the entire training data for each prediction.
Ideally, TabPFN runs on a GPU.
There are also limits on how large the training data can be.

<!-- TODO: check current dataset size limits and license for the latest TabPFN version (3.5) -->

Note that the TabPFN model weights come with their own license, which is worth checking before using TabPFN commercially.


# Getting Started with TabPFN

TabPFN implements the familiar scikit-learn interface, so using it is straightforward.
First, install it with:

```bash
pip install tabpfn
```

Then we can use it like any other scikit-learn classifier:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = TabPFNClassifier()
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, proba))
```

That's it! No preprocessing pipeline and no hyperparameter tuning.
`TabPFNRegressor` works the same way for regression tasks.


# Example: TabPFN vs. Gradient Boosting

To see how TabPFN compares to gradient boosting models in practice, I extended my [Kaggle notebook on airline passenger satisfaction](https://www.kaggle.com/code/stefanbschneider/airline-satisfaction-ml-cookbook).
The notebook previously compared XGBoost, LightGBM, and CatBoost. Now, it also includes TabPFN.

<!-- TODO: describe dataset (rows, features, target) and setup (subsample size, CV, metric, hardware) -->

<!-- TODO: results figure, e.g., ROC AUC vs. training set size per model, plus fit and predict time -->

<!-- TODO: results table -->

<!-- TODO: short discussion of results -->

Of course, this is just a single dataset.
The results depend on the dataset, the train/test split, how much effort went into tuning each model, and the metric we look at.
To compare different approaches properly, we need a systematic benchmark across many datasets.
This is the topic of my [next blog post](TODO).


# What's Next?

* [Next post: Benchmarking tabular models and ensembling TFMs](TODO)
* [TabPFN on GitHub](https://github.com/PriorLabs/TabPFN) and the [TabPFN docs](https://docs.priorlabs.ai/)
* [TabPFN paper in Nature](https://www.nature.com/articles/s41586-024-08328-6) by Hollmann et al.
* [TabArena leaderboard](https://tabarena.ai/) comparing tabular models across many datasets
