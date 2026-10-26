# Formative Assignment: Probability Distributions, Bayesian Probability and Gradient Descent

**Group:** [Insert Group Number]  
**Course:** Machine Learning  
**Institution:** African Leadership University

## Project Overview

This project explores some of the mathematical ideas used in machine learning. It covers Gaussian distributions, the Expectation-Maximization algorithm, Bayesian probability, and gradient descent.

The assignment combines Python implementation with mathematical calculations to help us understand how machine learning models learn from data, make predictions, and improve their performance.

## Repository Structure

```text
├── GaltonFamilies.csv
├── IMDb Dataset.csv
├── Formative_Assignment.ipynb
└── README.md
```

- `GaltonFamilies.csv`: Dataset used for the height distribution task.
- `IMDb Dataset.csv`: Dataset containing movie reviews used for sentiment analysis.
- `Formative_Assignment.ipynb`: Jupyter notebook containing the code, calculations, outputs, and visualizations.
- `README.md`: Documentation explaining the project and its main steps.

## Part 1: Gaussian Mixture Model and Expectation-Maximization

### Objective

The first task explores how a dataset can be represented using a Gaussian Mixture Model (GMM). A GMM assumes that the data comes from a combination of multiple Gaussian distributions.

A Gaussian distribution is a bell-shaped curve described by its mean and variance.

### Dataset Preparation

We load `GaltonFamilies.csv` using Pandas and select the `father` and `childHeight` columns. These values are combined into a single array called `heights`.

We then shuffle the array to mix the observations before fitting the model.

### Initializing the Model

We start with two Gaussian components. Each component has:

- **Mean:** The center of the distribution.
- **Variance:** How spread out the values are.
- **Mixing proportion:** The estimated fraction of observations belonging to that component.

The initial means are set using the minimum and maximum values in the dataset, adjusted by five units. Both variances start at 10, and both mixing proportions start at 0.5.

### Expectation-Maximization Algorithm

The Expectation-Maximization (EM) algorithm repeatedly updates the model parameters.

**Step 1: Expectation step (E-step)**

The algorithm estimates the probability that each observation belongs to each Gaussian component. These probabilities are called responsibilities.

**Step 2: Maximization step (M-step)**

The algorithm uses the responsibilities to update the means, variances, and mixing proportions of the two components.

**Step 3: Repeat**

The E-step and M-step are repeated for 10 iterations. We record the parameter values and log-likelihood at each iteration to observe how the model changes.

### Tracking and Visualization

The notebook creates a tracking table containing the parameter values and log-likelihood across iterations.

The visualizations help us examine:

- The distribution of the height observations.
- The fitted Gaussian components.
- How the log-likelihood changes during training.

Log-likelihood measures how well the model explains the observed data. An increasing log-likelihood generally indicates that the model is fitting the data better.

### Classifying a Height

The `classify_height()` function calculates the relative probabilities that a given height belongs to Component 1 or Component 2.

The calculation is performed in log space to reduce numerical underflow, which can occur when Gaussian probabilities become extremely small.

To test a height, run:

```python
classify_height(75)
```

Replace `75` with the test value specified in the assignment. The input must use the same units as the dataset.

**Note:** The two components are statistical groups learned by the model. Their labels should not automatically be interpreted as specific real-world categories without evidence or the correct dataset.

## Part 2: Bayesian Probability and Sentiment Analysis

### Objective

The second task applies Bayesian probability to movie reviews from the IMDb dataset. The goal is to explore how the presence of certain words relates to whether a review is positive or negative.

### Dataset Preparation

We load `IMDb Dataset.csv` using Pandas and examine the review text and sentiment labels.

The dataset is then used to count reviews containing selected keywords under each sentiment category.

### Selecting Keywords

The notebook uses the following words:

**Positive keywords**
- fantastic
- wonderful
- stunning
- best

**Negative keywords**
- bad
- boring
- awful
- worst

These words are used to examine the relationship between vocabulary and review sentiment.

### Applying Bayes' Theorem

Bayes' theorem helps us update a probability when new evidence is available.

The formula is:

\[
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}
\]

Where:

- \(P(A\mid B)\) is the posterior probability: the probability of an event after considering evidence.
- \(P(B\mid A)\) is the likelihood: the probability of observing the evidence given the event.
- \(P(A)\) is the prior probability: the probability of the event before considering the evidence.
- \(P(B)\) is the marginal probability: the overall probability of observing the evidence.

In this task, the notebook calculates probabilities related to positive sentiment and the selected keywords. These calculations demonstrate how evidence from words in a review can change our estimate of its sentiment.

### Interpreting the Results

The results help us examine whether selected words are associated more strongly with positive or negative reviews.

However, a keyword alone does not always determine sentiment. For example, a review might use a positive word sarcastically or contain both positive and negative opinions.

The results should therefore be interpreted as probabilities based on the dataset, rather than as a guarantee that a review has a particular sentiment.

## Part 3: Manual Gradient Descent Calculations

This section presents the mathematical calculations required by the assignment.

Gradient descent is an optimization method
