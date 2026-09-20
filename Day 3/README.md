# Day 3 — Logistic Regression + Binary Classification

## Overview

Day 3 focused on understanding how a machine learning model can move from predicting continuous values to making **binary classification** decisions.

Building directly on the linear regression implementation from Day 2, this day introduced **Logistic Regression** and the mathematical foundations behind binary classification.

The main goal was not to use a pre-built machine learning library, but to understand and implement the complete learning process manually using **NumPy**.

---

## Learning Objectives

By the end of Day 3, I learned how to:

* Understand binary classification
* Understand why linear regression is not directly suitable for classification
* Derive and implement the sigmoid activation function
* Interpret sigmoid outputs as probabilities
* Convert probabilities into binary predictions using a threshold
* Understand decision boundaries
* Understand Binary Cross-Entropy / Log Loss
* Calculate gradients for logistic regression
* Implement gradient descent for logistic regression
* Extend logistic regression from one feature to multiple features
* Standardize features before optimization
* Implement train/test splitting
* Train a logistic regression model entirely from scratch using NumPy
* Evaluate predictions against actual labels

---

# 1. From Linear Regression to Classification

In Day 2, the model was:

$$
\hat{y}=Xw+b
$$

This works naturally for continuous targets such as:

* house prices
* temperature
* salary
* demand

However, binary classification requires predictions such as:

$$
y\in\{0,1\}
$$

For example:

```text
0 → Negative
1 → Positive
```

A linear model can produce arbitrary values:

```text
-4.2
0.7
3.8
17.2
```

These are not probabilities.

Therefore, we first calculate a linear score:

$$
z=Xw+b
$$

and then transform it into a value between 0 and 1.

---

# 2. Sigmoid Function

The sigmoid function is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

It maps any real-valued input into:

$$
0<\sigma(z)<1
$$

Important properties:

$$
\sigma(0)=0.5
$$

$$
z\rightarrow+\infty
\Rightarrow
\sigma(z)\rightarrow1
$$

$$
z\rightarrow-\infty
\Rightarrow
\sigma(z)\rightarrow0
$$

The implementation used was:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

NumPy's `np.exp()` allows the function to operate both on individual values and entire arrays.

---

# 3. Logistic Regression

The complete logistic regression model is:

$$
z=Xw+b
$$

followed by:

$$
p=\sigma(z)
$$

Therefore:

$$
\boxed{
p=P(y=1|X)
}
$$

The model output is a **probability**, not yet a class.

For example:

```text
p = 0.12
p = 0.37
p = 0.81
p = 0.97
```

These represent the model's estimated probability that the example belongs to class 1.

---

# 4. Probability to Class

To convert the probability into a binary prediction, a threshold can be used.

With the standard threshold:

$$
0.5
$$

we have:

$$
\hat y=
\begin{cases}
1 & p\ge0.5\\
0 & p<0.5
\end{cases}
$$

Implementation:

```python
predictions = (p >= 0.5).astype(int)
```

An important distinction learned on Day 3:

> Logistic regression produces a probability.
> The threshold converts that probability into a class.

---

# 5. Decision Boundary

The decision boundary occurs at:

$$
p=0.5
$$

Since:

$$
\sigma(z)=0.5
$$

when:

$$
z=0
$$

the decision boundary is:

$$
Xw+b=0
$$

For one feature:

$$
wx+b=0
$$

and therefore:

$$
x=-\frac{b}{w}
$$

This means that although logistic regression produces probabilities, the underlying decision boundary is still determined by a **linear function**.

---

# 6. Binary Cross-Entropy

A classification model needs a loss function to measure how good its predictions are.

For binary classification, Binary Cross-Entropy (BCE), also called Log Loss, is used:

$$
L=
-\frac1m
\sum_{i=1}^{m}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
$$

For a single example:

$$
L=
-\left[
y\log(p)+(1-y)\log(1-p)
\right]
$$

### If the true class is 1

The loss becomes:

$$
L=-\log(p)
$$

A prediction of:

$$
p=0.99
$$

produces a very small loss.

A prediction of:

$$
p=0.01
$$

produces a very large loss.

### If the true class is 0

The loss becomes:

$$
L=-\log(1-p)
$$

Therefore, confidently incorrect predictions are heavily penalized.

---

# 7. Gradient Derivation

The model is:

$$
z=Xw+b
$$

$$
p=\sigma(z)
$$

$$
L=\text{BCE}(p,y)
$$

Applying the chain rule gives the particularly useful result:

$$
\boxed{
\frac{\partial L}{\partial z}=p-y
}
$$

This became the central error signal for the model.

For the weights:

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac1mX^T(p-y)
}
$$

For the bias:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac1m\sum(p-y)
}
$$

The intuition behind \(p-y\):

### Actual = 1

If:

$$
y=1,\quad p=0.8
$$

then:

$$
p-y=-0.2
$$

The prediction is too low, so the optimization needs to move the model toward a higher output.

### Actual = 0

If:

$$
y=0,\quad p=0.8
$$

then:

$$
p-y=+0.8
$$

The prediction is too high, so the optimization needs to move the model toward a lower output.

Thus:

> \(p-y\) provides a signed error signal that determines the direction in which the parameters need to move.

---

# 8. Gradient Descent

Once the gradients are calculated, the parameters are updated using gradient descent:

$$
w\leftarrow
w-\eta\frac{\partial L}{\partial w}
$$

$$
b\leftarrow
b-\eta\frac{\partial L}{\partial b}
$$

where:

$$
\eta
$$

is the learning rate.

The complete training cycle is:

```text
Forward Pass
     ↓
Prediction
     ↓
Loss
     ↓
Gradient
     ↓
Parameter Update
     ↓
Repeat
```

This was the first time the complete classification learning loop was implemented from scratch.

---

# Project 1 — Logistic Regression from Scratch

## Objective

Implement logistic regression using only NumPy on a tiny binary classification dataset.

### Dataset

```python
X = np.array([1, 2, 3, 4, 5])
y = np.array([0, 0, 0, 1, 1])
```

Initial parameters:

```python
w = 1.0
b = -3.0
```

---

## Forward Pass

```python
z = w * X + b
p = sigmoid(z)
```

Initial probabilities were approximately:

```text
[0.1192  0.2689  0.5000  0.7311  0.8808]
```

The model initially classified:

```text
[0 0 1 1 1]
```

while the actual labels were:

```text
[0 0 0 1 1]
```

The example at \(X=3\) was initially sitting exactly at the decision boundary.

---

## Loss

Binary cross-entropy was implemented manually:

```python
loss = -(1 / len(y)) * np.sum(
    y * np.log(p) +
    (1 - y) * np.log(1 - p)
)
```

---

## Gradients

The gradients were implemented as:

```python
dw = (1 / len(y)) * (X.T @ (p - y))
db = (1 / len(y)) * np.sum(p - y)
```

---

## Training

Gradient descent was then applied repeatedly:

```python
w = w - lr * dw
b = b - lr * db
```

After training, the model learned approximately:

```text
w ≈ 11.881
b ≈ -41.451
```

The final probabilities were approximately:

```text
[1.44e-13,
 2.08e-08,
 2.996e-03,
 9.977e-01,
 9.99999984e-01]
```

The resulting predictions were:

```text
Predictions:
[0 0 0 1 1]

Actual:
[0 0 0 1 1]
```

The model successfully classified all five training examples.

---

# Project 2 — Breast Cancer Classification from Scratch

## Objective

Apply the same mathematics to a real-world dataset with:

* 569 examples
* 30 features
* 2 classes

The goal was to build a multi-feature logistic regression classifier without using a pre-built logistic regression model.

---

## Dataset

The Breast Cancer Wisconsin dataset was loaded for the project.

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target
```

The dataset dimensions were:

```text
X → (569, 30)
y → (569,)
```

There were 30 numerical features, including measurements such as:

* mean radius
* mean texture
* mean perimeter
* mean area
* mean smoothness
* mean compactness
* mean concavity
* worst radius
* worst texture
* worst perimeter
* worst area
* etc.

The two classes were distributed as:

```text
Class 0 → 212
Class 1 → 357
```

---

# Train/Test Split

The dataset was divided into:

```text
80% → training data
20% → test data
```

The split was implemented using NumPy rather than relying on a pre-built splitting function.

Approximate resulting dimensions:

```text
X_train → (455, 30)
X_test  → (114, 30)

y_train → (455,)
y_test  → (114,)
```

The test set was kept separate so that it could represent unseen data during evaluation.

---

# Feature Standardization

The 30 features had very different numerical scales.

To make gradient descent behave more effectively, the features were standardized.

For each feature:

$$
x'=\frac{x-\mu}{\sigma}
$$

where:

* \(\mu\) = training-set feature mean
* \(\sigma\) = training-set feature standard deviation

The means and standard deviations were calculated **only from the training data**.

Then the same values were used to transform the test data.

This avoids test-set information leaking into the training process.

After standardization:

```text
Training feature mean ≈ 0
Training feature standard deviation ≈ 1
```

---

# Multi-Feature Logistic Regression

The one-feature model:

$$
z=wx+b
$$

became the multi-feature model:

$$
\boxed{
z=Xw+b
}
$$

where:

$$
X\in\mathbb{R}^{455\times30}
$$

and:

$$
w\in\mathbb{R}^{30}
$$

The implementation therefore used:

```python
w = np.zeros(X_train_scaled.shape[1])
b = 0.0
```

and the forward pass:

```python
z = X_train_scaled @ w + b
p = sigmoid(z)
```

The resulting probability vector had shape:

```text
(455,)
```

---

# Gradient Calculation

The multi-feature gradients were:

```python
dz = p - y_train

dw = (1 / len(y_train)) * (
    X_train_scaled.T @ dz
)

db = (1 / len(y_train)) * np.sum(dz)
```

This is the vectorized form of the same mathematics used in the one-feature implementation.

---

# Training Loop

The final training structure was:

```python
w = np.zeros(X_train_scaled.shape[1])
b = 0.0

lr = 0.1

for epoch in range(1000):

    # Forward pass
    z = X_train_scaled @ w + b
    p = sigmoid(z)

    # Binary cross-entropy
    loss = -(1 / len(y_train)) * np.sum(
        y_train * np.log(p) +
        (1 - y_train) * np.log(1 - p)
    )

    # Gradients
    dz = p - y_train

    dw = (1 / len(y_train)) * (
        X_train_scaled.T @ dz
    )

    db = (1 / len(y_train)) * np.sum(dz)

    # Gradient descent
    w = w - lr * dw
    b = b - lr * db
```

This is a complete logistic regression optimizer implemented from scratch.

---

# Key Lessons

## 1. Logistic regression is still based on a linear model

The underlying score is:

$$
z=Xw+b
$$

The sigmoid simply transforms that score into a probability.

---

## 2. Probability and classification are different

The model produces:

$$
P(y=1|X)
$$

A threshold then produces:

$$
\hat y\in\{0,1\}
$$

---

## 3. Cross-entropy cares about confidence

Being confidently wrong produces a much larger penalty than being uncertain.

---

## 4. \(p-y\) is the core error signal

For sigmoid + binary cross-entropy:

$$
\frac{\partial L}{\partial z}=p-y
$$

This makes the gradient calculation particularly elegant.

---

## 5. Multiple features are handled naturally with matrices

The one-feature equation:

$$
wx+b
$$

becomes:

$$
Xw+b
$$

The underlying learning algorithm remains the same.

---

## 6. Feature scaling matters

When features have very different scales, gradient descent can behave poorly.

Standardization puts features on comparable scales and generally makes optimization much easier.

---

## 7. Training and evaluation data must remain separate

The model should learn from the training set and be evaluated on data it did not use during training.

---

# Day 3 Mental Model

The entire day can be summarized as:

```text
                  LOGISTIC REGRESSION

                       Input X
                          │
                          ▼
                    Linear Score
                     z = Xw + b
                          │
                          ▼
                       Sigmoid
                    p = σ(z)
                          │
                          ▼
                    Probability
                  P(y = 1 | X)
                          │
                          ▼
                    Threshold 0.5
                          │
                          ▼
                    Class 0 / 1
                          │
                          ▼
                     Binary Loss
                          │
                          ▼
                      Gradient
                       p - y
                          │
                          ▼
                  Gradient Descent
                          │
                          ▼
                     Update w,b
                          │
                          └──────► Repeat
```

---

# Day 3 Final Takeaway

Day 2 established the mathematics of **linear prediction and gradient descent**.

Day 3 extended that foundation into **probabilistic binary classification**.

The most important equation to carry forward is:

$$
\boxed{
P(y=1|X)=\sigma(Xw+b)
}
$$

combined with:

$$
\boxed{
L_{\text{BCE}}
=
-\frac1m
\sum
[y\log(p)+(1-y)\log(1-p)]
}
$$

and:

$$
\boxed{
dw=\frac1mX^T(p-y)
}
$$

$$
\boxed{
db=\frac1m\sum(p-y)
}
$$

At this point, logistic regression is no longer a black box: its **forward pass, probability calculation, loss, gradients, and parameter updates** have all been implemented manually.

---

## Projects Completed

### Project 1 — Logistic Regression Toy Model

**Skills practiced:**

* Sigmoid
* Probability prediction
* Binary cross-entropy
* Gradient calculation
* Gradient descent
* Decision boundary
* Binary predictions

### Project 2 — Breast Cancer Binary Classifier

**Skills practiced:**

* Real-world dataset
* Train/test split
* Feature standardization
* 30-dimensional parameter vector
* Matrix-vector multiplication
* Vectorized gradients
* Logistic regression training from scratch
* End-to-end binary classification

---

## Day 3 Status

**Completed concepts:**

* [x] Binary classification
* [x] Sigmoid
* [x] Logistic regression
* [x] Probability prediction
* [x] Classification threshold
* [x] Decision boundary
* [x] Binary cross-entropy
* [x] Gradient derivation
* [x] Gradient descent
* [x] Multi-feature logistic regression
* [x] Train/test split
* [x] Feature standardization
* [x] NumPy implementation from scratch

**Core implementation:**
`Logistic Regression + Binary Classification`

**Primary tools:**
`Python + NumPy`

**Secondary tool:**
`scikit-learn` only for loading the real dataset, not for training the classifier.
