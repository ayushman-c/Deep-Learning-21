# Day 2 — Neural Network Mathematics

Day 2 focuses on the mathematical foundations behind neural-network computation and learning.

The goal was to move beyond a single neuron and understand how **vectors, matrices, derivatives, gradients, and gradient descent** allow us to build and train models efficiently.

---

## 🎯 Objectives

* Understand vectors and dot products
* Understand matrices and matrix multiplication
* Understand derivatives and their role in optimization
* Understand the chain rule
* Understand gradients
* Understand learning rate and gradient descent
* Implement Linear Regression from scratch using NumPy
* Extend the model from one feature to multiple features

---

## 🧠 Concepts Learned

### 1. Vectors

A vector is an ordered collection of numbers.

For example:

```python
x = np.array([2, 4, 6])
```

Vectors allow multiple related values to be represented and processed together.

---

### 2. Dot Product

The dot product multiplies corresponding elements and sums the results.

$$
w \cdot x = \sum_i w_i x_i
$$

This is the same operation used inside a neuron:

$$
z = w \cdot x + b
$$

---

### 3. Matrices

A matrix is a two-dimensional arrangement of numbers.

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
```

In machine learning, rows commonly represent training examples while columns represent features.

---

### 4. Matrix Multiplication

Matrix multiplication follows the rule:

$$
(m \times n)(n \times p)=(m \times p)
$$

The inner dimensions must match.

For example:

$$
(7\times3)(3\times2)=(7\times2)
$$

Each element of the resulting matrix is produced by a dot product between a row and a column.

---

## 📐 Derivatives

A derivative measures how much one quantity changes with respect to another.

For:

$$
y=x^2
$$

the derivative is:

$$
\frac{dy}{dx}=2x
$$

In machine learning, derivatives tell us how the loss changes when a parameter changes.

For example:

$$
\frac{\partial L}{\partial w}
$$

describes the sensitivity of the loss to the weight \(w\).

---

## 🔗 Chain Rule

Neural networks contain chains of mathematical operations.

For example:

```text
w → z → ŷ → L
```

The chain rule allows us to calculate how a change in an earlier parameter affects the final loss.

For example:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial z}
\frac{\partial z}{\partial w}
$$

This forms the mathematical foundation of **backpropagation**.

---

## 📈 Gradients

When a model has multiple parameters, we calculate a partial derivative for each parameter.

Together, these derivatives form the gradient:

$$
\nabla L=
\begin{bmatrix}
\frac{\partial L}{\partial w_1}\\
\frac{\partial L}{\partial w_2}\\
\vdots
\end{bmatrix}
$$

The gradient tells us how the loss changes with respect to the model's parameters.

---

## 🏃 Gradient Descent

Gradient descent updates parameters in the direction that reduces the loss.

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla L
$$

where:

* \(\theta\) = model parameter
* \(\eta\) = learning rate
* \(\nabla L\) = gradient of the loss

The learning rate controls the size of each update.

A very large learning rate can cause the optimization to overshoot or diverge, while a very small learning rate can make training unnecessarily slow.

---

# 🛠️ Main Project — Linear Regression From Scratch

The main project was to implement Linear Regression using only NumPy.

The model:

$$
\hat y = wx+b
$$

The loss function was Mean Squared Error:

$$
L=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat y_i-y_i)^2
$$

The gradients were derived manually:

$$
\frac{\partial L}{\partial w}
=
\frac{2}{n}
\sum_{i=1}^{n}
(\hat y_i-y_i)x_i
$$

$$
\frac{\partial L}{\partial b}
=
\frac{2}{n}
\sum_{i=1}^{n}
(\hat y_i-y_i)
$$

Parameter updates:

```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

The complete training process was:

```text
Input Data
    ↓
Forward Pass
    ↓
Prediction
    ↓
Loss
    ↓
Gradients
    ↓
Parameter Update
    ↓
Repeat
```

---

# 🚀 Extension — Multiple Features

The model was then extended from one feature to multiple features.

Instead of:

$$
\hat y=wx+b
$$

the model became:

$$
\boxed{\hat y=Xw+b}
$$

where:

* \(X\) = input matrix
* \(w\) = weight vector
* \(b\) = bias
* \(\hat y\) = predictions

The gradient for the weights becomes:

$$
\boxed{
dw=
\frac{2}{n}X^T(\hat y-y)
}
$$

and the bias gradient remains:

$$
\boxed{
db=
\frac{2}{n}\sum(\hat y-y)
}
$$

This extension introduced **vectorized computation**, which is fundamental to how neural networks are implemented.

---

## 💻 Technologies

* Python
* NumPy

No machine-learning libraries were used.

The model, gradients, and training loop were implemented manually.

---

## 🧪 Key Implementation

The core vectorized forward pass is:

```python
y_pred = X @ w + b
```

The loss:

```python
loss = np.mean((y_pred - y) ** 2)
```

The gradients:

```python
dw = 2 * np.mean((y_pred - y) * X)
db = 2 * np.mean(y_pred - y)
```

For the multi-feature version:

```python
dw = (2 / len(X)) * X.T @ (y_pred - y)
db = (2 / len(X)) * np.sum(y_pred - y)
```

---

## 📚 What I Learned

* A neuron is fundamentally a dot product followed by a bias and activation.
* Matrices allow multiple examples and features to be processed simultaneously.
* Derivatives measure how the loss changes with respect to parameters.
* The chain rule allows gradients to be propagated through multiple operations.
* Gradients provide the information needed for parameter updates.
* The learning rate controls the size of each optimization step.
* Linear regression can be trained entirely from scratch using NumPy.
* Matrix multiplication makes multi-feature models possible without manually processing every feature.
* The same mathematical ideas used here form the foundation of neural-network training and backpropagation.

---

## 🎯 Day 2 Outcome

By the end of Day 2, I implemented and trained:

1. **Single-feature Linear Regression**
2. **Multi-feature Linear Regression**

from scratch using NumPy.

The most important takeaway was understanding the complete learning loop:

$$
\boxed{
\text{Prediction}
\rightarrow
\text{Loss}
\rightarrow
\text{Gradient}
\rightarrow
\text{Update}
\rightarrow
\text{Repeat}
}
$$

This provides the mathematical foundation for moving from simple linear models to actual neural networks.
