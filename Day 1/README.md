# Day 1 — Deep Learning Foundations

## Overview

This project is the first stage of a 21-day Deep Learning learning roadmap focused on building a strong conceptual and practical understanding of Deep Learning.

Day 1 introduces the fundamental building blocks of neural networks and demonstrates how a model can learn parameters through gradient descent without relying on Deep Learning frameworks.

The implementation uses only:

* Python
* NumPy
* Matplotlib

No PyTorch, TensorFlow, Scikit-learn, or automatic differentiation libraries are used.

---

## Learning Objectives

By completing this project, you should understand:

* The relationship between Artificial Intelligence, Machine Learning, and Deep Learning
* The basic structure of a neural network
* What a neuron does
* The role of weights and biases
* Activation functions
* Forward propagation
* Loss functions
* Gradients
* Gradient descent
* Learning rate
* Parameter updates
* The basic neural network training loop
* How parameters are learned from data

---

## Core Concept

A basic neuron can be represented as:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

where:

* \(x_i\) is an input feature
* \(w_i\) is the corresponding weight
* \(b\) is the bias
* \(z\) is the weighted sum

An activation function can then be applied:

$$
a = f(z)
$$

For a simple regression problem, the activation function can be omitted, resulting in:

$$
\hat{y} = wx + b
$$

This project uses this simple model to demonstrate how gradient descent learns the parameters \(w\) and \(b\).

---

## Project

### Single-Neuron Linear Regression

The main implementation trains a single neuron to learn a simple linear relationship between an input feature and a target value.

The model is:

$$
\hat{y} = wx + b
$$

For the initial experiment, the training data follows approximately:

$$
y = 2x + 1
$$

The model starts with arbitrary parameters and gradually learns values close to:

$$
w = 2,\qquad b = 1
$$

---

## Training Process

The training process follows the fundamental Deep Learning workflow:

```text
Input Data
    |
    v
Forward Pass
    |
    v
Prediction
    |
    v
Loss Calculation
    |
    v
Gradient Calculation
    |
    v
Parameter Update
    |
    v
Repeat
```

### 1. Forward Pass

The model generates predictions using:

$$
\hat{y} = wx + b
$$

### 2. Loss Calculation

Mean Squared Error is used to measure prediction error:

$$
L = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

### 3. Gradient Calculation

The gradients describe how the loss changes with respect to the parameters.

For the weight:

$$
\frac{\partial L}{\partial w}
=
\frac{2}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)x_i
$$

For the bias:

$$
\frac{\partial L}{\partial b}
=
\frac{2}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)
$$

### 4. Parameter Update

Gradient descent updates the parameters using:

$$
w_{\text{new}}
=
w_{\text{old}}
-
\eta
\frac{\partial L}{\partial w}
$$

$$
b_{\text{new}}
=
b_{\text{old}}
-
\eta
\frac{\partial L}{\partial b}
$$

where \(\eta\) represents the learning rate.
<img width="460" height="375" alt="image" src="https://github.com/user-attachments/assets/047cc164-5292-4931-b745-17731803ac17" />


---

## Learning Rate Experiment

Different learning rates can significantly affect training.

### Small Learning Rate

A small learning rate produces smaller parameter updates.

Advantages:

* More gradual optimization
* Less risk of overshooting the minimum

Disadvantage:

* Training can be slow

### Large Learning Rate

A large learning rate produces larger parameter updates.

Advantages:

* Potentially faster convergence

Disadvantages:

* Can overshoot the minimum
* Can cause unstable training
* May prevent convergence

The project experiments with different learning rates to demonstrate this behavior.

---

## Project Structure

A recommended repository structure is:

```text
day-01-deep-learning-foundations/
│
├── README.md
├── neuron.py
├── requirements.txt
└── outputs/
    └── loss_curve.png
```

If the implementation is kept in a notebook:

```text
day-01-deep-learning-foundations/
│
├── README.md
├── day_01_neuron.ipynb
├── requirements.txt
└── outputs/
    └── loss_curve.png
```

---

## Requirements

Python 3.9 or later is recommended.

Install the required dependencies:

```bash
pip install numpy matplotlib
```

Alternatively, install them from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
matplotlib
```

---

## Running the Project

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to the Day 1 directory:

```bash
cd day-01-deep-learning-foundations
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the implementation:

```bash
python neuron.py
```

The program should train the neuron using gradient descent and display or save the training loss.

---

## Expected Results

After sufficient training, the learned parameters should approach the underlying relationship in the dataset.

For:

$$
y = 2x + 1
$$

the model should learn approximately:

```text
w ≈ 2
b ≈ 1
```

The training loss should decrease as the model learns.

A typical training process will look conceptually like:

```text
High Loss
   |
   |\
   | \
   |  \
   |   \
   |    \____
   |
   +----------------
        Epochs
```

The exact values will depend on the initialization, learning rate, dataset, and number of training iterations.

---

## Key Concepts Demonstrated

### Weights

Weights determine how strongly individual input features influence a neuron's output.

For:

$$
z = wx+b
$$

a larger magnitude of \(w\) means the input has a stronger influence on \(z\).

### Bias

The bias allows the model to shift its output independently of the input.

### Loss

Loss quantifies the difference between the model's prediction and the target.

### Gradient

A gradient describes how changing a parameter affects the loss.

### Gradient Descent

Gradient descent uses the gradients to adjust model parameters in a direction that reduces the loss.

### Learning Rate

The learning rate determines the magnitude of each parameter update.

---

## Extension Challenge

After completing the single-feature implementation, extend the model to use multiple input features.

For example:

```text
Input Features
--------------
Area
Bedrooms
```

The model becomes:

$$
\hat{y} = w_1x_1 + w_2x_2 + b
$$

This extension prepares the concepts needed for Day 2, where scalar operations will be generalized into vectors and matrices.

---

## What Was Built

This project implements a simple learning system from scratch using NumPy.

The implementation demonstrates:

1. Dataset creation
2. Parameter initialization
3. Forward propagation
4. Prediction generation
5. Mean Squared Error calculation
6. Manual gradient calculation
7. Gradient descent
8. Parameter updates
9. Iterative training
10. Loss monitoring
11. Learning-rate experimentation

---

## Learning Outcome

After completing Day 1, the learner should be able to explain the following process without relying on a Deep Learning framework:

```text
Input
  |
  v
Weighted Sum
  |
  v
Prediction
  |
  v
Loss
  |
  v
Gradient
  |
  v
Parameter Update
  |
  v
Repeat
```

The primary objective is not to build a production-ready regression model. The objective is to understand the mechanics that form the foundation of neural network training.

---

## Limitations

This implementation is intentionally simple.

It does not include:

* Multiple hidden layers
* Backpropagation through multiple layers
* Mini-batch training
* Regularization
* Advanced optimizers
* Automatic differentiation
* GPU acceleration
* Production data pipelines

These topics are introduced progressively throughout the roadmap.

---

## Roadmap Progress

```text
Day 01  Deep Learning Foundations       [Completed]
Day 02  Neural Network Mathematics      [Next]
Day 03  Logistic Regression
Day 04  Neural Network From Scratch
Day 05  Backpropagation Deep Dive
...
Day 21  Deep Learning Capstone
```

---

## Educational Philosophy

This project follows a build-first approach.

Instead of relying immediately on high-level frameworks, the fundamental mechanics are implemented manually to develop an understanding of what happens underneath libraries such as PyTorch.

The implementation should therefore be treated as an educational reference rather than a recommended approach for production machine learning systems.

