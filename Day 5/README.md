# Day 5 — Backpropagation Deep Dive

## Overview

This project is part of the **21-Day Deep Learning Sprint** and focuses on one of the most important and conceptually challenging ideas in neural networks: **backpropagation**.

The goal of this project is not to use a deep learning framework or rely on automatic differentiation. Instead, a single neuron is implemented completely from scratch using Python and NumPy.

The project starts with a simple computational graph:

$$
x \rightarrow z \rightarrow a \rightarrow L
$$

where:

$$
z = wx+b
$$

$$
a = \sigma(z)
$$

and:

$$
L = \frac{1}{2}(a-y)^2
$$

The gradients of the loss with respect to the model parameters are then derived manually using the **chain rule** and used to update the parameters through gradient descent.

The purpose is to understand what backpropagation is actually doing internally before moving toward multi-layer neural networks and matrix-based implementations.

---

## Learning Objectives

By completing this project, the following concepts should become clear:

* What a computational graph is
* How forward propagation works
* How a loss function measures prediction error
* What a derivative represents in the context of machine learning
* How the chain rule connects different operations
* How gradients are propagated backward through a computational graph
* How the gradients of weights and biases are calculated
* The difference between backpropagation and gradient descent
* How a neural network learns by repeatedly updating its parameters
* How the mathematical derivation maps directly to NumPy code

---

# 1. The Problem

We want to train a single neuron to perform a very simple binary classification task.

The dataset is:

```python
X = [1, 2, 3, 4, 5, 6]

y = [0, 0, 0, 1, 1, 1]
```

Conceptually, the desired behavior is:

```text
1 → 0
2 → 0
3 → 0
4 → 1
5 → 1
6 → 1
```

The model must learn this relationship using only one weight and one bias.

---

# 2. Model Architecture

The model consists of a single neuron.

The computational flow is:

```text
Input
  |
  v
z = wx + b
  |
  v
a = sigmoid(z)
  |
  v
Loss
```

There are only two trainable parameters:

$$
w
$$

and

$$
b
$$

The entire model can therefore be described mathematically as:

$$
\boxed{z=wx+b}
$$

followed by:

$$
\boxed{a=\sigma(z)}
$$

where the sigmoid function is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The output \(a\) represents the neuron's prediction.

---

# 3. Forward Propagation

Forward propagation is the process of moving from the input toward the output.

Given an input \(x\), weight \(w\), and bias \(b\), the neuron first calculates:

$$
z=wx+b
$$

This value is called the **pre-activation**.

The pre-activation is then passed through the sigmoid function:

$$
a=\sigma(z)
$$

The sigmoid converts the value into a number between 0 and 1.

For example:

$$
z=0
$$

produces:

$$
\sigma(0)=0.5
$$

while a large positive value approaches 1 and a large negative value approaches 0.

---

# 4. Loss Function

To determine how well the neuron is performing, a loss function is used.

This project uses:

$$
\boxed{
L=\frac12(a-y)^2
}
$$

where:

* \(a\) is the prediction
* \(y\) is the true label
* \(L\) is the loss

If the prediction is close to the target, the loss is small.

If the prediction is far from the target, the loss is larger.

For the complete dataset, the mean loss is calculated:

$$
L_{\text{mean}}
=
\frac{1}{n}
\sum_{i=1}^{n}
\frac12(a_i-y_i)^2
$$

---

# 5. Why Backpropagation Is Necessary

Calculating the loss tells us how wrong the model is.

However, the loss alone does not tell us how to improve the model.

We need to know:

$$
\frac{\partial L}{\partial w}
$$

and:

$$
\frac{\partial L}{\partial b}
$$

These gradients tell us how sensitive the loss is to changes in the weight and bias.

For example:

$$
\frac{\partial L}{\partial w}<0
$$

means that increasing \(w\) will locally tend to decrease the loss.

The magnitude of the gradient indicates how strongly the loss changes with respect to that parameter.

Backpropagation is the process used to calculate these gradients efficiently.

---

# 6. Computational Graph

The most important conceptual tool in this project is the computational graph.

The model can be represented as:

```text
             w
             |
             v
x --------> Multiply
             |
             v
             z
             |
             v
          Sigmoid
             |
             v
             a
             |
             v
            Loss
             ^
             |
             y
```

Mathematically:

$$
w,x \rightarrow z \rightarrow a \rightarrow L
$$

During the forward pass, information moves from left to right.

During backpropagation, gradient information moves from right to left.

```text
Forward:

x → z → a → L


Backward:

L → a → z → w,b
```

This is the fundamental idea behind backpropagation.

---

# 7. The Chain Rule

The key mathematical tool is the chain rule.

Suppose:

$$
w\rightarrow z\rightarrow a\rightarrow L
$$

We want to determine how changing \(w\) affects \(L\).

Since \(w\) does not directly appear in the loss, we follow the dependency chain:

$$
w\rightarrow z\rightarrow a\rightarrow L
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
}
$$

This is the core of backpropagation.

Rather than calculating the entire derivative at once, the problem is broken into simple local derivatives.

---

# 8. Deriving the Gradients

## 8.1 Derivative of the Loss

The loss is:

$$
L=\frac12(a-y)^2
$$

Differentiating with respect to \(a\):

$$
\boxed{
\frac{\partial L}{\partial a}=a-y
}
$$

In code:

```python
dL_da = a - y
```

---

## 8.2 Derivative of Sigmoid

The activation function is:

$$
a=\sigma(z)
$$

The derivative of sigmoid has a particularly useful form:

$$
\boxed{
\frac{\partial a}{\partial z}=a(1-a)
}
$$

In code:

```python
da_dz = a * (1 - a)
```

Using the already calculated activation \(a\) avoids having to recalculate the sigmoid.

---

## 8.3 Gradient With Respect to \(z\)

Applying the chain rule:

$$
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial z}
=
(a-y)a(1-a)
}
$$

In code:

```python
dL_dz = dL_da * da_dz
```

This quantity is especially important because it becomes the gradient that is propagated backward through the neuron.

---

# 9. Gradient With Respect to the Weight

The pre-activation is:

$$
z=wx+b
$$

Differentiating with respect to \(w\):

$$
\frac{\partial z}{\partial w}=x
$$

Using the chain rule:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial w}
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial z}x
}
$$

Since the project operates on multiple examples, the individual gradient contributions are averaged:

$$
\boxed{
dw=
\frac{1}{n}
\sum_{i=1}^{n}
\frac{\partial L_i}{\partial z_i}x_i
}
$$

In NumPy:

```python
dw = np.mean(dL_dz * X)
```

---

# 10. Gradient With Respect to the Bias

Again:

$$
z=wx+b
$$

Differentiating with respect to \(b\):

$$
\frac{\partial z}{\partial b}=1
$$

Therefore:

$$
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial z}
$$

For multiple examples:

$$
\boxed{
db=
\frac{1}{n}
\sum_{i=1}^{n}
\frac{\partial L_i}{\partial z_i}
}
$$

In code:

```python
db = np.mean(dL_dz)
```

---

# 11. Complete Backpropagation

The complete backward pass can therefore be summarized as:

$$
\boxed{
\frac{\partial L}{\partial a}=a-y
}
$$

$$
\boxed{
\frac{\partial a}{\partial z}=a(1-a)
}
$$

$$
\boxed{
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
}
$$

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial z}x
}
$$

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial z}
}
$$

In code:

```python
dL_da = a - y

da_dz = a * (1 - a)

dL_dz = dL_da * da_dz

dw = np.mean(dL_dz * X)

db = np.mean(dL_dz)
```

These lines are the mathematical derivation translated directly into NumPy.

---

# 12. Gradient Descent

Backpropagation calculates the gradients.

It does not update the parameters itself.

Parameter updates are performed using gradient descent.

For the weight:

$$
\boxed{
w_{\text{new}}
=
w-\eta\frac{\partial L}{\partial w}
}
$$

For the bias:

$$
\boxed{
b_{\text{new}}
=
b-\eta\frac{\partial L}{\partial b}
}
$$

where \(\eta\) is the learning rate.

In code:

```python
w -= learning_rate * dw
b -= learning_rate * db
```

This distinction is important:

```text
Backpropagation
      |
      v
Calculate gradients
      |
      v
Gradient Descent
      |
      v
Update parameters
```

Backpropagation tells the model **which direction to move**.

Gradient descent determines how the parameters are actually moved.

---

# 13. Complete Training Loop

The complete learning process follows this sequence:

```text
Initialize w and b
       |
       v
Forward pass
       |
       v
Calculate predictions
       |
       v
Calculate loss
       |
       v
Backpropagation
       |
       v
Calculate dw and db
       |
       v
Update w and b
       |
       v
Repeat
```

The process is repeated for many epochs.

As training progresses, the expectation is that the loss decreases and the predictions become more accurate.

---

# 14. Numerical Example

Consider:

$$
x=2
$$

$$
w=0.5
$$

$$
b=0
$$

$$
y=1
$$

### Forward pass

$$
z=wx+b
$$

$$
z=(0.5)(2)=1
$$

The sigmoid output is:

$$
a=\sigma(1)\approx0.7310586
$$

The loss is:

$$
L=\frac12(0.7310586-1)^2
$$

$$
L\approx0.036164
$$

### Backward pass

First:

$$
\frac{\partial L}{\partial a}
=
a-y
$$

$$
\approx-0.2689414
$$

Then:

$$
\frac{\partial a}{\partial z}
=
a(1-a)
$$

$$
\approx0.1966119
$$

Therefore:

$$
\frac{\partial L}{\partial z}
=
(-0.2689414)(0.1966119)
$$

$$
\approx-0.0528771
$$

For the weight:

$$
\frac{\partial z}{\partial w}=x=2
$$

so:

$$
\frac{\partial L}{\partial w}
=
(-0.0528771)(2)
$$

$$
\boxed{
\frac{\partial L}{\partial w}\approx-0.1057542
}
$$

For the bias:

$$
\frac{\partial z}{\partial b}=1
$$

therefore:

$$
\boxed{
\frac{\partial L}{\partial b}\approx-0.0528771
}
$$

This example demonstrates the entire backward pass numerically.

---

# 15. Implementation

The implementation uses only NumPy.

The core training loop is:

```python
for epoch in range(epochs):

    # Forward pass
    z = w * X + b
    a = sigmoid(z)

    # Loss
    L = np.mean(0.5 * (a - y) ** 2)

    # Backpropagation
    dL_da = a - y
    da_dz = a * (1 - a)
    dL_dz = dL_da * da_dz

    dw = np.mean(dL_dz * X)
    db = np.mean(dL_dz)

    # Gradient descent
    w -= learning_rate * dw
    b -= learning_rate * db
```

No automatic differentiation is involved.

Every gradient is explicitly calculated.

---

# 16. Experiments

The project also investigates the effect of different learning rates.

The following values can be tested:

```text
0.01
0.1
1.0
```

The learning rate determines the size of each parameter update.

A small learning rate generally results in smaller updates, while a larger learning rate produces larger updates.

The experiment demonstrates why the learning rate is an important hyperparameter and why parameter updates must be monitored rather than assumed to work automatically.

---

# 17. Loss Curve

The training process can be visualized by recording the loss after every epoch.

Conceptually, a successful training run should show:

```text
Loss
 ^
 |\
 | \
 |  \
 |   \
 |    \____
 |
 +--------------> Epochs
```

The exact shape depends on the initialization and learning rate.

The important observation is that the loss should generally decrease as the model learns.

---

# 18. Final Predictions

After training, predictions are generated using:

```python
z = w * X + b
probabilities = sigmoid(z)

predictions = (probabilities >= 0.5).astype(int)
```

The sigmoid output is converted into a binary class using a threshold of 0.5.

The final predictions can then be compared with the original labels.

For this simple dataset, the trained neuron should learn to distinguish the lower-valued inputs from the higher-valued inputs.

---

# 19. Key Concepts Learned

## Forward Propagation

Forward propagation calculates the prediction:

$$
x\rightarrow z\rightarrow a
$$

---

## Loss

The loss quantifies how far the prediction is from the target:

$$
L=\frac12(a-y)^2
$$

---

## Backpropagation

Backpropagation calculates how the loss changes with respect to the parameters.

$$
\frac{\partial L}{\partial w},
\qquad
\frac{\partial L}{\partial b}
$$

---

## Chain Rule

The chain rule connects the local derivatives:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
$$

---

## Gradient Descent

Gradient descent uses the calculated gradients to update the parameters:

$$
w=w-\eta dw
$$

$$
b=b-\eta db
$$

---

# 20. Backpropagation Mental Model

The most important takeaway from this project is the following:

> Backpropagation is not a special collection of neural-network formulas. It is repeated application of the chain rule through a computational graph.

For this neuron:

```text
Forward:

x
↓
z = wx + b
↓
a = sigmoid(z)
↓
Loss


Backward:

Loss
↓
dL/da
↓
dL/dz
↓
dw, db
```

At every operation, the general rule is:

$$
\boxed{
\text{gradient with respect to input}
=
\text{incoming gradient}
\times
\text{local derivative}
}
$$

This principle scales from a single neuron to large, multi-layer neural networks.

---

# 21. Why This Project Matters

Modern deep learning frameworks hide most of these calculations behind automatic differentiation.

For example, frameworks such as PyTorch can calculate gradients automatically.

However, understanding the underlying mechanism is essential for understanding:

* neural network training
* vanishing gradients
* exploding gradients
* activation functions
* optimization
* weight initialization
* computational graphs
* automatic differentiation
* deep neural network architectures

The single neuron implemented in this project represents the smallest useful example of the same mathematical process used to train much larger neural networks.

---

# 22. Project Structure

A suggested repository structure is:

```text
Day-5-Backpropagation/
│
├── README.md
│
├── Backpropagation_From_Scratch.ipynb
│
└── assets/
    └── loss_curve.png
```

The primary artifact is the Jupyter/Google Colab notebook containing the complete implementation and experiments.

---

# 23. Technologies Used

* Python
* NumPy
* Matplotlib
* Google Colab / Jupyter Notebook

No deep learning frameworks or automatic differentiation libraries are used.

---

# 24. Conclusion

This project builds backpropagation from first principles using a single neuron.

Starting from:

$$
z=wx+b
$$

and:

$$
a=\sigma(z)
$$

the project derives the gradients using the chain rule and uses them to optimize the model through gradient descent.

The most important conceptual progression is:

$$
\boxed{
\text{Computational Graph}
\rightarrow
\text{Chain Rule}
\rightarrow
\text{Gradients}
\rightarrow
\text{Backpropagation}
\rightarrow
\text{Gradient Descent}
}
$$

Once this process is understood for a single neuron, the next step is to extend the same reasoning to multiple neurons and hidden layers.

The mathematics does not fundamentally change. The main challenge becomes expressing the same chain-rule operations efficiently using vectors and matrices.
