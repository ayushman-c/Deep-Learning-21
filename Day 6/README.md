# Day 6 — PyTorch Fundamentals

## Overview

Day 6 of the `ML-21_DAY_SPRINT` marks the transition from implementing neural networks manually to using **PyTorch**, one of the most widely used deep learning frameworks.

During the previous days, the mathematical foundations of neural networks were implemented from scratch. Matrix multiplication, activation functions, loss functions, gradients, and backpropagation were explicitly calculated and implemented.

The purpose of this day is to understand how those same concepts are represented and automated in PyTorch.

The objective is not to treat PyTorch as a collection of functions to memorize. Instead, the goal is to understand the relationship between the mathematical operations already learned and the abstractions provided by the framework.

The central idea throughout this project is:

```text
Tensor
   ↓
Model
   ↓
Forward Pass
   ↓
Loss
   ↓
Autograd
   ↓
Gradients
   ↓
Optimizer
   ↓
Updated Parameters
```

By the end of this day, a complete neural-network training pipeline can be implemented using PyTorch.

---

## Learning Objectives

By completing this project, the following concepts should be understood:

* What PyTorch is and why it is used for deep learning
* PyTorch tensors and their properties
* Tensor shapes and dimensions
* Tensor data types
* Tensor creation and initialization
* Tensor indexing and slicing
* Tensor reshaping
* Element-wise operations
* Matrix multiplication
* Automatic differentiation
* Computational graphs
* `requires_grad`
* `.backward()`
* `.grad`
* `nn.Module`
* `nn.Linear`
* Activation functions
* Loss functions
* Optimizers
* The PyTorch training loop
* `Dataset`
* `DataLoader`
* Mini-batch training
* Train, validation, and test datasets
* Training and evaluation modes
* `torch.no_grad()`
* CPU and GPU execution
* Model evaluation
* Classification metrics
* Saving and loading model parameters
* Building an end-to-end binary classification model

---

# 1. Why PyTorch?

A neural network can be implemented entirely from scratch using NumPy and explicit mathematical derivations.

For example, a simple neural network may perform:

$$
Z = XW + b
$$

followed by an activation function:

$$
A = f(Z)
$$

The loss is then calculated:

$$
L = \mathcal{L}(A,y)
$$

Backpropagation calculates:

$$
\frac{\partial L}{\partial W}
$$

and:

$$
\frac{\partial L}{\partial b}
$$

The parameters are then updated using gradient descent:

$$
W := W-\eta\frac{\partial L}{\partial W}
$$

This approach is extremely valuable for understanding the mathematics, but manually implementing every operation becomes impractical for larger networks.

PyTorch provides abstractions for these operations.

For example:

| Mathematical Concept      | PyTorch                       |
| ------------------------- | ----------------------------- |
| Matrix/tensor computation | `torch.Tensor`                |
| Linear transformation     | `nn.Linear`                   |
| Neural network            | `nn.Module`                   |
| Activation function       | `torch.relu`, `torch.sigmoid` |
| Loss                      | `nn.*Loss`                    |
| Backpropagation           | `loss.backward()`             |
| Gradients                 | `.grad`                       |
| Parameter update          | `optimizer.step()`            |
| Gradient reset            | `optimizer.zero_grad()`       |
| GPU execution             | `.to(device)`                 |

PyTorch therefore does not replace the mathematics learned previously. It provides an efficient framework for implementing that mathematics.

---

# 2. Project Structure

The project is centered around a Google Colab notebook.

A typical structure is:

```text
Day-6/
│
├── Day_6_PyTorch_Fundamentals.ipynb
├── README.md
└── model.pth
```

The notebook is organized progressively so that each PyTorch concept is introduced before it is used in the final project.

---

# 3. PyTorch Fundamentals

## 3.1 Importing PyTorch

The notebook begins by importing the core PyTorch library along with the neural-network and optimization modules.

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

`torch` provides tensor operations and the core PyTorch functionality.

`torch.nn` provides neural-network building blocks.

`torch.optim` provides optimization algorithms such as SGD and Adam.

---

# 4. Tensors

The most fundamental data structure in PyTorch is the tensor.

A tensor is a multidimensional numerical structure that can represent:

* Scalars
* Vectors
* Matrices
* Higher-dimensional data

Examples include:

```text
Scalar      → 0 dimensions
Vector      → 1 dimension
Matrix      → 2 dimensions
Image batch → 4 dimensions
```

A tensor can be created from a Python list:

```python
x = torch.tensor([1, 2, 3])
```

Its shape can be inspected using:

```python
x.shape
```

---

# 5. Tensor Shapes

Understanding tensor shapes is one of the most important practical skills in PyTorch.

Consider:

```python
X = torch.randn(100, 5)
```

The shape is:

```text
[100, 5]
```

This can be interpreted as:

```text
100 samples
5 features per sample
```

If the weight matrix has shape:

```text
[5, 3]
```

then:

$$
(100\times5)(5\times3)
$$

produces:

$$
100\times3
$$

This shape reasoning is fundamental to understanding neural-network architectures.

---

# 6. Tensor Data Types

PyTorch tensors have a data type.

For example:

```python
x = torch.tensor([1, 2, 3])
```

may produce an integer tensor.

Neural-network computations generally use floating-point values:

```python
x = torch.tensor(
    [1, 2, 3],
    dtype=torch.float32
)
```

The notebook demonstrates how to inspect and control tensor data types.

---

# 7. Tensor Creation

Several methods for creating tensors are explored.

### Zeros

```python
torch.zeros(3, 4)
```

Creates a tensor filled with zeros.

### Ones

```python
torch.ones(3, 4)
```

Creates a tensor filled with ones.

### Uniform random values

```python
torch.rand(3, 4)
```

Generates random values between 0 and 1.

### Standard normal values

```python
torch.randn(3, 4)
```

Generates values approximately from a standard normal distribution.

Random initialization is particularly important for neural-network parameters.

---

# 8. Tensor Operations

The notebook demonstrates standard mathematical operations.

Element-wise addition:

```python
a + b
```

Element-wise subtraction:

```python
a - b
```

Element-wise multiplication:

```python
a * b
```

Element-wise division:

```python
a / b
```

These operations should not be confused with matrix multiplication.

---

# 9. Matrix Multiplication

Matrix multiplication is central to neural networks.

PyTorch provides:

```python
A @ B
```

and:

```python
torch.matmul(A, B)
```

For example:

```text
X: [100, 5]
W: [5, 3]

X @ W

Result: [100, 3]
```

This corresponds directly to the linear transformation performed inside dense neural-network layers.

---

# 10. Reshaping

Tensor shapes often need to be changed during preprocessing or model construction.

For example:

```python
x = x.reshape(2, 3)
```

The total number of elements must remain the same.

A tensor containing six values can therefore be reshaped from:

```text
[3, 2]
```

to:

```text
[2, 3]
```

but not to:

```text
[4, 2]
```

because the number of elements would change.

---

# 11. Autograd

One of the most important features of PyTorch is **automatic differentiation**.

Consider:

$$
y=x^2
$$

The derivative is:

$$
\frac{dy}{dx}=2x
$$

PyTorch can calculate this automatically.

```python
x = torch.tensor(
    3.0,
    requires_grad=True
)

y = x ** 2

y.backward()

print(x.grad)
```

The result is:

```text
6
```

because:

$$
2(3)=6
$$

---

# 12. `requires_grad`

When a tensor is created using:

```python
requires_grad=True
```

PyTorch tracks operations involving that tensor.

This allows PyTorch to construct a computational graph and later calculate derivatives.

This mechanism is essential for trainable neural-network parameters.

---

# 13. Computational Graph

During the forward pass, PyTorch tracks operations that involve tensors requiring gradients.

Conceptually:

```text
x
↓
operation
↓
y
↓
loss
```

When:

```python
loss.backward()
```

is executed, PyTorch traverses the computation graph backward and applies the chain rule.

This is the automated version of the backpropagation process implemented manually during previous days.

---

# 14. Gradients

After:

```python
loss.backward()
```

gradients are available through `.grad` for the relevant leaf tensors.

For a parameter:

```python
W.grad
```

represents:

$$
\frac{\partial L}{\partial W}
$$

Similarly:

```python
b.grad
```

represents:

$$
\frac{\partial L}{\partial b}
$$

This provides a direct connection between the mathematical notation and PyTorch's implementation.

---

# 15. Manual Neural-Network Calculation

Before using high-level PyTorch modules, the notebook demonstrates the neural-network calculation directly.

The basic transformation is:

$$
Z=XW+b
$$

For binary classification, the result can then be passed through a sigmoid:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

A loss can then be calculated.

Finally:

```python
loss.backward()
```

computes the required gradients.

This section is important because it demonstrates that PyTorch is not hiding the mathematics; it is automating the derivative calculation.

---

# 16. `nn.Module`

PyTorch uses `nn.Module` as the foundation for neural-network models.

A typical model is defined as:

```python
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):

        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)

        return x
```

There are two important components.

### `__init__`

Defines the layers and trainable components.

### `forward`

Defines how input data flows through those layers.

---

# 17. `nn.Linear`

The `nn.Linear` layer represents a learnable linear or affine transformation.

```python
nn.Linear(2, 4)
```

means:

```text
2 input features
4 output neurons
```

Conceptually:

$$
Z=XW+b
$$

The layer contains learnable weights and biases.

These parameters can be inspected:

```python
model.layer1.weight
```

and:

```python
model.layer1.bias
```

---

# 18. Parameter Counting

PyTorch allows all model parameters to be accessed through:

```python
model.parameters()
```

The number of trainable parameters can be calculated using:

```python
total_params = sum(
    p.numel()
    for p in model.parameters()
)
```

Parameter counting becomes increasingly important as models become larger.

---

# 19. Activation Functions

Activation functions introduce nonlinearity into neural networks.

The notebook introduces:

### ReLU

$$
ReLU(x)=\max(0,x)
$$

Implemented using:

```python
torch.relu(x)
```

### Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Implemented using:

```python
torch.sigmoid(x)
```

For binary classification, sigmoid is commonly used to convert logits into probabilities during inference.

---

# 20. Loss Functions

Different machine-learning problems require different loss functions.

## Binary Classification

```python
nn.BCEWithLogitsLoss()
```

This combines sigmoid and binary cross-entropy in a numerically stable implementation.

When using this loss, the model should generally return raw logits.

## Regression

```python
nn.MSELoss()
```

Mean squared error is commonly used for regression.

## Multiclass Classification

```python
nn.CrossEntropyLoss()
```

Cross-entropy loss is commonly used when predicting one class from multiple classes.

---

# 21. Optimizers

Once gradients have been calculated, model parameters must be updated.

PyTorch provides optimization algorithms for this.

## SGD

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)
```

The underlying idea is:

$$
\theta=\theta-\eta\nabla_\theta L
$$

where:

* \(\theta\) represents model parameters
* \(\eta\) is the learning rate
* \(\nabla_\theta L\) is the gradient

## Adam

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Adam uses adaptive estimates based on gradients and is widely used in deep learning.

---

# 22. The Training Loop

The PyTorch training loop is one of the most important patterns in this project.

```python
for epoch in range(epochs):

    predictions = model(X)

    loss = loss_fn(predictions, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

The five stages are:

```text
1. Forward pass
2. Calculate loss
3. Clear old gradients
4. Backpropagation
5. Update parameters
```

---

# 23. `optimizer.zero_grad()`

PyTorch accumulates gradients by default.

Therefore, gradients from previous iterations need to be cleared.

```python
optimizer.zero_grad()
```

This ensures that the current update is based on the current batch's gradients rather than an accumulation of previous gradients.

---

# 24. `loss.backward()`

This performs automatic differentiation.

Conceptually, it calculates:

$$
\frac{\partial L}{\partial \theta}
$$

for the trainable parameters \(\theta\).

This is the PyTorch implementation of the backpropagation process studied previously.

---

# 25. `optimizer.step()`

This updates the model parameters.

Conceptually:

$$
\theta\leftarrow\theta-\eta\nabla_\theta L
$$

The exact update depends on the optimizer being used.

---

# 26. Dataset and DataLoader

Real-world datasets are generally too large or inefficient to process as a single tensor during every training iteration.

PyTorch provides:

```python
TensorDataset
```

and:

```python
DataLoader
```

A dataset can be created using:

```python
dataset = TensorDataset(X, y)
```

A DataLoader can then be created:

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

The DataLoader handles batching and iteration.

---

# 27. Mini-Batch Training

Instead of training on the entire dataset at once:

```text
Entire dataset
      ↓
Model
```

training can use:

```text
Batch 1 → Model
Batch 2 → Model
Batch 3 → Model
...
```

A training loop becomes:

```python
for X_batch, y_batch in loader:

    predictions = model(X_batch)

    loss = loss_fn(
        predictions,
        y_batch
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

This approach scales much better to large datasets.

---

# 28. Train, Validation, and Test Data

A machine-learning project should separate data according to purpose.

### Training Set

Used to learn model parameters.

### Validation Set

Used during development for model selection and tuning decisions.

### Test Set

Used for final evaluation after development decisions have been made.

Conceptually:

```text
Training
    ↓
Learn parameters

Validation
    ↓
Make development decisions

Test
    ↓
Final evaluation
```

The notebook demonstrates how datasets can be divided using PyTorch utilities.

---

# 29. Training Mode and Evaluation Mode

PyTorch models have two important modes.

During training:

```python
model.train()
```

During evaluation:

```python
model.eval()
```

This matters for layers such as:

* Dropout
* Batch Normalization

These layers can behave differently during training and inference.

---

# 30. `torch.no_grad()`

During inference, gradients are generally unnecessary.

Therefore:

```python
model.eval()

with torch.no_grad():

    predictions = model(X)
```

This prevents unnecessary gradient tracking.

It reduces memory consumption and computational overhead.

---

# 31. CPU and GPU

One of PyTorch's major advantages is the ability to use hardware accelerators.

A device can be selected using:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

The model can then be moved:

```python
model = model.to(device)
```

Data must also be moved:

```python
X = X.to(device)
y = y.to(device)
```

The model and tensors involved in a computation must be on compatible devices.

---

# 32. Evaluation

After training, the model must be evaluated.

For binary classification, logits can be converted into probabilities:

```python
probabilities = torch.sigmoid(logits)
```

A threshold can then be applied:

```python
predictions = (
    probabilities >= 0.5
).float()
```

The resulting predictions can be compared with the true labels.

---

# 33. Classification Metrics

The project introduces several important metrics.

## Accuracy

Measures the fraction of predictions that are correct.

$$
Accuracy=
\frac{Correct\ Predictions}
{Total\ Predictions}
$$

## Precision

Measures how many predicted positives are actually positive.

$$
Precision=
\frac{TP}{TP+FP}
$$

## Recall

Measures how many actual positives were detected.

$$
Recall=
\frac{TP}{TP+FN}
$$

## F1 Score

The harmonic mean of precision and recall.

$$
F1=
2\frac{Precision\cdot Recall}
{Precision+Recall}
$$

## Confusion Matrix

A confusion matrix organizes predictions into:

```text
True Positive
True Negative
False Positive
False Negative
```

These metrics provide more information than loss alone.

---

# 34. Saving a Model

A trained PyTorch model can be saved using its state dictionary.

```python
torch.save(
    model.state_dict(),
    "model.pth"
)
```

The `state_dict` contains the learned parameters.

---

# 35. Loading a Model

The architecture must first be recreated.

```python
model = NeuralNetwork()
```

Then the learned parameters can be loaded:

```python
model.load_state_dict(
    torch.load("model.pth")
)
```

For inference:

```python
model.eval()
```

This allows the trained model to be used again without retraining.

---

# 36. Final Project

The final notebook combines all the concepts into one complete binary classification pipeline.

The workflow is:

```text
Dataset
   ↓
Data Preparation
   ↓
Train / Validation / Test Split
   ↓
Tensor Conversion
   ↓
TensorDataset
   ↓
DataLoader
   ↓
Neural Network
   ↓
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Parameter Updates
   ↓
Validation
   ↓
Test Evaluation
   ↓
Metrics
   ↓
Save Model
   ↓
Load Model
   ↓
Inference
```

This is the first point in the sprint where the entire neural-network workflow is expressed using a modern deep-learning framework.

---

# 37. Connection to Previous Days

Day 6 is directly connected to the work completed earlier in the sprint.

## Days 2-3

You learned the mathematical foundation:

$$
XW+b
$$

and:

$$
\sigma(z)
$$

along with gradient-based optimization.

## Day 4

You implemented a neural network from scratch.

You explicitly controlled:

* Parameters
* Forward propagation
* Loss
* Gradients
* Parameter updates

## Day 5

You studied backpropagation in greater depth and understood how the chain rule allows gradients to flow backward through a neural network.

## Day 6

PyTorch now automates much of that process.

The mapping is:

| From Scratch          | PyTorch                         |
| --------------------- | ------------------------------- |
| Weight matrix         | Model parameter                 |
| Bias                  | Model parameter                 |
| Matrix multiplication | Tensor operations / `nn.Linear` |
| Activation            | `torch.relu`, `torch.sigmoid`   |
| Loss calculation      | Loss modules                    |
| Manual derivative     | Autograd                        |
| Backpropagation       | `loss.backward()`               |
| Gradient storage      | `.grad`                         |
| Gradient descent      | Optimizer                       |
| Parameter update      | `optimizer.step()`              |

The important lesson is that the framework is an implementation layer over the mathematical ideas already learned.

---

# 38. Common Errors

Several errors are especially common when beginning PyTorch.

## Shape mismatch

Always inspect:

```python
X.shape
y.shape
model(X).shape
```

when debugging.

## Forgetting `zero_grad()`

Gradients accumulate by default.

Use:

```python
optimizer.zero_grad()
```

before calculating the new backward pass.

## Incorrect use of sigmoid

When using:

```python
nn.BCEWithLogitsLoss()
```

do not normally apply sigmoid to the model output before calculating the loss.

Use sigmoid when converting logits to probabilities for interpretation.

## Forgetting evaluation mode

Before inference:

```python
model.eval()
```

## Forgetting `torch.no_grad()`

During inference:

```python
with torch.no_grad():
```

## Device mismatch

Check:

```python
next(model.parameters()).device
```

and:

```python
X.device
```

They should be compatible.

## Confusing multiplication operations

Remember:

```python
a * b
```

is element-wise multiplication.

Whereas:

```python
A @ B
```

is matrix multiplication.

---

# 39. What This Project Teaches

The main purpose of this project is not to memorize PyTorch syntax.

It is to develop the ability to translate between three levels of understanding.

### Level 1: Mathematics

$$
Z=XW+b
$$

$$
L=\mathcal{L}(Z,y)
$$

$$
\nabla_\theta L
$$

$$
\theta\leftarrow\theta-\eta\nabla_\theta L
$$

### Level 2: PyTorch

```python
logits = model(X)

loss = loss_fn(
    logits,
    y
)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

### Level 3: Neural-network behavior

```text
Input
  ↓
Forward computation
  ↓
Prediction
  ↓
Error
  ↓
Gradient calculation
  ↓
Parameter update
  ↓
Improved model
```

Being able to move between these three representations is one of the main goals of Day 6.

---

# 40. Day 6 Completion Checklist

Before considering Day 6 complete, you should be able to answer "yes" to the following:

* [ ] I understand what a PyTorch tensor is.
* [ ] I can create tensors using different methods.
* [ ] I can inspect tensor shape and dimensions.
* [ ] I understand tensor dtypes.
* [ ] I can index and slice tensors.
* [ ] I can reshape tensors.
* [ ] I understand element-wise multiplication.
* [ ] I understand matrix multiplication.
* [ ] I can reason about matrix shapes.
* [ ] I understand `requires_grad`.
* [ ] I understand computational graphs.
* [ ] I understand `.backward()`.
* [ ] I understand `.grad`.
* [ ] I can create a custom `nn.Module`.
* [ ] I understand `nn.Linear`.
* [ ] I can inspect model parameters.
* [ ] I understand common activation functions.
* [ ] I understand the major basic loss functions.
* [ ] I understand SGD and Adam at a basic level.
* [ ] I can write a complete PyTorch training loop.
* [ ] I understand why `zero_grad()` is required.
* [ ] I understand what `optimizer.step()` does.
* [ ] I can use `TensorDataset`.
* [ ] I can use `DataLoader`.
* [ ] I understand mini-batch training.
* [ ] I understand training, validation, and test sets.
* [ ] I understand `model.train()`.
* [ ] I understand `model.eval()`.
* [ ] I understand `torch.no_grad()`.
* [ ] I can move a model and tensors to a device.
* [ ] I can evaluate a classification model.
* [ ] I can calculate basic classification metrics.
* [ ] I can save a model's `state_dict`.
* [ ] I can load a saved model.
* [ ] I can explain how PyTorch relates to the backpropagation implementation from previous days.

---

# Conclusion

Day 6 represents an important transition in the `ML-21_DAY_SPRINT`.

The previous days focused heavily on understanding what happens inside a neural network by implementing the mathematics manually. That foundation is essential because it prevents PyTorch from becoming a black box.

PyTorch now provides the tools required to implement those concepts at a practical scale.

The core workflow can be summarized as:

```text
Tensor
   ↓
Neural Network
   ↓
Forward Pass
   ↓
Loss
   ↓
Autograd
   ↓
Gradients
   ↓
Optimizer
   ↓
Updated Parameters
   ↓
Repeat
```

The most important takeaway is that the framework has changed, but the underlying mathematics has not.

The matrix operations are still matrix operations.

The loss is still a mathematical function.

Backpropagation is still the chain rule.

Gradient descent is still parameter optimization.

PyTorch simply provides an efficient, scalable, and practical way to implement these concepts.

This foundation will be used heavily in the upcoming days as the sprint moves from basic neural networks toward more advanced deep-learning architectures and workflows.
