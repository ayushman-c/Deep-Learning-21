# Day 7 — Training Neural Networks

## Overview

Day 7 focuses on one of the most important processes in deep learning: **training a neural network**.

The previous days of the sprint established the mathematical and conceptual foundations of neural networks, backpropagation, and PyTorch. This day connects those concepts into a complete training pipeline.

The primary objective was to understand how a neural network learns from data through repeated cycles of:

$$
\text{Forward Pass}
\rightarrow
\text{Loss Calculation}
\rightarrow
\text{Backpropagation}
\rightarrow
\text{Parameter Update}
$$

The concepts were implemented using PyTorch rather than manually calculating gradients and updating parameters.

By the end of this day, the training process was no longer treated as a black box. Each component of the training pipeline was constructed and understood individually before being combined into a complete workflow.

---

# Learning Objectives

The main objectives of Day 7 were:

* Understand what neural network training actually means.
* Understand the complete training loop.
* Understand forward propagation during training.
* Understand loss functions and their role in optimization.
* Understand gradient accumulation in PyTorch.
* Understand the purpose of `optimizer.zero_grad()`.
* Understand what `loss.backward()` does.
* Understand the role of an optimizer.
* Understand `optimizer.step()`.
* Understand epochs, iterations, and batches.
* Understand mini-batch training.
* Use `TensorDataset` and `DataLoader`.
* Understand training and validation datasets.
* Understand `model.train()` and `model.eval()`.
* Understand `torch.no_grad()`.
* Understand learning rate.
* Understand SGD and Adam at a conceptual level.
* Understand overfitting and underfitting.
* Track training and validation loss.
* Build a complete neural network training pipeline.

---

# Connection to Previous Days

Day 7 builds directly on the previous six days.

The progression of the sprint is:

```text
Day 1
Deep Learning Fundamentals
        ↓
Day 2
Mathematical Foundations
        ↓
Day 3
Backpropagation
        ↓
Day 4
Neural Network From Scratch
        ↓
Day 5
Backpropagation Deep Dive
        ↓
Day 6
PyTorch Fundamentals
        ↓
Day 7
Training Neural Networks
```

The major transition on Day 7 is from understanding individual components to combining them into a complete learning system.

Previously, gradient calculations and parameter updates were studied mathematically and implemented manually.

Now PyTorch performs those operations automatically through:

```python
loss.backward()
```

and:

```python
optimizer.step()
```

---

# 1. What Is Neural Network Training?

A neural network contains trainable parameters such as weights and biases.

Let the parameters be represented by:

$$
\theta
$$

The neural network produces a prediction:

$$
\hat{y}=f(X;\theta)
$$

The prediction is compared against the actual target:

$$
y
$$

using a loss function:

$$
L(\hat{y},y)
$$

The objective of training is to find parameter values that minimize this loss.

The gradient of the loss with respect to the parameters is:

$$
\nabla_{\theta}L
$$

The parameters are then updated using an optimization algorithm.

For basic gradient descent:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla_{\theta}L
$$

where:

$$
\eta
$$

is the learning rate.

This process is repeated many times.

---

# 2. The Training Cycle

The fundamental training cycle is:

```text
Input Data
    ↓
Forward Pass
    ↓
Prediction
    ↓
Loss Calculation
    ↓
Gradient Calculation
    ↓
Parameter Update
    ↓
Repeat
```

In PyTorch, the core operations are:

```python
predictions = model(X_batch)

loss = loss_fn(predictions, y_batch)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

Each operation represents a specific stage of the learning process.

---

# 3. Forward Pass

The forward pass is the process of passing input data through the neural network.

For a network containing multiple layers, the input is transformed sequentially.

For example:

$$
X
\rightarrow
Linear
\rightarrow
ReLU
\rightarrow
Linear
\rightarrow
ReLU
\rightarrow
Linear
\rightarrow
\hat{y}
$$

In PyTorch, the forward pass occurs when the model is called:

```python
predictions = model(X_batch)
```

The network uses its current weights and biases to produce predictions.

At the beginning of training, these parameters are generally initialized to values that do not produce good predictions.

Training progressively modifies them.

---

# 4. Loss Function

A model needs a numerical measurement of how wrong its predictions are.

The loss function provides this measurement.

For the Day 7 project, the task was regression, so Mean Squared Error was used.

The MSE equation is:

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

In PyTorch:

```python
loss_fn = nn.MSELoss()
```

The loss is calculated using:

```python
loss = loss_fn(predictions, y_batch)
```

A smaller loss generally indicates that predictions are closer to the targets for the evaluated data.

---

# 5. Computational Graph and Backpropagation

During the forward pass, PyTorch tracks operations involving tensors that require gradients.

This creates a computational graph.

Conceptually:

```text
Input
  ↓
Linear Layer
  ↓
Activation
  ↓
Linear Layer
  ↓
Prediction
  ↓
Loss
```

When:

```python
loss.backward()
```

is called, PyTorch traverses the computational graph backward and calculates gradients for the trainable parameters.

For a parameter \(w\):

$$
\frac{\partial L}{\partial w}
$$

represents how the loss changes with respect to that parameter.

These gradients tell the optimizer which direction the parameters should move.

---

# 6. Why `optimizer.zero_grad()` Is Necessary

PyTorch accumulates gradients by default.

Suppose one backward pass produces:

$$
\frac{\partial L}{\partial w}=0.5
$$

and another produces:

$$
\frac{\partial L}{\partial w}=0.3
$$

Without clearing the gradient, PyTorch can accumulate them:

$$
0.5+0.3=0.8
$$

This is normally not what we want for independent training steps.

Therefore, before calculating the gradients for a new batch, we use:

```python
optimizer.zero_grad()
```

The standard sequence is:

```text
Clear old gradients
        ↓
Calculate new gradients
        ↓
Update parameters
```

---

# 7. Optimizers

The optimizer is responsible for updating the model's trainable parameters based on their gradients.

The project used Adam:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

The optimizer receives:

```python
model.parameters()
```

which contains the trainable weights and biases.

The learning rate determines the scale of the parameter updates.

Conceptually:

$$
\theta
\rightarrow
\theta_{\text{updated}}
$$

after each optimization step.

---

# 8. `optimizer.step()`

After gradients have been calculated:

```python
loss.backward()
```

the optimizer applies the update:

```python
optimizer.step()
```

This is the operation that actually changes the model's trainable parameters.

Without `optimizer.step()`, the model would calculate gradients but its parameters would not be updated.

The complete sequence is therefore:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

# 9. Epochs, Batches, and Iterations

Large datasets are generally not processed as one massive tensor during training.

Instead, the dataset is divided into batches.

For example:

```text
Dataset
  ↓
Batch 1
Batch 2
Batch 3
...
```

## Batch

A batch is a subset of the training dataset processed together.

The Day 7 project used:

```text
batch_size = 32
```

Therefore, each training iteration processes 32 samples.

## Iteration

One iteration generally corresponds to one batch being processed and one optimizer update being performed.

## Epoch

An epoch represents one complete pass through the training dataset.

For example, if there are 800 training examples and the batch size is 32:

$$
\frac{800}{32}=25
$$

Therefore, one epoch consists of 25 training iterations.

If training is performed for 100 epochs:

$$
25\times100=2500
$$

optimizer updates are performed.

---

# 10. `TensorDataset`

PyTorch provides `TensorDataset` for pairing tensors containing inputs and targets.

The project created:

```text
train_dataset
val_dataset
```

The training dataset contains:

```text
X_train
y_train
```

while the validation dataset contains:

```text
X_val
y_val
```

This maintains the correspondence between each input and its target.

Conceptually:

```text
X₁ → y₁
X₂ → y₂
X₃ → y₃
...
```

---

# 11. `DataLoader`

A `DataLoader` provides batches from a dataset.

The training loader used:

```text
batch_size = 32
shuffle = True
```

The validation loader used:

```text
batch_size = 32
shuffle = False
```

The training DataLoader allows the model to process mini-batches instead of the entire dataset at once.

A batch can be retrieved using:

```python
X_batch, y_batch = next(iter(train_loader))
```

The resulting shapes were:

```text
X_batch → (32, 2)
y_batch → (32, 1)
```

---

# 12. Why Shuffle Training Data?

The training dataset was configured with:

```text
shuffle=True
```

This changes the order of training examples between epochs.

Randomizing the training order prevents the model from repeatedly encountering examples in exactly the same sequence.

Validation data does not need to be shuffled because it is being used for evaluation rather than parameter updates.

---

# 13. Dataset Split

The synthetic dataset contained 1000 samples.

It was divided into:

```text
80% Training
20% Validation
```

Resulting in:

```text
Training:
800 samples

Validation:
200 samples
```

The split was performed before converting the arrays to PyTorch tensors.

The training set is used for learning model parameters.

The validation set is used to monitor how the model performs on data that was not used for parameter updates.

---

# 14. The Neural Network Architecture

The main project used the following architecture:

```text
Input
  ↓
2 neurons
  ↓
Linear
  ↓
32 neurons
  ↓
ReLU
  ↓
16 neurons
  ↓
ReLU
  ↓
1 neuron
  ↓
Output
```

In dimensional form:

$$
2\rightarrow32\rightarrow16\rightarrow1
$$

The model was implemented as a custom `nn.Module`.

The network used two hidden layers with ReLU activation functions.

The final layer produced a single continuous value because this was a regression problem.

---

# 15. Why ReLU?

The hidden layers use ReLU:

$$
ReLU(x)=\max(0,x)
$$

ReLU introduces nonlinearity into the network.

Without nonlinear activation functions, stacking multiple linear layers would still result in an overall linear transformation.

Because the synthetic dataset was intentionally generated using a nonlinear relationship, nonlinear activation functions were required for the network to model the relationship effectively.

---

# 16. Training Loop

The core training loop follows this structure:

```text
For each epoch:

    For each batch:

        Forward pass
        Calculate loss
        Clear gradients
        Backpropagate
        Update parameters

    Record epoch loss
```

The fundamental operations are:

```python
predictions = model(X_batch)

loss = loss_fn(predictions, y_batch)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

This sequence is the central implementation of Day 7.

---

# 17. Tracking Training Loss

Initially, the loss from only the final batch of each epoch was being stored.

This is not an ideal representation of epoch-level performance because different batches can have slightly different losses.

Instead, the losses from all batches in an epoch should be accumulated and averaged.

Conceptually:

$$
L_{\text{epoch}}
=
\frac{1}{N}
\sum_{i=1}^{N}
L_i
$$

where \(N\) is the number of batches.

This produces a more stable representation of training performance.

The numerical value can be extracted using:

```python
loss.item()
```

This converts a scalar tensor into a Python number that can be stored for logging and plotting.

---

# 18. Training Mode and Evaluation Mode

PyTorch models have two important modes.

## Training Mode

```python
model.train()
```

This tells the model that it is being trained.

Certain layers, such as Dropout and Batch Normalization, behave differently during training.

## Evaluation Mode

```python
model.eval()
```

This tells the model that it is being evaluated.

This distinction becomes important as model architectures become more sophisticated.

---

# 19. `torch.no_grad()`

During validation or inference, gradients are generally unnecessary.

Therefore:

```python
with torch.no_grad():
```

can be used.

This prevents unnecessary gradient tracking.

A validation process conceptually looks like:

```text
Set model to evaluation mode
        ↓
Disable gradient tracking
        ↓
Forward pass
        ↓
Calculate validation loss
```

No parameter updates are performed during validation.

---

# 20. Training vs Validation

Training and validation serve different purposes.

## Training

The model's parameters are updated.

```text
Training Data
    ↓
Prediction
    ↓
Loss
    ↓
Backpropagation
    ↓
Parameter Update
```

## Validation

The model's parameters are not updated.

```text
Validation Data
    ↓
Prediction
    ↓
Loss
    ↓
Performance Measurement
```

This distinction is essential for understanding generalization.

---

# 21. Overfitting

A model can become very good at reproducing the training data while becoming worse at handling unseen data.

This is called overfitting.

A common pattern is:

```text
Training loss
      ↓↓↓↓↓↓↓

Validation loss
      ↓↓↓↑↑↑
```

The training loss continues decreasing while validation loss begins increasing.

This indicates that improvements on the training set are no longer translating into improvements on unseen data.

---

# 22. Underfitting

Underfitting occurs when the model is unable to adequately learn the underlying relationship.

A typical pattern is:

```text
Training loss    → high
Validation loss  → high
```

Potential causes include:

* Insufficient model capacity
* Insufficient training
* Poor feature representation
* Inappropriate learning rate
* Excessive regularization

---

# 23. Learning Rate

The learning rate controls the scale of parameter updates.

For example:

```python
lr=0.001
```

A learning rate that is too small can make training extremely slow.

A learning rate that is too large can cause unstable optimization.

Conceptually:

```text
Very small learning rate
→ slow learning

Very large learning rate
→ unstable updates

Appropriate learning rate
→ useful optimization
```

Learning rate is therefore one of the most important hyperparameters in neural network training.

---

# 24. Adam

Adam was used as the optimizer for the main project.

Adam is an adaptive optimization algorithm that uses information from gradients over time to adjust parameter updates.

The implementation is:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

The goal at this stage was not to memorize the complete Adam derivation, but to understand its position in the training pipeline:

```text
Loss
  ↓
Gradients
  ↓
Adam
  ↓
Parameter Updates
```

---

# 25. SGD

Stochastic Gradient Descent is the fundamental optimization algorithm behind much of the training theory studied earlier in the sprint.

Its basic update rule is:

$$
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta L
$$

PyTorch provides SGD through:

```python
torch.optim.SGD
```

SGD is conceptually important because it directly connects the mathematical gradient descent equations from earlier days with practical neural network training.

---

# 26. Synthetic Dataset

The project used a synthetic nonlinear regression dataset.

The inputs contained two features:

$$
X\in\mathbb{R}^{1000\times2}
$$

The target contained one continuous value per example:

$$
y\in\mathbb{R}^{1000\times1}
$$

Noise was added to the target so that the neural network was not simply memorizing a perfectly deterministic mathematical function.

This created a controlled environment for studying training behavior.

---

# 27. Project Pipeline

The complete project pipeline is:

```text
Generate Synthetic Data
        ↓
Inspect Dataset
        ↓
Train / Validation Split
        ↓
Convert NumPy → PyTorch
        ↓
Create TensorDataset
        ↓
Create DataLoader
        ↓
Build Neural Network
        ↓
Define Loss Function
        ↓
Define Optimizer
        ↓
Training Loop
        ↓
Validation Loop
        ↓
Track Training Loss
        ↓
Track Validation Loss
        ↓
Plot Loss Curves
        ↓
Evaluate Model
        ↓
Analyze Generalization
```

This pipeline represents the basic structure of many supervised deep learning projects.

---

# 28. Important PyTorch Concepts Learned

The following PyTorch concepts were reinforced during Day 7:

| Concept                 | Purpose                              |
| ----------------------- | ------------------------------------ |
| `nn.Module`             | Base class for neural network models |
| `nn.Sequential`         | Sequentially combines layers         |
| `nn.Linear`             | Fully connected layer                |
| `nn.ReLU`               | Nonlinear activation                 |
| `nn.MSELoss`            | Regression loss                      |
| `TensorDataset`         | Pairs inputs and targets             |
| `DataLoader`            | Provides mini-batches                |
| `model.train()`         | Enables training behavior            |
| `model.eval()`          | Enables evaluation behavior          |
| `torch.no_grad()`       | Disables gradient tracking           |
| `loss.backward()`       | Calculates gradients                 |
| `optimizer.zero_grad()` | Clears accumulated gradients         |
| `optimizer.step()`      | Updates parameters                   |
| `loss.item()`           | Extracts a scalar value              |

---

# 29. Most Important Conceptual Distinction

One of the most important lessons from Day 7 is the distinction between:

### Forward propagation

Answers:

> What does the model predict?

### Loss function

Answers:

> How wrong is the prediction?

### Backpropagation

Answers:

> How should each parameter change to reduce the loss?

### Optimizer

Performs:

> The actual parameter update.

Together:

$$
\boxed{
\text{Forward}
\rightarrow
\text{Loss}
\rightarrow
\text{Backward}
\rightarrow
\text{Update}
}
$$

---

# 30. Common Mistakes

Several mistakes are particularly common when implementing training loops.

## Forgetting `zero_grad()`

Incorrect:

```python
loss.backward()
optimizer.step()
```

Correct:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Updating During Validation

Validation should not modify model parameters.

Do not call:

```python
loss.backward()
optimizer.step()
```

during validation.

---

## Forgetting `model.eval()`

Evaluation should generally use:

```python
model.eval()
```

before calculating validation or test predictions.

---

## Tracking Only the Final Batch Loss

Recording only the final batch's loss can give a noisy representation of the epoch.

Instead, aggregate losses across batches.

---

## Using the Test Set for Training Decisions

The test set should remain separate from training decisions.

The validation set is used during development to evaluate generalization and make modeling decisions.

---

# 31. Key Questions to Be Able to Answer

After completing Day 7, you should be able to answer:

### What is an epoch?

One complete pass through the training dataset.

### What is a batch?

A subset of training examples processed together.

### What is an iteration?

One training update, generally corresponding to one processed batch.

### Why do gradients need to be cleared?

Because PyTorch accumulates gradients.

### What does `loss.backward()` do?

It calculates gradients of the loss with respect to trainable parameters.

### What does `optimizer.step()` do?

It updates the parameters using their gradients.

### What is the learning rate?

A hyperparameter controlling the scale of parameter updates.

### Why use a validation set?

To evaluate how the model performs on data not used to update its parameters.

### What is overfitting?

When training performance continues improving while performance on unseen data deteriorates.

### Why use `torch.no_grad()` during validation?

Because gradients are unnecessary when parameters are not being updated.

---

# 32. Final Mental Model

The most important mental model from this day is:

```text
                 Training Data
                      ↓
                ┌───────────┐
                │   Model   │
                └─────┬─────┘
                      ↓
                 Prediction
                      ↓
                ┌───────────┐
                │    Loss   │
                └─────┬─────┘
                      ↓
                Backpropagation
                      ↓
                  Gradients
                      ↓
                ┌───────────┐
                │ Optimizer │
                └─────┬─────┘
                      ↓
              Updated Parameters
                      ↓
                 Next Batch
                      ↓
                 Next Epoch
```

The model repeatedly moves through this cycle until its parameters produce sufficiently useful predictions.

---

# 33. Day 7 Outcome

By completing this project, the neural network is no longer just an architecture defined using PyTorch layers.

It is now a trainable machine learning system.

The complete progression is:

```text
Data
  ↓
Model
  ↓
Prediction
  ↓
Loss
  ↓
Gradients
  ↓
Optimizer
  ↓
Updated Parameters
  ↓
Repeated Training
  ↓
Validation
  ↓
Generalization Analysis
```

This is the fundamental workflow that underlies supervised neural network training.

Day 7 therefore serves as the transition from **PyTorch fundamentals** to practical **deep learning model development**.
