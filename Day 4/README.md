# Day 4 — Neural Network From Scratch

> **21-Day Deep Learning Journey — Day 4**

On Day 4, the goal was to move from a **single-layer model** to a complete **multi-layer neural network**, while still implementing the learning process ourselves using NumPy.

Instead of using PyTorch, TensorFlow, or another deep learning framework, this project builds the essential components of a neural network manually:

* Neural network architecture
* Parameter initialization
* Forward propagation
* Activation functions
* Binary cross-entropy loss
* Backpropagation
* Gradient descent
* Parameter updates
* Training loop
* Evaluation
* Decision boundary visualization

The purpose of this project is not simply to achieve high accuracy.

The purpose is to understand **what is actually happening inside a neural network when it learns**.

---

# 1. What This Project Is About

A neural network can be thought of as a sequence of transformations:

$$
X
\rightarrow
\text{Linear Transformation}
\rightarrow
\text{Activation}
\rightarrow
\text{Linear Transformation}
\rightarrow
\text{Activation}
\rightarrow
\text{Output}
$$

During training, the network repeatedly performs:

$$
\boxed{
\text{Forward Pass}
\rightarrow
\text{Loss}
\rightarrow
\text{Backpropagation}
\rightarrow
\text{Parameter Update}
}
$$

This loop is the fundamental mechanism behind neural network training.

The project implements this process from scratch.

---

# 2. Connection With Previous Days

Day 4 builds directly on everything learned during Days 1–3.

## Day 1 — Deep Learning Fundamentals

We learned the basic computation performed by a neuron:

$$
z=w^Tx+b
$$

followed by an activation:

$$
a=f(z)
$$

We also introduced:

* weights
* bias
* activation functions
* loss
* gradients
* optimization

---

## Day 2 — Mathematics

We developed the mathematical foundation required to understand neural networks:

* vectors
* matrices
* dot products
* derivatives
* partial derivatives
* chain rule
* gradients

The chain rule becomes especially important during backpropagation.

---

## Day 3 — Logistic Regression From Scratch

We implemented binary classification using:

$$
z=Xw+b
$$

$$
\hat y=\sigma(z)
$$

and Binary Cross-Entropy:

$$
L=
-\frac{1}{m}
\sum
\left[
y\log(\hat y)
+
(1-y)\log(1-\hat y)
\right]
$$

We also derived the gradient:

$$
\frac{\partial L}{\partial w}
=
\frac{1}{m}X^T(\hat y-y)
$$

Day 4 extends this idea by adding **multiple layers and nonlinear activation functions**.

---

# 3. Why Do We Need a Neural Network?

A single logistic regression model is fundamentally limited to learning a linear decision boundary.

For example, a problem like:

```text
Class 0     |     Class 1
Class 0     |     Class 1
Class 0     |     Class 1
```

can be separated with a straight line.

But many real-world problems are nonlinear.

For this project, we use the **Two Moons** dataset.

Conceptually:

```text
        ● ● ●
      ● ● ●
    ● ●
   ●

                ○
              ○ ○
            ○ ○ ○
          ○ ○ ○
```

The classes curve around each other.

A simple straight-line decision boundary is not sufficient.

This gives us a practical reason to introduce hidden layers and nonlinear activation functions.

---

# 4. Dataset

The dataset is generated using Scikit-learn's `make_moons`.

```python
X, y = make_moons(
    n_samples=1000,
    noise=0.2,
    random_state=42
)
```

We have:

```text
1000 samples
2 features
```

Therefore:

$$
X\in\mathbb{R}^{1000\times2}
$$

Each sample looks conceptually like:

$$
[x_1,x_2]
$$

and has a binary target:

$$
y\in\{0,1\}
$$

---

# 5. Train/Test Split

The dataset is divided into:

* 80% training data
* 20% testing data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Therefore:

$$
X_{train}:(800,2)
$$

$$
y_{train}:(800,)
$$

and:

$$
X_{test}:(200,2)
$$

$$
y_{test}:(200,)
$$

The training data is used to learn the parameters.

The test data is kept separate so that we can evaluate how the trained model performs on unseen examples.

---

# 6. Feature Scaling

The input features are standardized:

$$
x'=\frac{x-\mu}{\sigma}
$$

where:

* \(\mu\) = training-set mean
* \(\sigma\) = training-set standard deviation

The scaler is fitted only on the training data:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

This distinction is important.

We calculate the scaling parameters from the training data and then apply those same parameters to the test data.

We do **not** calculate separate statistics from the test set because that would allow information from the test distribution to leak into the preprocessing pipeline.

---

# 7. Network Architecture

The neural network used in this project is:

$$
\boxed{2\rightarrow16\rightarrow8\rightarrow1}
$$

Meaning:

```text
Input Layer
     ↓
2 features
     ↓
Hidden Layer 1
     ↓
16 neurons
     ↓
Hidden Layer 2
     ↓
8 neurons
     ↓
Output Layer
     ↓
1 neuron
```

The hidden layers use ReLU.

The output layer uses sigmoid.

Therefore:

$$
\boxed{
2
\rightarrow
16\;(\text{ReLU})
\rightarrow
8\;(\text{ReLU})
\rightarrow
1\;(\text{Sigmoid})
}
$$

---

# 8. Understanding the Matrix Dimensions

One of the most important parts of the project is understanding why the weight matrices have their particular shapes.

The rule is:

$$
\boxed{
W=(\text{number of inputs},\text{number of neurons})
}
$$

Therefore:

### First layer

2 inputs → 16 neurons:

$$
W_1:(2,16)
$$

### Second layer

16 inputs → 8 neurons:

$$
W_2:(16,8)
$$

### Third layer

8 inputs → 1 neuron:

$$
W_3:(8,1)
$$

The biases are:

$$
b_1:(1,16)
$$

$$
b_2:(1,8)
$$

$$
b_3:(1,1)
$$

---

# 9. Matrix Flow Through the Network

Our training data has:

$$
X:(800,2)
$$

First layer:

$$
Z_1=XW_1+b_1
$$

giving:

$$
(800,2)(2,16)
=
(800,16)
$$

Therefore:

$$
Z_1:(800,16)
$$

After ReLU:

$$
A_1:(800,16)
$$

Second layer:

$$
Z_2=A_1W_2+b_2
$$

giving:

$$
(800,16)(16,8)
=
(800,8)
$$

Therefore:

$$
A_2:(800,8)
$$

Finally:

$$
Z_3=A_2W_3+b_3
$$

giving:

$$
(800,8)(8,1)
=
(800,1)
$$

After sigmoid:

$$
\hat y:(800,1)
$$

The complete shape flow is:

```text
X
(800,2)
   ↓
Z1
(800,16)
   ↓
A1
(800,16)
   ↓
Z2
(800,8)
   ↓
A2
(800,8)
   ↓
Z3
(800,1)
   ↓
ŷ
(800,1)
```

---

# 10. Activation Functions

Without activation functions, stacking multiple linear transformations would still result in an overall linear transformation.

This would defeat the main purpose of the hidden layers.

Therefore, nonlinear activation functions are required.

---

## 10.1 ReLU

The hidden layers use ReLU:

$$
ReLU(x)=\max(0,x)
$$

Examples:

$$
ReLU(-3)=0
$$

$$
ReLU(2)=2
$$

So negative values become zero while positive values pass through.

The derivative is:

$$
ReLU'(x)=
\begin{cases}
1 & x>0\\
0 & x\leq0
\end{cases}
$$

The derivative becomes important during backpropagation.

---

## 10.2 Sigmoid

The output layer uses sigmoid:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Sigmoid maps values into:

$$
(0,1)
$$

This makes the output interpretable as a probability for the positive class.

For example:

$$
\sigma(2)\approx0.88
$$

can be interpreted as approximately an 88% predicted probability for class 1.

---

# 11. Parameter Initialization

The network needs initial values for all weights and biases.

The biases are initialized to zero:

$$
b=0
$$

The weights are randomly initialized.

Initially, simple random initialization was considered:

```python
np.random.randn(...) * 0.01
```

However, because the hidden layers use ReLU, the project uses **He initialization**.

The standard deviation is:

$$
\sqrt{\frac{2}{n_{in}}}
$$

where \(n_{in}\) is the number of inputs to the layer.

Therefore:

$$
W_1
\sim
N
\left(
0,
\frac{2}{2}
\right)
$$

$$
W_2
\sim
N
\left(
0,
\frac{2}{16}
\right)
$$

$$
W_3
\sim
N
\left(
0,
\frac{2}{8}
\right)
$$

Implementation:

```python
W1 = np.random.randn(2, 16) * np.sqrt(2 / 2)
b1 = np.zeros((1, 16))

W2 = np.random.randn(16, 8) * np.sqrt(2 / 16)
b2 = np.zeros((1, 8))

W3 = np.random.randn(8, 1) * np.sqrt(2 / 8)
b3 = np.zeros((1, 1))
```

---

# 12. Why Not Initialize Weights to Zero?

If all weights were zero, neurons in the same layer would initially behave identically.

They would receive the same signals and gradients.

As a result, the neurons would continue learning the same thing.

Random initialization breaks this symmetry and allows different neurons to learn different representations.

---

# 13. Forward Propagation

The forward pass calculates the network's prediction.

The equations are:

### Layer 1

$$
Z_1=XW_1+b_1
$$

$$
A_1=ReLU(Z_1)
$$

### Layer 2

$$
Z_2=A_1W_2+b_2
$$

$$
A_2=ReLU(Z_2)
$$

### Output

$$
Z_3=A_2W_3+b_3
$$

$$
A_3=\sigma(Z_3)
$$

and:

$$
\boxed{\hat y=A_3}
$$

---

# 14. The Forward Function

The forward pass is encapsulated into:

```python
def forward(X, params):

    W1 = params["W1"]
    b1 = params["b1"]

    W2 = params["W2"]
    b2 = params["b2"]

    W3 = params["W3"]
    b3 = params["b3"]

    Z1 = X @ W1 + b1
    A1 = ReLU(Z1)

    Z2 = A1 @ W2 + b2
    A2 = ReLU(Z2)

    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3
    }

    return A3, cache
```

---

# 15. Why Do We Store a Cache?

The forward pass produces intermediate values:

$$
Z_1,A_1,Z_2,A_2,Z_3,A_3
$$

These values are needed later during backpropagation.

For example:

$$
dZ_2=dA_2\odot ReLU'(Z_2)
$$

Therefore, the backward pass needs access to \(Z_2\).

The cache prevents us from having to recompute all the intermediate values.

Conceptually:

```text
Forward Pass
     ↓
Cache intermediate values
     ↓
Backward Pass
     ↓
Use cached values to calculate gradients
```

---

# 16. Binary Cross-Entropy Loss

After the forward pass, we have predictions:

$$
\hat y
$$

We need to measure how wrong those predictions are.

For binary classification, we use Binary Cross-Entropy:

$$
L=
-\frac{1}{m}
\sum_{i=1}^{m}
[
y_i\log(\hat y_i)
+
(1-y_i)\log(1-\hat y_i)
]
$$

A confident correct prediction produces a small loss.

A confident incorrect prediction produces a large loss.

---

# 17. Why Use Epsilon?

The implementation uses:

```python
epsilon = 1e-8
```

because:

$$
\log(0)
$$

is undefined.

If the model produces exactly:

$$
\hat y=0
$$

then:

$$
\log(\hat y)
=
\log(0)
$$

which can produce an infinite value numerically.

Therefore:

$$
\log(\hat y+\epsilon)
$$

and:

$$
\log(1-\hat y+\epsilon)
$$

provide numerical stability.

The epsilon is not intended to change the learning behavior significantly.

It simply prevents numerical problems.

---

# 18. Backpropagation

This is the central mathematical component of the project.

The goal is to determine:

> How much did each parameter contribute to the final loss?

We calculate:

$$
\frac{\partial L}{\partial W_1},
\frac{\partial L}{\partial b_1},
\frac{\partial L}{\partial W_2},
\frac{\partial L}{\partial b_2},
\frac{\partial L}{\partial W_3},
\frac{\partial L}{\partial b_3}
$$

These are the gradients.

---

# 19. Why Does Backpropagation Work Backwards?

The forward computation is:

```text
X
 ↓
Z1
 ↓
A1
 ↓
Z2
 ↓
A2
 ↓
Z3
 ↓
ŷ
 ↓
Loss
```

The loss depends on the prediction.

The prediction depends on \(Z_3\).

\(Z_3\) depends on \(A_2\) and \(W_3\).

And so on.

Therefore, using the chain rule, we calculate the gradients in reverse:

```text
Loss
 ↓
dZ3
 ↓
dA2
 ↓
dZ2
 ↓
dA1
 ↓
dZ1
```

This is the essence of backpropagation.

---

# 20. Output Layer Gradient

For sigmoid combined with Binary Cross-Entropy, the derivative simplifies to:

$$
\boxed{
dZ_3=A_3-y
}
$$

This is an important result.

Then:

$$
dW_3=
\frac{1}{m}
A_2^TdZ_3
$$

and:

$$
db_3=
\frac{1}{m}
\sum dZ_3
$$

---

# 21. Backpropagation Through Hidden Layer 2

First:

$$
dA_2=dZ_3W_3^T
$$

Then we pass through ReLU:

$$
dZ_2=
dA_2\odot ReLU'(Z_2)
$$

Then:

$$
dW_2=
\frac{1}{m}
A_1^TdZ_2
$$

and:

$$
db_2=
\frac{1}{m}
\sum dZ_2
$$

---

# 22. Backpropagation Through Hidden Layer 1

First:

$$
dA_1=dZ_2W_2^T
$$

Then:

$$
dZ_1=
dA_1\odot ReLU'(Z_1)
$$

Finally:

$$
dW_1=
\frac{1}{m}
X^TdZ_1
$$

and:

$$
db_1=
\frac{1}{m}
\sum dZ_1
$$

---

# 23. Gradient Shape Principle

An important debugging principle is:

> **Every parameter's gradient must have exactly the same shape as that parameter.**

Therefore:

```text
W1  → (2,16)
dW1 → (2,16)

b1  → (1,16)
db1 → (1,16)

W2  → (16,8)
dW2 → (16,8)

b2  → (1,8)
db2 → (1,8)

W3  → (8,1)
dW3 → (8,1)

b3  → (1,1)
db3 → (1,1)
```

This is one of the most useful checks when implementing neural networks manually.

---

# 24. Complete Backpropagation

The implementation follows the mathematical derivation:

```python
def backward(X, y, params, cache):

    m = X.shape[0]

    W1 = params["W1"]
    W2 = params["W2"]
    W3 = params["W3"]

    Z1 = cache["Z1"]
    A1 = cache["A1"]

    Z2 = cache["Z2"]
    A2 = cache["A2"]

    A3 = cache["A3"]

    y = y.reshape(-1, 1)

    # Output layer
    dZ3 = A3 - y

    dW3 = (1 / m) * (A2.T @ dZ3)

    db3 = (1 / m) * np.sum(
        dZ3,
        axis=0,
        keepdims=True
    )

    # Hidden layer 2
    dA2 = dZ3 @ W3.T

    dZ2 = dA2 * (Z2 > 0)

    dW2 = (1 / m) * (A1.T @ dZ2)

    db2 = (1 / m) * np.sum(
        dZ2,
        axis=0,
        keepdims=True
    )

    # Hidden layer 1
    dA1 = dZ2 @ W2.T

    dZ1 = dA1 * (Z1 > 0)

    dW1 = (1 / m) * (X.T @ dZ1)

    db1 = (1 / m) * np.sum(
        dZ1,
        axis=0,
        keepdims=True
    )

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3
    }

    return grads
```

---

# 25. Gradient Descent

Calculating gradients isn't enough.

We need to use them to modify the parameters.

The gradient descent update rule is:

$$
\boxed{
\theta_{new}
=
\theta_{old}
-
\eta
\nabla_\theta L
}
$$

For our weights:

$$
W_1:=W_1-\eta dW_1
$$

$$
W_2:=W_2-\eta dW_2
$$

$$
W_3:=W_3-\eta dW_3
$$

For the biases:

$$
b_1:=b_1-\eta db_1
$$

$$
b_2:=b_2-\eta db_2
$$

$$
b_3:=b_3-\eta db_3
$$

The learning rate controls how large each update is.

---

# 26. Learning Rate

The project starts with:

$$
\eta=0.01
$$

A very small learning rate can make training extremely slow.

A very large learning rate can make training unstable.

Later experiments will investigate this relationship.

---

# 27. The Complete Learning Process

At this point, all the individual pieces fit together:

```text
                ┌─────────────────────────────┐
                │                             │
                ▼                             │
             Forward                         │
                │                             │
                ▼                             │
            Prediction                       │
                │                             │
                ▼                             │
               Loss                          │
                │                             │
                ▼                             │
          Backpropagation                     │
                │                             │
                ▼                             │
             Gradients                        │
                │                             │
                ▼                             │
          Gradient Descent                    │
                │                             │
                └────────── Update ───────────┘
```

Mathematically:

$$
\boxed{
X
\rightarrow
\hat y
\rightarrow
L
\rightarrow
\nabla L
\rightarrow
\theta_{new}
}
$$

This cycle is repeated many times.

---

# 28. Training Loop

The next stage is to repeatedly execute:

1. Forward pass
2. Calculate loss
3. Backpropagate
4. Update parameters
5. Record the loss

Conceptually:

```python
for epoch in range(epochs):

    y_pred, cache = forward(X_train, params)

    loss = binary_cross_entropy(
        y_train.reshape(-1, 1),
        y_pred
    )

    grads = backward(
        X_train,
        y_train,
        params,
        cache
    )

    # Update parameters
```

As training progresses, we expect the loss to decrease.

---

# 29. What Does an Epoch Mean?

An **epoch** means the model has processed the complete training dataset once.

For example:

```text
800 training samples
        ↓
Forward
        ↓
Loss
        ↓
Backward
        ↓
Update
        ↓
1 epoch
```

If we train for 1000 epochs, the model repeatedly sees the training data 1000 times.

---

# 30. What We Expect During Training

Initially:

```text
Prediction
   ↓
mostly untrained
   ↓
higher loss
```

As training progresses:

```text
Prediction
   ↓
improves
   ↓
loss decreases
```

Ideally, a loss curve will look generally like:

```text
Loss
 │\
 │ \
 │  \
 │   \
 │    \____
 │         \____
 └──────────────── Epoch
```

The exact shape will vary.

---

# 31. Evaluation

After training, the model produces probabilities:

$$
\hat y=P(y=1|X)
$$

We convert them into class predictions using a threshold:

$$
\hat y\geq0.5
\Rightarrow
class\ 1
$$

$$
\hat y<0.5
\Rightarrow
class\ 0
$$

Then we evaluate on the test set.

The important distinction is:

```text
Training data
→ used to learn parameters

Test data
→ used to evaluate the learned model
```

---

# 32. Decision Boundary

One of the most useful visualizations for this project is the decision boundary.

The network receives:

$$
(x_1,x_2)
$$

and produces:

$$
P(y=1)
$$

We can evaluate the network across a grid of points in the feature space.

For each point, we determine which class the network predicts.

This allows us to visualize the learned boundary.

For a successful model, the boundary should be **nonlinear**, following the structure of the Two Moons dataset.

This provides a visual demonstration of why the hidden layers matter.

---

# 33. Experiments

Once the base model works, the project should not end there.

The experiments help understand the effect of architecture and optimization.

---

## Experiment 1 — Logistic Regression Baseline

Use:

$$
2\rightarrow1
$$

This is essentially the model from Day 3.

Compare its decision boundary with the neural network.

---

## Experiment 2 — One Hidden Layer

Try:

$$
2\rightarrow16\rightarrow1
$$

Compare it against the deeper architecture.

---

## Experiment 3 — Two Hidden Layers

Main architecture:

$$
2\rightarrow16\rightarrow8\rightarrow1
$$

---

## Experiment 4 — Hidden Layer Size

Try different configurations:

$$
2\rightarrow4\rightarrow1
$$

$$
2\rightarrow8\rightarrow1
$$

$$
2\rightarrow16\rightarrow1
$$

$$
2\rightarrow32\rightarrow1
$$

Observe how the representation and training behavior change.

---

## Experiment 5 — Learning Rate

Compare:

$$
0.001
$$

$$
0.01
$$

$$
0.1
$$

Record:

* training loss
* convergence behavior
* test performance
* decision boundary

---

# 34. Important Lessons From the Project

## 34.1 A neural network is a composition of functions

The network isn't just a collection of neurons.

It is a mathematical composition:

$$
f(X)
=
f_3(
f_2(
f_1(X)
)
)
$$

Each layer transforms the representation.

---

## 34.2 Hidden layers learn representations

The input begins as:

$$
(x_1,x_2)
$$

The first hidden layer transforms it into 16 learned features.

The second transforms those into 8 learned features.

The output layer then uses those representations for classification.

Therefore:

$$
\text{Raw Features}
\rightarrow
\text{Learned Features}
\rightarrow
\text{Prediction}
$$

---

## 34.3 Backpropagation is the chain rule applied repeatedly

Backpropagation isn't a mysterious neural-network algorithm.

At its core, it is the chain rule applied through a sequence of mathematical operations.

$$
\frac{\partial L}{\partial W_1}
$$

depends on all the transformations between \(W_1\) and \(L\).

That's why the gradients are calculated backwards.

---

## 34.4 Gradient descent performs the learning

Backpropagation tells us:

> "Which direction should each parameter move?"

Gradient descent actually moves the parameters.

So:

```text
Backpropagation
      ↓
Calculate gradients
      ↓
Gradient Descent
      ↓
Update parameters
```

These are related but different concepts.

---

# 35. Final Architecture

The complete model is:

```text
                   Neural Network

Input               Hidden 1        Hidden 2       Output

x₁ ────────┐
           │
x₂ ────────┘
              ↓
          Linear Layer
              ↓
         16 neurons
              ↓
            ReLU
              ↓
          8 neurons
              ↓
            ReLU
              ↓
          1 neuron
              ↓
           Sigmoid
              ↓
        P(y = 1)
```

Mathematically:

$$
Z_1=XW_1+b_1
$$

$$
A_1=ReLU(Z_1)
$$

$$
Z_2=A_1W_2+b_2
$$

$$
A_2=ReLU(Z_2)
$$

$$
Z_3=A_2W_3+b_3
$$

$$
\hat y=\sigma(Z_3)
$$

Then:

$$
L=
-\frac1m
\sum
[
y\log(\hat y)
+
(1-y)\log(1-\hat y)
]
$$

Backpropagation calculates the gradients, and gradient descent updates the parameters.

---

# 36. Day 4 Takeaway

Day 3 showed how to build a **single trainable linear classifier**.

Day 4 extends that idea into a complete multi-layer neural network.

The most important conceptual progression is:

$$
\boxed{
\text{Neuron}
\rightarrow
\text{Layer}
\rightarrow
\text{Network}
}
$$

and:

$$
\boxed{
\text{Forward}
\rightarrow
\text{Loss}
\rightarrow
\text{Backprop}
\rightarrow
\text{Update}
}
$$

At the end of this project, the neural network is no longer a black box.

Every major operation inside it has been implemented explicitly:

* matrix multiplication
* bias addition
* activation functions
* loss calculation
* gradient calculation
* parameter updates

This provides the mathematical foundation required to understand higher-level frameworks such as PyTorch rather than simply using them as abstractions.

---

# 37. What Comes Next

The next stage of the project is to complete the **training loop** and allow the network to repeatedly perform:

$$
\boxed{
Forward
\rightarrow
Loss
\rightarrow
Backward
\rightarrow
Update
}
$$

After training, we will evaluate the model, visualize the decision boundary, and run controlled experiments on:

* architecture
* hidden-layer size
* learning rate
* initialization

The ultimate goal is not simply to make the model work.

It is to understand **why it works**.
