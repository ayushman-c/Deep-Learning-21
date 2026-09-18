# Deep Learning 21-Day Roadmap

A project-based 21-day journey to develop a strong conceptual and practical understanding of Deep Learning using Python and PyTorch.

The goal of this repository is not to cover Deep Learning through theory alone. Each concept is introduced when it becomes necessary for building a project, with implementations progressing from fundamental mathematics and NumPy to PyTorch, Computer Vision, NLP, Transformers, RAG, and modern AI systems.

---

## Objectives

This roadmap is designed to build an understanding of:

* Neural network fundamentals
* Deep Learning mathematics
* Training and optimization
* Computer Vision
* Natural Language Processing
* Transformers and attention
* Large Language Models
* Retrieval-Augmented Generation
* Fine-tuning and optimization
* End-to-end AI application development

The emphasis throughout the roadmap is on understanding **how and why models work**, rather than treating frameworks as black boxes.

---

## Technology Stack

### Core

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* PyTorch

### Later Stages

* Hugging Face Transformers
* OpenCV
* FastAPI
* Vector databases
* Docker
* Git/GitHub

Libraries are introduced only when they become relevant to a specific project or concept.

---

## Repository Structure

```text
Deep-Learning-21/
│
├── Day 01/
│   ├── README.md
│   └── ...
│
├── Day 02/
│   └── ...
│
├── Day 03/
│   └── ...
│
├── ...
│
└── Day 21/
    └── ...
```

Each day contains the implementation, documentation, and practical work associated with that stage of the roadmap.

---

## Learning Approach

The roadmap follows a progressive build-first methodology:

```text
Concept
   ↓
Intuition
   ↓
Mathematics
   ↓
Small Implementation
   ↓
Hands-on Exercise
   ↓
Project Component
   ↓
Understanding Check
   ↓
Next Concept
```

Important components are implemented manually before relying on higher-level frameworks whenever practical.

For example, the first neural-network implementation is built using NumPy before introducing PyTorch.

---

# Roadmap Progress

| Day | Topic                                       | Status      |
| --- | ------------------------------------------- | ----------- |
| 01  | Deep Learning Foundations                   | Completed   |
| 02  | Neural Network Mathematics                  | In Progress |
| 03  | Logistic Regression + Binary Classification | Upcoming    |
| 04  | Neural Network From Scratch                 | Upcoming    |
| 05  | Backpropagation Deep Dive                   | Upcoming    |
| 06  | PyTorch Fundamentals                        | Upcoming    |
| 07  | Training Neural Networks                    | Upcoming    |
| 08  | Improving Neural Networks                   | Upcoming    |
| 09  | CNN Fundamentals                            | Upcoming    |
| 10  | CNN Project                                 | Upcoming    |
| 11  | Transfer Learning                           | Upcoming    |
| 12  | Computer Vision Beyond Classification       | Upcoming    |
| 13  | NLP Foundations                             | Upcoming    |
| 14  | RNN, LSTM and GRU                           | Upcoming    |
| 15  | Attention                                   | Upcoming    |
| 16  | Transformers                                | Upcoming    |
| 17  | Modern LLMs                                 | Upcoming    |
| 18  | Generative AI + Small Language Model        | Upcoming    |
| 19  | Retrieval-Augmented Generation              | Upcoming    |
| 20  | Fine-Tuning + Optimization                  | Upcoming    |
| 21  | Capstone Project                            | Upcoming    |

---

## Day 1 — Deep Learning Foundations

The first day establishes the fundamental mechanics behind neural network training.

Topics covered:

* AI, Machine Learning, and Deep Learning
* Neurons
* Weights and biases
* Activation functions
* Forward propagation
* Loss functions
* Gradients
* Gradient descent
* Learning rate
* Parameter updates
* Fully connected layers

### Day 1 Project

A single neuron was implemented from scratch using NumPy and trained using gradient descent.

The model:

$$
\hat{y} = wx+b
$$

was trained to learn a simple linear relationship without using PyTorch or automatic differentiation.

The implementation manually performs:

```text
Forward Pass
     ↓
Loss Calculation
     ↓
Gradient Calculation
     ↓
Parameter Update
     ↓
Repeat
```

The project also includes experiments with different learning rates to understand their effect on convergence.

See [`Day 01`](./Day%2001/) for the complete implementation and documentation.

---

## Current Focus

The next stage is **Day 2 — Neural Network Mathematics**.

Day 2 builds the mathematical foundation required to move from scalar operations such as:

$$
wx+b
$$

to vector and matrix operations used in real neural networks:

$$
\mathbf{W}\mathbf{x}+\mathbf{b}
$$

Topics include vectors, matrices, dot products, matrix multiplication, derivatives, the chain rule, and gradients.

---

## Contribution

This is intended as a community-oriented learning repository.

Contributions that improve:

* Explanations
* Mathematical clarity
* Implementations
* Examples
* Documentation
* Exercises
* Learning resources

are welcome.

When contributing, prioritize clarity and educational value over unnecessary abstraction or complexity.

---

## License

Add the project's chosen open-source license here.
