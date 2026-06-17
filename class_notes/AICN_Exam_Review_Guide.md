# AICN Complete Exam Review (All Taught Files)

Prepared on: 2026-06-16

## Exam Pattern

- Section A: Attempt 6 out of 7 short questions, 10 marks each
- Section B: Attempt 2 out of 3 long questions, 20 marks each

## Master Syllabus

1. Python foundations: introduction, data types, loops, functions, OOP
2. Data tools: NumPy, Pandas, Matplotlib, Seaborn
3. EDA and feature engineering
4. Calculus: derivatives, chain rule, gradients
5. Linear algebra: matrices, eigenvalues, eigenvectors
6. Probability and statistics
7. Machine learning workflow: encoding, scaling, train-test split
8. Linear regression and logistic regression
9. Model evaluation metrics
10. Imbalanced data handling
11. Supervised learning algorithms
12. Unsupervised learning and dimensionality reduction
13. Deep learning fundamentals
14. Neural network from scratch
15. PyTorch core concepts
16. CNN and transfer learning
17. RNN, LSTM, and GRU
18. Transformers and BERT sentiment analysis

## Section A Short Questions

### 1. What is supervised learning, unsupervised learning, and deep learning?

- Supervised learning learns from labeled data `(X, y)`.
- Unsupervised learning finds patterns in unlabeled data.
- Deep learning uses multi-layer neural networks to learn complex representations.

Examples:

- Supervised: logistic regression for heart disease prediction
- Unsupervised: K-Means for customer segmentation
- Deep learning: CNN for image classification

### 2. What are encoding and feature scaling?

Encoding converts categorical values into numeric values.

- Label Encoding: converts categories into integers
- One-Hot Encoding: creates separate binary columns for each category

Feature scaling makes features comparable in magnitude.

- MinMaxScaler: `x' = (x - min) / (max - min)`
- StandardScaler: `x' = (x - mean) / std`
- RobustScaler: uses median and IQR, good for outliers

### 3. Write the chain rule and explain its role in backpropagation.

Chain rule:

- If `y = f(g(x))`, then `dy/dx = (dy/dg) * (dg/dx)`

In neural networks, the gradient of loss with respect to earlier weights is found by multiplying local derivatives layer by layer. This is the core idea of backpropagation.

### 4. Differentiate linear regression and logistic regression.

Linear regression:

- Predicts continuous output
- Formula: `y_hat = b0 + b1x1 + ... + bnxn`
- Common loss: MSE

Logistic regression:

- Predicts probability of a class
- Formula: `z = b0 + b1x1 + ... + bnxn`
- Sigmoid: `p = 1 / (1 + e^(-z))`
- Common loss: binary cross-entropy

### 5. What are accuracy, precision, recall, and F1-score?

- Accuracy = `(TP + TN) / (TP + TN + FP + FN)`
- Precision = `TP / (TP + FP)`
- Recall = `TP / (TP + FN)`
- F1-score = `2 * Precision * Recall / (Precision + Recall)`

Accuracy can be misleading in imbalanced datasets.

### 6. Explain K-Means, PCA, DBSCAN, and t-SNE.

- K-Means: centroid-based clustering
- PCA: dimensionality reduction using maximum variance directions
- DBSCAN: density-based clustering with noise detection
- t-SNE: nonlinear visualization for high-dimensional data

### 7. What are activation function, loss function, optimizer, and regularization?

- Activation function: adds non-linearity, e.g. ReLU, Sigmoid, Tanh
- Loss function: measures prediction error, e.g. MSE, cross-entropy
- Optimizer: updates weights, e.g. SGD, RMSprop, Adam
- Regularization: reduces overfitting, e.g. Dropout, L2, early stopping

## Formula Bank

- Derivative: `d(x^n)/dx = n * x^(n-1)`
- Chain rule: `d(f(g(x)))/dx = f'(g(x)) * g'(x)`
- Gradient descent: `theta = theta - eta * grad(J(theta))`
- MSE: `(1/n) * sum((y - y_hat)^2)`
- Sigmoid: `1 / (1 + e^(-z))`
- Binary cross-entropy: `-[y log(p) + (1-y) log(1-p)]`
- Precision: `TP / (TP + FP)`
- Recall: `TP / (TP + FN)`
- F1-score: `2PR / (P + R)`
- CNN output height: `H_out = floor((H_in + 2p - k)/s) + 1`

## Linear Regression From Scratch

### Core Idea

Linear regression models the relationship between input features and a continuous target.

- Hypothesis: `y_hat = Xw + b`
- Cost: `J(w, b) = (1/m) * sum((y_hat - y)^2)`

### Gradients

- `dJ/dw = (2/m) * X^T (y_hat - y)`
- `dJ/db = (2/m) * sum(y_hat - y)`

### Update Rule

- `w = w - alpha * dJ/dw`
- `b = b - alpha * dJ/db`

### Exam Steps

1. Split features and target
2. Apply scaling if needed
3. Initialize weights and bias
4. Compute predictions
5. Compute loss
6. Compute gradients
7. Update parameters
8. Repeat for epochs
9. Evaluate using MAE, RMSE, and R2

### Short sklearn Code

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)
```

## Logistic Regression From Scratch

### Core Idea

Logistic regression is used for binary classification.

- Linear part: `z = Xw + b`
- Probability output: `p = sigmoid(z)`

### Loss

- Binary cross-entropy:
- `J = -(1/m) * sum(y log(p) + (1-y) log(1-p))`

### Prediction

- If `p >= 0.5`, predict class `1`
- Else predict class `0`

## Neural Network From Scratch

### Architecture

For a simple 2-layer network:

- `Z1 = XW1 + b1`
- `A1 = ReLU(Z1)` or `tanh(Z1)`
- `Z2 = A1W2 + b2`
- `A2 = sigmoid(Z2)` for binary classification

### Loss

- `J = -(1/m) * sum(y log(A2) + (1-y) log(1-A2))`

### Backpropagation

Using chain rule:

- Find derivative at output layer
- Move backward through each layer
- Compute `dW`, `db`
- Update with learning rate

### Minimal NumPy Skeleton

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
y_hat = sigmoid(Z2)

eps = 1e-8
loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

dZ2 = (y_hat - y) / len(y)
dW2 = A1.T @ dZ2
db2 = np.sum(dZ2, axis=0, keepdims=True)
dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0)
dW1 = X.T @ dZ1
db1 = np.sum(dZ1, axis=0, keepdims=True)

lr = 0.01
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2
```

### What to Write in Exam

1. Initialization of weights and biases
2. Forward propagation
3. Loss computation
4. Backpropagation using chain rule
5. Parameter update
6. Prediction and evaluation

## Model Evaluation Metrics

### Classification

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Log Loss

### Regression

- MAE
- MSE
- RMSE
- R2 score

## Imbalanced Dataset Handling

- Random undersampling
- Random oversampling
- SMOTE
- Class weights

Important note:

- Accuracy alone is not enough for imbalanced datasets
- Precision and recall are often more informative

## Supervised Learning Algorithms

- KNN Classifier
- KNN Regressor
- Naive Bayes
- Decision Tree Classifier
- Decision Tree Regressor

## Unsupervised Learning

- K-Means clustering
- DBSCAN
- Hierarchical clustering
- PCA
- t-SNE
- Customer segmentation

## Deep Learning Fundamentals

- Activation functions: Sigmoid, Tanh, ReLU
- Forward propagation
- Loss functions: MSE, binary cross-entropy, categorical cross-entropy
- Backpropagation
- Optimizers: SGD, RMSprop, Adam
- Regularization: Dropout, L2 regularization

## PyTorch Important Concepts

- Tensor creation and tensor shape
- dtype and device
- Indexing and slicing
- Reshape, squeeze, unsqueeze
- Autograd
- `nn.Module`
- Loss functions
- Optimizers
- `model.train()` and `model.eval()`
- `DataLoader`

## CNN and Transfer Learning

### CNN Components

- Convolution layer
- ReLU activation
- Pooling layer
- Fully connected layer

### Important Terms

- Kernel
- Stride
- Padding
- Feature map
- Frozen backbone
- Fine-tuning

### Transfer Learning

- Use pretrained model such as ResNet
- Freeze earlier layers
- Replace final classifier
- Train classifier head
- Optionally unfreeze deeper layers for fine-tuning

## RNN, LSTM, and GRU

- RNN handles sequential data
- LSTM solves vanishing gradient better using memory cell and gates
- GRU is a simpler alternative to LSTM with fewer gates

Common uses:

- Text modeling
- Time series forecasting
- Sentiment analysis

## Transformers and BERT

### Transformer

- Uses self-attention
- Handles long-range dependencies better than RNN
- Supports parallel computation

### BERT Sentiment Workflow

1. Tokenize text
2. Convert to input ids and attention masks
3. Feed into pretrained BERT
4. Get logits
5. Compute cross-entropy loss
6. Update with AdamW

## Section B Long Questions

### Long Question 1: End-to-End Classification

Write about:

1. Data cleaning
2. Encoding
3. Scaling
4. Train-test split
5. Model fitting
6. Confusion matrix and metrics
7. Imbalance handling if needed
8. Interpretation

### Long Question 2: Clustering + PCA

Write about:

1. Why scaling is needed
2. PCA for dimensionality reduction
3. K-Means training
4. Silhouette score
5. Interpretation of clusters

### Long Question 3: Deep Learning

Possible focus:

1. Neural network from scratch
2. PyTorch training loop
3. CNN and transfer learning
4. Transformer sentiment model

## Rapid Revision Tips

1. Memorize all metric formulas
2. Practice one regression answer and one classification answer
3. Practice one neural network answer from scratch
4. Revise scaling before KNN, K-Means, and PCA
5. Revise why accuracy fails on imbalanced datasets

## Common Mistakes

- Writing only definitions without examples
- Using accuracy only for imbalanced data
- Forgetting scaling where needed
- Mixing regression metrics and classification metrics
- Writing code without explaining steps

## High-Yield Closing Lines

- Preprocessing quality strongly affects model performance.
- Accuracy should always be interpreted with dataset balance and business context.
- Backpropagation works through repeated application of the chain rule.
- Transfer learning improves results when labeled data is limited.
