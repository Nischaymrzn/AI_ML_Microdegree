## Course Overview

The course is designed to introduce the foundations of artificial intelligence and machine learning through both theory and practice. It combines mathematical preparation, programming skills, classical machine learning, and deep learning with hands-on implementation using Python libraries such as NumPy, pandas, matplotlib, seaborn, scikit-learn, and PyTorch.

The notebooks in this folder show a practical progression:

- Python basics and programming foundations
- Data manipulation and visualization
- Mathematical concepts needed for AI
- Machine learning preprocessing and workflows
- Supervised and unsupervised learning algorithms
- Deep learning from fundamentals to implementation
- CNNs and transfer learning
- Sequence modeling and transformers

## Repository Structure

- `1_intro (1).ipynb` to `5_oop (1).ipynb`: Python fundamentals
- `6_f_numpy (1).ipynb` to `10_j_eda1.ipynb`: data handling, visualization, and exploratory analysis
- `11_l_calculus.ipynb` to `14_m_statistics.ipynb`: mathematical foundations for AI
- `15_n_machinelearning (1).ipynb` to `19_r_imbalanced.ipynb`: ML preprocessing, regression, classification, metrics, and imbalance handling
- `20_t_supervisedalgorithms.ipynb` and `21_t_supervisedalgorithms (1).ipynb`: supervised algorithms
- `22_u_unsupervisedlearning (1).ipynb` to `26_y_t-SNE (1).ipynb`: clustering and dimensionality reduction
- `27_z_deeplearning (5).ipynb` and `28_PyTorch_Important_Concepts_From_Basic_to_Advanced.ipynb`: deep learning theory and implementation
- `29_transfer_learning_notebook.ipynb` and `30_Unit_7_CNNs_and_Transfer_Learning.ipynb`: CNNs and transfer learning
- `31_transformer_sentiment_analysis.ipynb`, `32_rnnlstmgrucomp.ipynb`, and `33_transformer_from_scratch_sentiment.ipynb`: sequence modeling and transformers

## Syllabus Alignment

### Unit I: Basics of Python

This unit builds the programming foundation required for all later AI and ML work. The material begins with introduction to Python syntax, variables, expressions, conditional statements, and control flow. It then moves into loops, functions, common built-in patterns, and the main data structures used in data science programming.

The notebooks also introduce object-oriented programming concepts such as classes, objects, constructors, methods, encapsulation, inheritance, and polymorphism. These topics support clean code organization for later machine learning and deep learning implementations.

Relevant notebooks:

- `1_intro (1).ipynb`
- `2_datatypes (1).ipynb`
- `3_loop.ipynb`
- `4_functions (1).ipynb`
- `5_oop (1).ipynb`

### Unit II: Python for Machine Learning

This unit focuses on the practical Python libraries used in machine learning workflows. NumPy is used for arrays, indexing, slicing, reshaping, broadcasting, random number generation, and matrix-style operations. pandas is used for tabular data handling, filtering, grouping, joining, missing-value handling, and data import/export. matplotlib and seaborn are used to create plots for understanding data distributions and relationships.

The unit also includes exploratory data analysis and feature-oriented data inspection. The Google Play Store EDA notebook shows how to inspect raw data, understand feature behavior, and create meaningful visual summaries before modeling.

Relevant notebooks:

- `6_f_numpy (1).ipynb`
- `7_g_pandas (2).ipynb`
- `8_h_matplotlib.ipynb`
- `9_i_seaborn.ipynb`
- `9_i_seaborn (1).ipynb`
- `10_j_eda.ipynb`
- `10_j_eda1.ipynb`

### Unit III: Mathematical Foundation of AI

This unit provides the mathematical concepts that support optimization, model interpretation, and algorithm design. The calculus content covers derivatives, derivative rules, chain rule, gradients, and the intuition needed for optimization and backpropagation. The linear algebra content introduces matrices, eigenvalues, eigenvectors, and related ideas that later connect to dimensionality reduction and deep learning.

The probability and statistics notebooks cover random variables, probability mass functions, probability density functions, common distributions, and core statistical measures such as mean, variance, and standard deviation. Sampling is also introduced as part of statistical reasoning.

Relevant notebooks:

- `11_l_calculus.ipynb`
- `12_k_linalg.ipynb`
- `13_m_probability.ipynb`
- `14_m_statistics.ipynb`

### Unit IV: Supervised Learning

This unit introduces the machine learning workflow and the main supervised algorithms taught in the course. The preprocessing notebook covers data encoding, feature scaling, and train-test split strategy. The regression and classification notebooks then move into model fitting, evaluation, and interpretation.

Linear regression is taught with gradient descent intuition and practical dataset work. Logistic regression is covered both conceptually and from scratch, including use of sigmoid activation, probability-based prediction, and classification evaluation. The unit also includes K-Fold validation, confusion-matrix-based metrics, and techniques to handle imbalanced data.

Relevant notebooks:

- `15_n_machinelearning (1).ipynb`
- `16_o_linalg (1).ipynb`
- `17_p_logisticregression (2).ipynb`
- `18_q_modelevaluationmetrics.ipynb`
- `19_r_imbalanced.ipynb`
- `20_t_supervisedalgorithms.ipynb`
- `21_t_supervisedalgorithms (1).ipynb`

Topics visibly covered in the notebooks:

- Data preprocessing workflow
- Encoding categorical variables
- Feature scaling with MinMaxScaler, StandardScaler, MaxAbsScaler, and RobustScaler
- Train-test split and validation concepts
- Linear regression with gradient descent
- Logistic regression from scratch
- KNN classification and regression
- Naive Bayes classification
- Decision Tree classification and regression
- Classification metrics: confusion matrix, accuracy, precision, recall, F1-score, ROC-AUC, log loss
- Regression metrics such as MSE and R2
- Imbalanced dataset handling with random undersampling, random oversampling, SMOTE, and class weights

### Unit V: Unsupervised Learning

This unit covers learning without target labels. The notebooks introduce the distinction between clustering and dimensionality reduction, and then move into practical clustering workflows using K-Means, DBSCAN, and hierarchical clustering. Customer segmentation is used as an applied example to show how unsupervised learning can produce meaningful groupings from behavior-based features.

Dimensionality reduction is covered through PCA and t-SNE. PCA is presented as a variance-preserving linear projection technique, while t-SNE is used as a visualization tool for high-dimensional data.

Relevant notebooks:

- `22_u_unsupervisedlearning (1).ipynb`
- `23_v_customersegmentation (1).ipynb`
- `24_w_pca (1).ipynb`
- `25_x_hierarchicalclustering.ipynb`
- `26_y_t-SNE (1).ipynb`

Topics visibly covered in the notebooks:

- K-Means clustering
- DBSCAN clustering
- Hierarchical clustering and linkage methods
- Customer segmentation
- PCA and explained variance
- Best practices for PCA, including feature standardization
- t-SNE visualization for high-dimensional data

### Unit VI: Deep Learning (1)

This unit introduces artificial neural networks from both theory and implementation. The deep learning notebook covers neuron structure, weights, bias, activation functions, forward propagation, loss functions, backpropagation, and gradient descent. It also discusses modern optimizers such as Adam, RMSprop, and SGD, along with regularization techniques like dropout and L2 regularization.

A major practical component of this unit is implementation of neural networks from scratch using NumPy. The notebooks show how hidden layers, activations, loss computation, backward propagation, and parameter updates fit into a training loop.

Relevant notebooks:

- `27_z_deeplearning (5).ipynb`
- `28_PyTorch_Important_Concepts_From_Basic_to_Advanced.ipynb`

Topics visibly covered in the notebooks:

- Artificial neural network structure
- Sigmoid, Tanh, and ReLU activation functions
- Forward propagation
- MSE, binary cross-entropy, and categorical cross-entropy
- Backpropagation and gradient flow
- SGD, mini-batch updates, RMSprop, and Adam
- Dropout and L2 regularization
- Neural network implementation from scratch with NumPy
- Transition from conceptual deep learning to PyTorch-based implementation

### Unit VII: Convolutional Neural Networks and Transfer Learning

This unit focuses on computer vision. The CNN notebook explains why convolutional networks work well for images and introduces the main components of CNN architecture: convolution layers, activation layers, pooling, and fully connected layers. It also discusses padding, stride, output-size reasoning, and the role of feature hierarchies in image understanding.

The transfer learning notebook shows how pretrained models can be adapted to new datasets. It distinguishes between feature extraction and fine-tuning, introduces frozen and trainable layers, and uses PyTorch-based training loops, dataloaders, and pretrained models to support practical image classification workflows.

Relevant notebooks:

- `29_transfer_learning_notebook.ipynb`
- `30_Unit_7_CNNs_and_Transfer_Learning.ipynb`

Topics visibly covered in the notebooks:

- Convolution and pooling
- Padding and stride
- CNN architecture and output shapes
- Image classification workflow
- Transfer learning
- Frozen backbone vs fine-tuning
- ResNet-based practical implementation
- Training and validation loops in PyTorch
- Visualization and interpretation concepts for CNN models

### Unit VIII: Sequence Modeling, Transformers, and Advanced Architectures

This unit covers sequence models and modern NLP architectures. The sequence notebook compares RNN, LSTM, and GRU models for sequential data and explains their purpose in learning time-dependent or order-sensitive patterns. The transformer notebooks then shift to self-attention-based models for sentiment analysis.

Two styles of transformer use are present in the materials. One notebook uses a pretrained transformer model for sentiment analysis, while another implements a transformer-style sentiment pipeline from scratch, including tokenization, dataset handling, dataloaders, model definition, training, evaluation, and prediction.

Relevant notebooks:

- `31_transformer_sentiment_analysis.ipynb`
- `32_rnnlstmgrucomp.ipynb`
- `33_transformer_from_scratch_sentiment.ipynb`

Topics visibly covered in the notebooks:

- RNN, LSTM, and GRU sequence modeling
- Sentiment analysis
- Pretrained transformer usage with tokenizers and classification heads
- Transformer model workflow in PyTorch
- Transformer from scratch for text classification
- Training loop, inference, and evaluation for NLP tasks

## Later Syllabus Units

The official syllabus continues beyond the material strongly represented in this folder. Units IX to XIII cover large language models, retrieval-augmented generation, voice AI, scalable AI systems, deployment, research workflows, ethics, and production-scale AI engineering.

Those later units are present in the syllabus document, but dedicated notebooks for them are not clearly available in this current folder. Because of that, this repository currently serves best as a teaching archive for the foundational and intermediate units rather than a complete implementation archive for the full syllabus.

Syllabus-only units that appear in the official document but are not directly represented here with clear lecture notebooks:

- Unit IX: Large Language Models and pre-trained models
- Unit X: Advanced LLMs and Retrieval-Augmented Generation
- Unit XI: Voice AI, text-to-speech, and future trends
- Unit XII: Building and scaling AI solutions
- Unit XIII: AI research and innovation

## Practical Orientation of the Materials

The contents of this folder are not only theoretical. The notebooks consistently emphasize implementation and hands-on learning. Across the units, students are expected to learn by writing and running code for:

- Data handling and visualization
- Regression and classification workflows
- Supervised learning with scikit-learn
- Clustering and PCA
- Neural network training from scratch
- PyTorch-based deep learning workflows
- CNN image classification and transfer learning
- Transformer-based sentiment analysis

This matches the practical emphasis of the syllabus, which highlights tutorial work, implementation practice, and model building using standard AI and ML libraries.

## Tools and Libraries Used

The notebooks in this repository make consistent use of the following tools and frameworks:

- Python
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- PyTorch
- torchvision
- Hugging Face transformers

The syllabus also mentions TensorFlow and broader deployment-focused tools, but the notebook materials in this folder are more strongly centered on scikit-learn and PyTorch-based implementations.

## How to Read This Repository

The notebooks are arranged in a mostly progressive learning order. A good reading path is:

1. Start with Python foundations and data structures.
2. Move into NumPy, pandas, and visualization.
3. Study the math notebooks before the main ML units.
4. Work through preprocessing, regression, classification, and evaluation.
5. Continue with unsupervised learning and dimensionality reduction.
6. Study deep learning theory before moving into PyTorch and CNNs.
7. Finish with sequence models and transformer-based sentiment analysis.

## Summary

This repository is a practical course companion for the AI/ML syllabus, with strong and well-documented notebook coverage of Units I to VIII. It provides a complete learning path from Python basics to transformer-based sentiment analysis, and it includes both conceptual explanation and implementation practice across the major topics actually taught in the current materials.
