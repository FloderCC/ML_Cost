# ML Cost Framework (on develop)

## Dataset Features:
#### Before codifying the dataset:
- Class Imbalance Ratio
- Gini Impurity
- Entropy
- Number of samples
- Number of features
- Completeness (Percentage of missing values)
- Consistency (quantity of non contradictions within the data)
- Uniqueness (percentage of duplicate records)

#### After codifying the dataset:
- Redundancy (avg and std of the values in the correlation matrix)
- Maximum - minimum feature's value
- Global avg and std of the input features
- Avg and std of features' avg and std (for considering the Coefficient of Variation on the features)


## Classifiers Type Features:

### Tree-Based Classifiers:
- Decision Trees
- Random Forest
- Gradient Boosting Machines (GBM)
- XGBoost
- LightGBM
- CatBoost

### Linear Classifiers:
- Logistic Regression
- Support Vector Machines (SVM) with linear kernel
- Perceptron
- Linear Discriminant Analysis (LDA)

### Instance-Based Classifiers:
- k-Nearest Neighbors (k-NN)

### Naive Bayes Classifiers:
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

### Neural Network Classifiers:
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Transformers

### Ensemble Classifiers:
- Voting Classifier
- Bagging (e.g., BaggingClassifier)
- Boosting (e.g., AdaBoost, Gradient Boosting Machines)
- Stacking

### Rule-Based Classifiers:
- OneR
- JRip
- PART (Partial Decision Trees)

### Distance-Based Classifiers:
- k-Nearest Neighbors (k-NN)

### Probabilistic Classifiers:
- Logistic Regression
- Naive Bayes

### Support Vector Machines (SVM):
- SVM with different kernels: linear, polynomial, radial basis function (RBF), etc.
