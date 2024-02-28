import random

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, \
    LogisticRegressionCV, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

# from src.utils.dnn_models import DNNClassifier1, DNNClassifier2


def reset_setup(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.set_visible_devices([], 'GPU')


def create_models(global_random_seed):
    return {
        # Linear Models
        'LogisticRegression': LogisticRegression(random_state=global_random_seed, max_iter=1000),
        'Ridge': RidgeClassifier(random_state=global_random_seed),
        'SGD': SGDClassifier(random_state=global_random_seed),
        'Perceptron': Perceptron(random_state=global_random_seed),
        'LogisticRegressionCV': LogisticRegressionCV(),

        # Decision Trees
        'DT': DecisionTreeClassifier(random_state=global_random_seed),
        'ExtraTree': ExtraTreeClassifier(random_state=global_random_seed),

        # Nearest Neighbors
        'KNN': KNeighborsClassifier(),
        'RadiusNeighbors': RadiusNeighborsClassifier(outlier_label='most_frequent'),

        # Support Vector Machines
        'SVM': SVC(random_state=global_random_seed),
        'NuSVC': NuSVC(random_state=global_random_seed, kernel='rbf', nu=0.01),
        'LinearSVC': LinearSVC(random_state=global_random_seed, max_iter=10000),

        # Naive Bayes
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'ComplementNB': ComplementNB(),
        'BernoulliNB': BernoulliNB(),

        # Gaussian Processes
        'GaussianProcess': GaussianProcessClassifier(random_state=global_random_seed),

        # Ensemble Methods
        'RF': RandomForestClassifier(random_state=global_random_seed),
        'ExtraTrees': ExtraTreesClassifier(random_state=global_random_seed),
        'AdaBoost': AdaBoostClassifier(random_state=global_random_seed, algorithm="SAMME"),
        'GradientBoosting': GradientBoostingClassifier(random_state=global_random_seed),
        'Bagging': BaggingClassifier(random_state=global_random_seed),
        # 'Voting': VotingClassifier(estimators=[('DT', DecisionTreeClassifier(random_state=global_random_seed)), ('SGD', SGDClassifier(loss="modified_huber", random_state=global_random_seed))], voting='soft'),

        # Neural Networks
        'MLP': MLPClassifier(random_state=global_random_seed),

        # DNN
        # 'DNN1v0': DNNClassifier1(),
        # 'DNN2v0': DNNClassifier2(),
    }
