# File: automated_ml_optimization.py
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and their hyperparameter spaces
models = {
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': Integer(10, 300),
            'max_depth': Integer(2, 30),
            'min_samples_split': Real(0.01, 0.99),
            'criterion': Categorical(['gini', 'entropy'])
        }
    },
    'SVM': {
        'model': SVC(),
        'params': {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'kernel': Categorical(['linear', 'rbf', 'poly'])
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(),
        'params': {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.01, 1.0),
            'max_depth': Integer(2, 10)
        }
    }
}

# Bayesian Optimization and Model Selection
best_score = 0
best_model = None

for model_name, config in models.items():
    print(f"\nOptimizing {model_name}...")
    
    # Bayesian Optimization
    opt = BayesSearchCV(
        estimator=config['model'],
        search_spaces=config['params'],
        n_iter=50,
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    
    opt.fit(X_train, y_train)
    
    # Evaluate
    test_score = opt.score(X_test, y_test)
    print(f"Best {model_name} params: {opt.best_params_}")
    print(f"Validation accuracy: {opt.best_score_:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Track best model
    if test_score > best_score:
        best_score = test_score
        best_model = opt.best_estimator_

# Final evaluation
print(f"\nBest model: {best_model.__class__.__name__}")
print(f"Final Test Accuracy: {best_score:.3f}")