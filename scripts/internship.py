import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, random_state=42)

# Define models and their search spaces
models = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': Integer(10, 200),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
            'criterion': Categorical(['gini', 'entropy'])
        }
    },
    'SVM': {
        'model': SVC(random_state=42),
        'params': {
            'C': Real(1e-3, 1e3, prior='log-uniform'),
            'gamma': Real(1e-3, 1e3, prior='log-uniform'),
            'kernel': Categorical(['linear', 'rbf', 'poly'])
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': Integer(50, 200),
            'learning_rate': Real(1e-3, 1, prior='log-uniform'),
            'max_depth': Integer(3, 10),
            'min_samples_split': Integer(2, 10)
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': Integer(3, 20),
            'weights': Categorical(['uniform', 'distance']),
            'p': Integer(1, 2)  # 1: manhattan, 2: euclidean
        }
    }
}

# Perform Bayesian Optimization for each model
results = {}
best_score = -np.inf
best_model_name = None
best_model = None

for model_name, config in models.items():
    print(f"\nOptimizing {model_name}...")
    opt = BayesSearchCV(
        estimator=config['model'],
        search_spaces=config['params'],
        n_iter=50,  # number of Bayesian optimization iterations
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    opt.fit(X, y)
    results[model_name] = {
        'best_score': opt.best_score_,
        'best_params': opt.best_params_,
        'optimizer': opt
    }
    
    if opt.best_score_ > best_score:
        best_score = opt.best_score_
        best_model_name = model_name
        best_model = opt.best_estimator_

# Visualization
plt.figure(figsize=(12, 6))

# Model comparison bar plot
scores = [results[name]['best_score'] for name in results]
model_names = list(results.keys())

plt.subplot(1, 2, 1)
bars = plt.bar(model_names, scores, color=['blue', 'green', 'red', 'purple'])
plt.title('Model Comparison (Cross-Validation Accuracy)')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom')

# Convergence plot for the best model
plt.subplot(1, 2, 2)
plot_convergence(results[best_model_name]['optimizer'].optimizer_results_[0])
plt.title(f'Convergence Plot for {best_model_name}')

plt.tight_layout()
plt.show()

# Print results
print("\n=== Final Results ===")
for model_name in results:
    print(f"\n{model_name}:")
    print(f"Best Accuracy: {results[model_name]['best_score']:.4f}")
    print("Best Parameters:")
    for param, value in results[model_name]['best_params'].items():
        print(f"  {param}: {value}")

print(f"\nBest Model: {best_model_name} with accuracy {best_score:.4f}")# File: automated_ml_optimization.py
