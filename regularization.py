import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """
    Preprocesses the input data by scaling features and splitting into training and test sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    return [X_train, X_test, y_train, y_test]


def get_regression_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the diabetes dataset for regression tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)


def get_classification_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the breast cancer dataset for classification tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a linear regression model on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a ridge regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best ridge regression model found by GridSearchCV.
    """
    ridge = Ridge()
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid)
    grid_search.fit(X, y)

    return grid_search.best_estimator_


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a lasso regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best lasso regression model found by GridSearchCV.
    """
    lasso = Lasso(max_iter=10000)
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid)
    grid_search.fit(X, y)

    return grid_search.best_estimator_


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model without regularization on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X, y)
    return model


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L2 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L2 regularization found by GridSearchCV.
    """
    model = LogisticRegression(penalty='l2')
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_search.fit(X, y)

    return grid_search.best_estimator_


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L1 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L1 regularization found by GridSearchCV.
    """
    model = LogisticRegression(penalty='l1', solver='liblinear')
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_search.fit(X, y)

    return grid_search.best_estimator_


def compare_models():
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = get_regression_data()
    X_train_class, X_test_class, y_train_class, y_test_class = get_classification_data()

    regression_results = {}
    classification_results = {}

    models_regression = {
        'Linear Regression': linear_regression,
        'Ridge Regression': ridge_regression,
        'Lasso Regression': lasso_regression,
    }

    for model_name, model_func in models_regression.items():
        model = model_func(X_train_reg, y_train_reg)
        y_pred_reg = model.predict(X_test_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        regression_results[model_name] = {'R^2': r2, 'MSE': mse}

    models_classification = {
        'Logistic Regression': logistic_regression,
        'Logistic L2 Regression': logistic_l2_regression,
        'Logistic L1 Regression': logistic_l1_regression,
    }

    for model_name, model_func in models_classification.items():
        model = model_func(X_train_class, y_train_class)
        y_pred_class = model.predict(X_test_class)
        accuracy = accuracy_score(y_test_class, y_pred_class)
        classification_results[model_name] = {'Accuracy': accuracy}

    print("Regression:")
    for model_name, results in regression_results.items():
        print(f"{model_name}: R^2 = {results['R^2']:.4f}, MSE = {results['MSE']:.4f}")

    print("\nClasification:")
    for model_name, results in classification_results.items():
        print(f"{model_name}: Accuracy = {results['Accuracy']:.4f}")


compare_models()
