import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset
import torch

class MLModels:
    """
    Class to handle machine learning models training and evaluation using scikit-learn and XGBoost.
    """

    def __init__(self, model_key='lr'):
        """
        Initializes the class with the model specified by the key.

        Args:
            model_key (str): Key to specify the machine learning model to be used.
        """
        self.models = self._get_models()
        self.model = self.models.get(model_key, LogisticRegression())

    def _get_models(self):
        """
        Private method to return a dictionary of machine learning models.

        Returns:
            dict: A dictionary of instantiated machine learning models.
        """
        models = {
            'lr': LogisticRegression(max_iter=1000),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', C=100, gamma='scale'),
            'xgb': XGBClassifier(booster='gbtree', eval_metric='logloss', use_label_encoder=False,
                                 learning_rate=0.1, max_depth=5, min_child_weight=5, subsample=0.9)
        }
        return models

    def train_and_evaluate(self, X_train, Y_train, X_test, Y_test, batch_size=64):
        """
        Trains the model on the training data and evaluates it on the test data.

        Args:
            X_train (torch.Tensor): Training features.
            Y_train (torch.Tensor): Training labels.
            X_test (torch.Tensor): Test features.
            Y_test (torch.Tensor): Test labels.
            batch_size (int): Batch size for loading data.
        """
        # Convert PyTorch tensors to NumPy arrays for compatibility with scikit-learn
        X_train_np = X_train.numpy().reshape(X_train.shape[0], -1)
        Y_train_np = Y_train.numpy()
        X_test_np = X_test.numpy().reshape(X_test.shape[0], -1)
        Y_test_np = Y_test.numpy()

        # Train the model
        self.model.fit(X_train_np, Y_train_np)

        # Predictions
        Y_pred = self.model.predict(X_test_np)

        # Performance metrics
        metrics = self.calculate_metrics(Y_test_np, Y_pred)
        self.display_metrics(metrics)

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculates and returns performance metrics.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Returns:
            dict: Dictionary containing performance metrics.
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        metrics['AUC'] = auc(fpr, tpr)
        return metrics

    @staticmethod
    def display_metrics(metrics):
        """
        Displays the performance metrics.

        Args:
            metrics (dict): Dictionary containing performance metrics.
        """
        for metric, value in metrics.items():
            print(f'{metric}: {value:.4f}')

# Example usage
if __name__ == "__main__":
    # Assuming X_train, Y_train, X_test, Y_test are available as PyTorch tensors
    # Example: X_train, Y_train, X_test, Y_test = load_your_data()

    model_trainer = MLModels(model_key='lr') 
    model_trainer.train_and_evaluate(X_train, Y_train, X_test, Y_test)
