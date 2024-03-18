import argparse
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, precision_score, recall_score
from SplitDataset import get_data_set
from Embedding import embedding  # Ensure this is adapted for PyTorch
from torch.utils.data import DataLoader, TensorDataset

class Predictor:
    """
    A class to predict PTM site using a trained CNN model with PyTorch and protein language embeddings.
    """

    def __init__(self, benchmarks_dir, benchmark_name, site, window_size, plm, model_path, result_path):
        """
        Initializes the predictor with configuration parameters.

        Args:
            benchmarks_dir (str): Directory containing the datasets.
            benchmark_name (str): Name of the benchmark dataset.
            site (str): PTM site residues.
            window_size (int): Number of residues surrounding the PTM site.
            plm (str): Protein language model used for embeddings.
            model_path (str): Path to the trained model.
            result_path (str): Directory to save the prediction results.
        """
        self.benchmarks_dir = benchmarks_dir
        self.benchmark_name = benchmark_name
        self.site = site
        self.window_size = window_size
        self.plm = plm.upper()
        self.model_path = model_path
        self.result_path = result_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_class):
        """
        Loads the trained CNN model using PyTorch.

        Args:
            model_class (class): The class of the model to load.
        """
        self.model = model_class().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def perform_prediction(self, test_set):
        """
        Performs prediction on the test set and calculates performance metrics.

        Args:
            test_set (pd.DataFrame): Test dataset.
        """
        # Convert to PyTorch dataset and loader
        X_test, Y_test = embedding(self.plm, test_set, self.window_size)
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float).to(self.device), 
                                  torch.tensor(Y_test, dtype=torch.long).to(self.device))
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        # Perform prediction
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = self.calculate_metrics(np.array(all_labels), np.array(all_preds))
        self.save_results(metrics)

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculates performance metrics based on true labels and predictions.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }

    def save_results(self, metrics):
        """
        Saves the prediction results to a CSV file.

        Args:
            metrics (dict): A dictionary containing performance metrics.
        """
        results_df = pd.DataFrame([metrics], index=['testing_results'])
        results_path = os.path.join(self.result_path, 'testing_results.csv')
        results_df.to_csv(results_path)
        print(f'Results saved to {results_path}')

def main():
    parser = argparse.ArgumentParser(description='ESM-PTM: A tool for PTM site prediction using Protein Language Models and CNN with PyTorch')
    parser.add_argument('--BENCHMARKS_DIR', type=str, default='datasets/', help='Directory containing the datasets')
    parser.add_argument('--benchmark_name', type=str, default='N_gly', help='Name of the benchmark dataset')
    parser.add_argument('--site', default='N', type=str, help="PTM site residues")
    parser.add_argument('--w', default=12, type=int, help='Window size for the residues surrounding the PTM site')
    parser.add_argument('--PLM', default='ESM-1B', type=str, help='Protein language model used for embeddings')
    parser.add_argument('--model_path', default='models/model.pt', type=str, help='Path to the trained model')
    parser.add_argument('--result_path', default='results/', type=str, help='Directory to save the prediction results')
    args = parser.parse_args()

    # Initialize Predictor
    predictor = Predictor(args.BENCHMARKS_DIR, args.benchmark_name, args.site, args.w, args.PLM, args.model_path, args.result_path)
    
    # Assuming CNNModel class exists and matches the saved model architecture
    from CNNModel import CNNModel

    # Load and test model
    predictor.load_model(CNNModel)
    test_set = get_data_set(args.PLM, args.BENCHMARKS_DIR, args.benchmark_name + '_test', args.w, args.site, balanced=1)
    predictor.perform_prediction(test_set)

if __name__ == '__main__':
    main()
