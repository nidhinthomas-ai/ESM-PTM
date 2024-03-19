import argparse
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset

from ExtractPeptide import PeptideExtractor
from Embedding import ProteinEmbeddingExtractor
from SplitDataset import DatasetSplitter
from CNNModel import CNNModel

class Predictor:
    def __init__(self, benchmarks_dir, benchmark_name, site, window_size, plm, model_path, result_path):
        self.benchmarks_dir = benchmarks_dir
        self.benchmark_name = benchmark_name
        self.site = site
        self.window_size = window_size
        self.plm = plm.upper()
        self.model_path = model_path
        self.result_path = result_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_class):
        self.model = model_class
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def perform_prediction(self, test_loader):

        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs).round()  # Adjust based on your model's output
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = self.calculate_metrics(np.array(all_labels), np.array(all_preds))
        self.save_results(metrics)

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }

    def save_results(self, metrics):
        results_df = pd.DataFrame([metrics], index=['testing_results'])
        results_path = os.path.join(self.result_path, 'testing_results.csv')
        results_df.to_csv(results_path)
        print(f'Results saved to {results_path}')

def main():
    parser = argparse.ArgumentParser(description='Predict PTM sites using Protein Language Models and CNN with PyTorch')
    parser.add_argument('--benchmarks_dir', type=str, required=True, help='Directory containing the datasets')
    parser.add_argument('--benchmark_name', type=str, required=True, help='Name of the benchmark dataset')
    parser.add_argument('--site', type=str, required=True, help="PTM site residues")
    parser.add_argument('--window_size', type=int, required=True, help='Window size for the residues surrounding the PTM site')
    parser.add_argument('--plm', type=str, required=True, help='Protein language model used for embeddings')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--result_path', type=str, required=True, help='Directory to save the prediction results')
    args = parser.parse_args()

    # Initialize Predictor
    predictor = Predictor(args.benchmarks_dir, args.benchmark_name, args.site, args.window_size, args.plm, args.model_path, args.result_path)

    # Load the CNN model
    predictor.load_model(CNNModel)

    embedding = ProteinEmbeddingExtractor(args.plm)
    dataset_splitter = DatasetSplitter(args.plm, args.benchmarks_dir, args.benchmark_name, args.window_size, args.site)
    test_set = dataset_splitter.get_data_set() 

    X_test = torch.from_numpy(embedding.get_embedding(test_set, args.window_size)).float().to(predictor.device)
    Y_test = torch.from_numpy(test_set['label'].values).float().to(predictor.device)

    # Convert numpy arrays to PyTorch tensors
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    Y_test_torch = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    # Assuming X_train is your input tensor
    X_test_torch = X_test_torch.permute(0, 2, 1)

    # Define dataset and dataloader
    test_dataset = TensorDataset(X_test_torch, Y_test_torch)
    test_loader = DataLoader(dataset=test_dataset, batch_size=int(64), shuffle=True)

    # Perform prediction and save results
    predictor.perform_prediction(test_loader)

if __name__ == '__main__':
    main()

