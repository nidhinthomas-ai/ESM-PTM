import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
import argparse

from ExtractPeptide import PeptideExtractor
from Embedding import ProteinEmbeddingExtractor
from SplitDataset import DatasetSplitter
from CNNModel import CNNModel

def main(args):
    # Check for device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate the existence of necessary files
    fasta_path = os.path.join(args.benchmarks_dir, f'{args.benchmark_name}.fasta')
    pos_path = os.path.join(args.benchmarks_dir, f'{args.benchmark_name}_pos.csv')
    if not os.path.exists(fasta_path) or not os.path.exists(pos_path):
        raise IOError(f"Required data files are missing in {args.benchmarks_dir}")

    embedding_extractor = ProteinEmbeddingExtractor(args.plm)
    dataset_splitter = DatasetSplitter(args.plm, args.benchmarks_dir, args.benchmark_name, args.window_size, args.site)
    train_set, valid_set = dataset_splitter.split_dataset()

    X_train = torch.from_numpy(embedding_extractor.get_embedding(train_set, args.window_size)).float()
    Y_train = torch.from_numpy(train_set['label'].values).float().to(device)
    X_valid = torch.from_numpy(embedding_extractor.get_embedding(valid_set, args.window_size)).float()
    Y_valid = torch.from_numpy(valid_set['label'].values).float().to(device)
    
    X_train, X_valid = X_train.to(device), X_valid.to(device)

    # Convert numpy arrays to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    Y_train_torch = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    X_valid_torch = torch.tensor(X_valid, dtype=torch.float32)
    Y_valid_torch = torch.tensor(Y_valid, dtype=torch.float32).view(-1, 1)

    # Ensure tensors are correctly shaped for PyTorch [batch_size, channels, seq_length]
    X_train_torch = X_train_torch.permute(0, 2, 1)
    X_valid_torch = X_valid_torch.permute(0, 2, 1)

    # Define dataset and dataloader
    train_dataset = TensorDataset(X_train_torch, Y_train_torch)
    valid_dataset = TensorDataset(X_valid_torch, Y_valid_torch)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

    # Model initialization
    model = CNNModel(input_shape=X_train_torch.shape[1:]).to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    # Training loop
    for epoch in range(100):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the model
    torch.save(model.state_dict(), os.path.join(args.model_save_path, "ESM_PTM.pt"))
    print(f'Model saved to {os.path.join(args.model_save_path, "ESM_PTM.pt")}')

    # Validation
    model.eval()
    valid_predictions = []
    valid_targets = []
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.sigmoid().round()  # Assuming binary classification
            valid_predictions.extend(predictions.cpu().numpy())
            valid_targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(np.round(valid_predictions), valid_targets)
    print(f'Validation Accuracy: {accuracy}')

    recall = recall_score(np.round(valid_predictions), valid_targets)
    precision = precision_score(np.round(valid_predictions), valid_targets)
    auc = roc_auc_score(np.round(valid_predictions), valid_targets)
    mcc = matthews_corrcoef(np.round(valid_predictions), valid_targets)

    metrics = {
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'AUC': auc,
        'MCC': mcc
    }

    # Save results to CSV
    results_df = pd.DataFrame([metrics], index=['testing_results'])
    results_path = os.path.join('validation_results.csv')
    results_df.to_csv(results_path)
    print(f'Results saved to {results_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and validate a CNN model for protein sequence classification.')
    parser.add_argument('--benchmarks_dir', type=str, default='datasets/', help='Directory containing the datasets')
    parser.add_argument('--benchmark_name', type=str, default='N_gly', help='Name of the benchmark dataset')
    parser.add_argument('--site', type=str, default='N', help='PTM site residues')
    parser.add_argument('--window_size', type=int, default=12, help='Window size for residues surrounding the PTM site')
    parser.add_argument('--plm', type=str, default='esm1v_t33_650M_UR90S_1', help='Protein language model used for embeddings')
    parser.add_argument('--model_save_path', type=str, default='models/', help='Path to save the trained model')
    args = parser.parse_args()

    main(args)
