import argparse
import os
from SplitDataset import split_dataset
from Embedding import ESMEmbedding
from CNNModel import CNNModel
import torch

class ESM_PTM_Trainer:
    """
    Trainer class for PTG-PLM, a tool for PTM site prediction using Protein Language Models and CNN.
    """
    
    def __init__(self, benchmarks_dir, benchmark_name, site, window_size, plm, config_file, model_save_path):
        """
        Initializes the trainer with configuration parameters.

        Args:
            benchmarks_dir (str): Directory path where benchmark datasets are stored.
            benchmark_name (str): Name of the benchmark dataset.
            site (str): PTM site residue(s); for more than one residue, format as ('X', 'Y').
            window_size (int): Number of residues surrounding the PTM residues.
            plm (str): Protein language model to use for embeddings.
            config_file (str): Path to the CNN parameters configuration file.
            model_save_path (str): Path to save the trained model.
        """
        self.benchmarks_dir = benchmarks_dir
        self.benchmark_name = benchmark_name
        self.site = site
        self.window_size = window_size
        self.plm = plm
        self.config_file = config_file
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        """
        Runs the training process with the specified configuration.
        """
        if not os.path.exists(os.path.join(self.benchmarks_dir, f'{self.benchmark_name}.fasta')):
            raise IOError(f'The protein sequences FASTA file does not exist: {os.path.join(self.benchmarks_dir, f"{self.benchmark_name}.fasta")}')
        
        if not os.path.exists(os.path.join(self.benchmarks_dir, f'{self.benchmark_name}_pos.csv')):
            raise IOError(f'The positive sites file does not exist: {os.path.join(self.benchmarks_dir, f"{self.benchmark_name}_pos.csv")}')
        
        if (2 * self.window_size + 1) % 2 == 0:
            print('The window size (2*w+1) must be odd!!')
            return
        
        # Initialize and get embeddings
        embedding = ESMEmbedding(self.plm)
        train_set, valid_set = split_dataset(self.benchmarks_dir, self.benchmark_name, self.window_size, self.site)
        X_train, Y_train = embedding.get_embeddings(train_set, self.window_size)
        X_valid, Y_valid = embedding.get_embeddings(valid_set, self.window_size)
        
        # Move tensors to the specified device
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        X_valid, Y_valid = X_valid.to(self.device), Y_valid.to(self.device)
        
        # Train the CNN model
        model = CNNModel(input_size=X_train.shape[1], config_file=self.config_file).to(self.device)
        model.train(X_train, Y_train, X_valid, Y_valid)
        
        model_path = os.path.join(self.model_save_path, f'PTG-PLM_{self.plm}.pt')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

def main():
    parser = argparse.ArgumentParser(description='ESM_PTM: A tool for PTM site prediction using ESM Protein Language Model and CNN')
    parser.add_argument('--BENCHMARKS_DIR', type=str, default='datasets/', help='Dataset path')
    parser.add_argument('--benchmark_name', type=str, default='N_gly', help='Dataset name')
    parser.add_argument('--site', default='N', type=str, help="PTM site residue(s) for more than one residue, format as ('X', 'Y')")
    parser.add_argument('--w', default=12, type=int, help='Number of residues surrounding the PTM residues')
    parser.add_argument('--PLM', default='ESM-1b', type=str, help='Protein language model to use for embeddings')
    parser.add_argument('--model_save_path', default='models/', type=str, help='Path to save the trained model')
    args = parser.parse_args()
    
    trainer = ESM_PTM_Trainer(args.BENCHMARKS_DIR, args.benchmark_name, args.site, args.w, args.PLM, args.config_file, args.model_save_path)
    trainer.run()

if __name__ == '__main__':
    main()
