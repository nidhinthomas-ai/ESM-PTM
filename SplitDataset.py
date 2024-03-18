import os
import pandas as pd
import numpy as np
import re
from ExtractPeptide import ExtractPeptideforTraining


class DatasetSplitter:
    """
    A class to handle splitting and balancing of peptide sequence datasets.
    """

    def __init__(self, model_name, benchmarks_dir, benchmark_name, window_size, site):
        """
        Initializes the DatasetSplitter with necessary parameters.

        Args:
            model_name (str): Name of the protein language model to use for embedding.
            benchmarks_dir (str): Directory path where benchmark datasets are stored.
            benchmark_name (str): Name of the specific benchmark dataset.
            window_size (int): Window size around the glycosylation site for peptide extraction.
            site (str): Specific amino acid or site to consider for peptide extraction.
        """
        self.model_name = model_name
        self.benchmarks_dir = benchmarks_dir
        self.benchmark_name = benchmark_name
        self.window_size = window_size
        self.site = site
        self.unknown_chr = '<unk>' if model_name == 'prot_xlnet' else 'X'
        self.space = True if model_name.startswith('prot') else False

    def balanced_subsample(self, df):
        """
        Balances the dataset to have an equal number of positive and negative samples.

        Args:
            df (pd.DataFrame): The dataset to balance.

        Returns:
            pd.DataFrame: The balanced dataset.
        """
        train_pos = df[df['label'] == 1]
        train_neg = df[df['label'] != 1]

        min_length = min(len(train_pos), len(train_neg))
        train_pos_s = train_pos.sample(min_length)
        train_neg_s = train_neg.sample(min_length)

        df_balanced = pd.concat([train_pos_s, train_neg_s])
        return df_balanced.reset_index(drop=True)

    def get_data_set(self, balanced=True):
        """
        Prepares the dataset by extracting peptides and optionally balancing it.

        Args:
            balanced (bool, optional): Whether to balance the dataset. Defaults to True.

        Returns:
            pd.DataFrame: The prepared dataset.
        """
        fasta_path = os.path.join(self.benchmarks_dir, f'{self.benchmark_name}.fasta')
        pos_path = os.path.join(self.benchmarks_dir, f'{self.benchmark_name}_pos.csv')
        train_frag, ids, poses, _ = ExtractPeptideforTraining(fasta_path, pos_path,
                                                              self.window_size, 'X',
                                                              site=self.site)
        train_frag = train_frag.dropna().drop_duplicates()

        df = pd.DataFrame({
            'seq': [''.join(train_frag.iloc[i, 1:self.window_size * 2 + 2]) for i in range(len(train_frag) - 1)],
            'label': train_frag.iloc[:-1, 0].tolist(),
            'sid': [(ids[i].replace('>', '')) + '_' + str(poses[i] + 1) for i in range(len(ids) - 1)]
        })

        df['seq'] = df['seq'].apply(lambda x: " ".join(x) if self.space else x.replace(' ', ''))
        df['seq'] = df['seq'].apply(lambda x: re.sub(r"[UZOBX]", self.unknown_chr, x))

        return self.balanced_subsample(df) if balanced else df

    def split_dataset(self):
        """
        Splits the dataset into training and validation sets.

        Returns:
            tuple: A tuple containing the training and validation datasets.
        """
        ds = self.get_data_set(balanced=True)
        valid_path = os.path.join(self.benchmarks_dir, self.benchmark_name + '_valid.fasta')

        if not os.path.exists(valid_path):
            split_idx = int(len(ds) * 0.9)
            train_set, valid_set = ds[:split_idx], ds[split_idx:]
        else:
            train_set = ds
            valid_set = self.get_data_set(balanced=True)

        return train_set, valid_set


if __name__ == "__main__":
    # Example usage
    splitter = DatasetSplitter(model_name='esm1v_t33_650M_UR90S_1',
                               benchmarks_dir='/path/to/benchmarks',
                               benchmark_name='example_benchmark',
                               window_size=15,
                               site='N')
    train_set, valid_set = splitter.split_dataset()
    print(train_set.head(), valid_set.head())
