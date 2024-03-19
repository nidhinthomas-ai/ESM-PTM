import os
import pandas as pd
import re

from ExtractPeptide import PeptideExtractor

class DatasetSplitter:
    """
    A class to handle splitting and balancing of peptide sequence datasets.
    """

    def __init__(self, model_name, benchmarks_dir, benchmark_name, window_size, site):
        """
        Initializes the DatasetSplitter with necessary parameters.
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
        """
        fasta_path = os.path.join(self.benchmarks_dir, f'{self.benchmark_name}.fasta')
        pos_path = os.path.join(self.benchmarks_dir, f'{self.benchmark_name}_pos.csv')
        peptide_extractor = PeptideExtractor(fasta_path, pos_path, self.window_size, '-', self.site)
        train_frag, ids, poses, site_list = peptide_extractor.extract_peptides_for_training()

        # Adjust the DataFrame creation to work with the extracted data
        df = pd.DataFrame({
            'seq': [''.join(seq[1:]) for seq in train_frag.values],
            'label': train_frag[0].values,
            'sid': [f"{ids[i].replace('>', '')}_{poses[i]+1}" for i in range(len(ids))]
        })

        df['seq'] = df['seq'].apply(lambda x: " ".join(x) if self.space else x.replace(' ', ''))
        df['seq'] = df['seq'].apply(lambda x: re.sub(r"[UZOBX]", self.unknown_chr, x))

        return self.balanced_subsample(df) if balanced else df

    def split_dataset(self):
        """
        Splits the dataset into training and validation sets.
        """
        ds = self.get_data_set(balanced=True)
        valid_path = os.path.join(self.benchmarks_dir, f'{self.benchmark_name}_valid.fasta')

        if not os.path.exists(valid_path):
            split_idx = int(len(ds) * 0.9)
            train_set, valid_set = ds[:split_idx], ds[split_idx:]
        else:
            train_set = ds
            valid_set = self.get_data_set(balanced=True)

        return train_set, valid_set