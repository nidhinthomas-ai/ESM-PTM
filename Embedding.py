import os
import numpy as np
import torch
from esm import Alphabet, pretrained
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class ProteinEmbeddingExtractor:
    """
    A class to extract protein sequence embeddings using the ESM language model.
    """

    def __init__(self, model_name='esm1v_t33_650M_UR90S_1'):
        """
        Initializes the ProteinEmbeddingExtractor with the specified ESM model.

        Args:
            model_name (str, optional): The name of the pre-trained ESM model to use.
                                        Defaults to 'esm1v_t33_650M_UR90S_1'.
        """
        self.model_name = model_name
        self.model, self.alphabet = pretrained.load_model_and_alphabet(self.model_name)

    def return_ds_list(self, dataset):
        """
        Converts the dataset into a list of tuples containing sequence IDs and sequences.

        Args:
            dataset (dict): A dictionary containing 'seq' and 'sid' keys with sequences and sequence IDs.

        Returns:
            list: A list of tuples with sequence IDs and sequences.
        """
        ds_list = [(str(dataset['sid'][i]), str(dataset['seq'][i])) for i in range(len(dataset['seq']))]
        return ds_list

    def get_embedding(self, dataset, window):
        """
        Generates embeddings for the given dataset using the specified ESM model.

        Args:
            dataset (dict): A dictionary containing 'seq' and 'sid' keys with sequences and sequence IDs.
            window (int): The window size to consider around each sequence.

        Returns:
            numpy.ndarray: An array of embeddings for the given sequences.
        """
        batch_converter = self.alphabet.get_batch_converter()
        data_list = self.return_ds_list(dataset)
        _, _, tokens = batch_converter(data_list)

        batch_tokens = tokens[:, :window * 2 + 3]
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
        token_representations = results['representations'][33]
        embeddings = token_representations.cpu().detach().numpy()
        embeddings = embeddings[:, 1:window * 2 + 2, :]
        return embeddings