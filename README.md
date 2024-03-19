# ESM-PTM: N-linked Glycosylation Site Prediction Using ESM Protein Language

## Summary
This project focuses on the prediction of N-linked glycosylation sites within proteins, leveraging a unique approach that combines evolutionary scale modeling (ESM) for peptide embedding and deep learning for classification. Given a protein amino acid sequence, the provided scripts analyze potential glycosylation sites, crucial for understanding protein function and for therapeutic applications.

## Pre-requisites

### Creating a Conda Environment:
To ensure compatibility and manage dependencies, it is recommended to create a new conda environment for this project:

```bash
conda create --name esm_ptm_env python=3.8
conda activate esm_ptm_env
# Install required packages listed in requirements.txt
pip install -r requirements.txt
```

### Downloading the Repository
Clone the repository from GitHub to your local machine:

```bash
git clone https://github.com/nidhinthomas-ai/ESM-PTM
cd ESM-PTM-main
```

## Introduction

### N-Linked Glycosylation:
N-linked glycosylation is a post-translational modification involving the attachment of carbohydrate moieties to nitrogen atoms on asparagine residues within proteins. This modification plays a vital role in protein folding, stability, and cell signaling processes. Understanding and predicting these sites can significantly impact the development of therapeutics. Typically N-linked glycolsylation requires the **Asn-Xxx-Thr/Ser (N-X-T/S)** motif in the protein. Xxx can be any amino acid except for **Pro**.

### Datasets:

The datasets and base model architecture used in this project are obtained from PTG-PLM GitHub repository (https://github.com/Alhasanalkuhlani/PTG-PLM).

### Repository Layout:

The repository is organized as follows:

ESM-PTM-main/  
├── esm_ptm/  
│   ├── CNNModel.py  
│   ├── Embedding.py  
│   ├── ExtractPeptide.py  
│   ├── SplitDataset.py  
│   ├── ML.py  
│   └── __init__()  
├── datasets/  
│   ├── N_Gly.fasta  
│   ├── N_Gly_pos.csv  
│   └── Rest of the dataset  
├── models/  
│   └── ESM_PTM.pt  
├── results/  
│   └── testing_results.csv  
├── predict.py  
├── train.py  
├── RESDME.md  
└── requirements.txt  

**esm_ptm**: Contains the main scripts for the project.  
**datasets**: Includes the datasets used for training and testing the model.
**models**: Stores the trained model files.
**results**: Stores the results from testing the model.

## Model Description

The project utilizes the ESM language model to obtain peptide embeddings, which are then fed into a deep learning model for the classification of N-linked glycosylation sites. The ESM model, pretrained on a vast corpus of protein sequences, captures the intricate patterns of protein language, providing a robust foundation for feature extraction.

## Obtaining Embeddings with ESM

The Embedding.py script is responsible for converting amino acid sequences into embeddings using the ESM model. These embeddings encapsulate the contextual information of each amino acid, enhancing the predictive performance of the subsequent classification model.

## Training

To train the model, use the train.py script with the following command line:

```bash
python train.py --benchmarks_dir=datasets/ --benchmark_name=N_gly --site=N --w=12 --plm=esm1v_t33_650M_UR90S_1 --model_save_path=models/
```

This script will train the model using the specified dataset and save the trained model at the provided path.

## Prediction

The trained model can be utilized for prediction using the predict.py script. Here is how to use it:

```bash
python predict.py --benchmarks_dir=datasets/ --benchmark_name=N_gly --site=N --w=12 --plm=esm1v_t33_650M_UR90S_1 --model_path=models/ESM_PTM.pt --result_path results/
```
The script takes the path to the trained model and the protein amino acid sequence file, outputting the predicted N-linked glycosylation sites.

For more detailed instructions on each script and additional options, refer to the respective script documentation within the repository.
