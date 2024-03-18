import pandas as pd

class PeptideExtractor:
    """
    A class for extracting peptide sequences from a fasta file of glycoproteins
    for scanning N-linked glycosylation sites.
    """
    
    def __init__(self, fasta_file, pos_file, windows=15, empty_aa="-", site='K'):
        """
        Initializes the PeptideExtractor with file paths and extraction parameters.

        Args:
            fasta_file (str): Path to the fasta file containing glycoprotein sequences.
            pos_file (str): Path to the file containing positions of glycosylation sites.
            windows (int, optional): Number of amino acids to include on each side of the glycosylation site. Defaults to 15.
            empty_aa (str, optional): Amino acid to use for padding when sequence is shorter than window size. Defaults to "-".
            site (str, optional): Amino acid representing the glycosylation site. Defaults to 'K'.
        """
        self.fasta_file = fasta_file
        self.pos_file = pos_file
        self.windows = windows
        self.empty_aa = empty_aa
        self.site = site

    def read_fasta(self):
        """
        Reads sequences from a fasta file and positions of glycosylation sites from a CSV file.

        Returns:
            tuple: A tuple containing dictionaries for sequences and positive positions, and a list of sequence IDs.
        """
        try:
            with open(self.fasta_file, 'r') as fp:
                lines = fp.readlines()
        except IOError:
            print(f'Cannot open {self.fasta_file}, check if it exists!')
            exit()

        post_prot = pd.read_csv(self.pos_file, index_col=0)
        
        fasta_dict, positive_dict, idlist = {}, {}, []
        gene_id, seq = "", ""
        
        for line in lines:
            if line.startswith('>'):
                if gene_id:
                    fasta_dict[gene_id] = seq
                    idlist.append(gene_id)
                gene_id, seq = line.strip(), ""
            else:
                seq += line.strip().replace(' ', '')
        
        if gene_id:
            fasta_dict[gene_id] = seq
            idlist.append(gene_id)

        for gene_id, sequence in fasta_dict.items():
            positions = post_prot.loc[gene_id.replace('>', ''), :].values.flatten()
            positive_dict[gene_id] = [int(pos) - 1 for pos in positions]
            fasta_dict[gene_id] = sequence.replace('#', '')

        return fasta_dict, positive_dict, idlist

    def get_peptide(self, fasta_dict, positive_dict, idlist):
        """
        Extracts peptide sequences based on the specified window size and glycosylation site.

        Args:
            fasta_dict (dict): Dictionary containing sequences indexed by gene ID.
            positive_dict (dict): Dictionary containing lists of positive positions indexed by gene ID.
            idlist (list): List of gene IDs.

        Returns:
            DataFrame: A DataFrame containing the extracted peptide sequences and their corresponding labels.
        """
        peptides = []
        
        for gene_id in idlist:
            seq = fasta_dict[gene_id]
            positive_list = positive_dict.get(gene_id, [])
            
            for pos, mid_aa in enumerate(seq):
                if mid_aa not in self.site:
                    continue
                
                start = max(0, pos - self.windows)
                end = min(len(seq), pos + self.windows + 1)
                
                left_seq = (self.empty_aa * (self.windows - (pos - start))) + seq[start:pos]
                right_seq = seq[pos + 1:end] + (self.empty_aa * (self.windows - (end - pos - 1)))
                
                final_seq = left_seq + mid_aa + right_seq
                label = 1 if pos in positive_list else 0
                
                peptides.append([label] + [aa for aa in final_seq])
        
        peptide_df = pd.DataFrame(peptides, columns=['Label'] + [f'Pos_{i}' for i in range(1, 2 * self.windows + 2)])
        
        return peptide_df

    def extract_peptides_for_training(self):
        """
        Facilitates the entire process of reading fasta and position files, and extracting peptide sequences.

        Returns:
            DataFrame: A DataFrame containing the extracted peptide sequences and their corresponding labels for training.
        """
        fasta_dict, positive_dict, idlist = self.read_fasta()
        peptide_df = self.get_peptide(fasta_dict, positive_dict, idlist)
        return peptide_df


if __name__ == "__main__":
    extractor = PeptideExtractor()
    # peptide_df = extractor.extract_peptides_for_training()
    # print(peptide_df.head())