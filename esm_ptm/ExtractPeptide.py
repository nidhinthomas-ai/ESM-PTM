import pandas as pd

class PeptideExtractor:
    """
    A class for extracting peptide sequences for training, focused on specific amino acid sites.
    """
    
    def __init__(self, fasta_file, pos_file, window_size=15, empty_aa="-", site='N'):
        """
        Initializes the PeptideExtractor with necessary parameters for peptide extraction.
        """
        self.fasta_file = fasta_file
        self.pos_file = pos_file
        self.window_size = window_size
        self.empty_aa = empty_aa
        self.site = site

    def read_fasta_and_positions(self):
        """
        Reads sequences from a fasta file and positions of sites from a CSV file.
        """
        try:
            with open(self.fasta_file) as fp:
                lines = fp.readlines()
        except IOError:
            print(f'Cannot open {self.fasta_file}. Please check if the file exists.')
            exit()

        post_prot = pd.read_csv(self.pos_file, index_col=0)
        
        fasta_dict = {}
        positive_dict = {}
        idlist = []
        gene_id = ""
        seq = ""

        for line in lines:
            if line.startswith('>'):
                if gene_id:
                    fasta_dict[gene_id] = seq
                    idlist.append(gene_id)
                gene_id = line.strip()
                seq = ""
            else:
                seq += line.strip().replace(' ', '')
        if gene_id:
            fasta_dict[gene_id] = seq
            idlist.append(gene_id)
        
        for gene_id in fasta_dict.keys():
            posnum = 0
            if gene_id.replace('>', '') in post_prot.index:
                for i in post_prot.loc[gene_id.replace('>', '')].values.flatten():
                    if posnum == 0:
                        positive_dict[gene_id] = [int(i)-1]
                    else:
                        positive_dict[gene_id].append(int(i)-1)
                    posnum += 1
            fasta_dict[gene_id] = fasta_dict[gene_id].replace('#', '')
        
        return fasta_dict, positive_dict, idlist

    def extract_peptides(self, fasta_dict, positive_dict, idlist):
        """
        Extracts peptide sequences based on the specified window size and site.
        """
        peptides = []
        ids = []
        poses = []
        site_list = []
        
        for gene_id in idlist:
            seq = fasta_dict[gene_id]
            positive_list = positive_dict.get(gene_id, [])
            
            for pos, mid_aa in enumerate(seq):
                if mid_aa not in self.site:
                    continue
                
                start = max(0, pos - self.window_size)
                end = min(len(seq), pos + self.window_size + 1)
                
                left_seq = self.empty_aa * (self.window_size - (pos - start)) + seq[start:pos]
                right_seq = seq[pos + 1:end] + self.empty_aa * (self.window_size - (end - pos - 1))
                
                final_seq = [1 if pos in positive_list else 0] + [aa for aa in left_seq + mid_aa + right_seq]
                
                peptides.append(final_seq)
                ids.append(gene_id)
                poses.append(pos)
                site_list.append(mid_aa)
        
        peptide_df = pd.DataFrame(peptides)
        return peptide_df, ids, poses, site_list

    def extract_peptides_for_training(self):
        """
        Orchestrates the extraction process, returning the final DataFrame for training.
        """
        fasta_dict, positive_dict, idlist = self.read_fasta_and_positions()
        peptides, ids, poses, site_list = self.extract_peptides(fasta_dict, positive_dict, idlist)
        return peptides, ids, poses, site_list