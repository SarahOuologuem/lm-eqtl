import numpy as np
import torch
from torch.utils.data import Dataset
import pysam
from itertools import chain


class GenoDataset(Dataset):
    def __init__(self, seq_df, transform, dataset_path, seq_len, max_augm_shift=0, 
                 mode='train', regression=False):

        if dataset_path.endswith('.fa'):
            self.fasta = pysam.FastaFile(dataset_path)
        else:
            self.fasta = None

        self.seq_df = seq_df
        self.transform = transform
        self.max_augm_shift = max_augm_shift
        self.mode = mode
        self.seq_len = seq_len

        self.regression = regression

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        if self.fasta:
            seq = self.fasta.fetch(self.seq_df.iloc[idx].seq_name).upper()
        else:
            seq = self.seq_df.iloc[idx].seq.upper()
        shift = np.random.randint(self.max_augm_shift+1) #random shift at training, must be chunk_size-input_params.seq_len
        seq = seq[shift:shift+self.seq_len] #shift the sequence and limit its size
        seg_label = self.seq_df.iloc[idx].seg_label #label for segment-aware training

        #for given genotype, randomly choose a haplotype for training/testing
        if np.random.rand()>0.5:
            seq = seq.replace('-','').replace('B','A').replace('F','A').replace('M','R')
        else:
            seq = seq.replace('-','').replace('B','A').replace('M','A').replace('F','R')

        masked_sequence, target_labels_masked, target_labels, _, _ = self.transform(seq)
        masked_sequence = (masked_sequence, seg_label)

        if self.regression: 
            seq_expr = self.seq_df.iloc[idx].expr.astype(np.float32)
            return masked_sequence, target_labels_masked, target_labels, seq, seq_expr

        return masked_sequence, target_labels_masked, target_labels, seq

    def close(self):
        self.fasta.close()


class HaploDataset(Dataset):

    def __init__(self, seq_df, transform, dataset_path, seq_len, max_augm_shift=0, 
                 mode='train', regression=False):

        if dataset_path.endswith('.fa'):
            self.fasta = pysam.FastaFile(dataset_path)
        else:
            self.fasta = None

        self.seq_df = seq_df
        self.transform = transform
        self.max_augm_shift = max_augm_shift
        self.mode = mode
        self.seq_len = seq_len

        self.regression = regression

    def __len__(self):

        return len(self.seq_df) # times two because returns both haplotypes 

    def __getitem__(self, idx):

        if self.fasta:
            seq = self.fasta.fetch(self.seq_df.iloc[idx].seq_name).upper()
        else:
            seq = self.seq_df.iloc[idx].seq.upper()

        shift = np.random.randint(self.max_augm_shift+1) #random shift at training, must be chunk_size-input_params.seq_len

        seq = seq[shift:shift+self.seq_len] #shift the sequence and limit its size
        seg_label = self.seq_df.iloc[idx].seg_label #label for segment-aware training
        #'''
        seq1 = seq.replace('-','').replace('B','A').replace('F','A').replace('M','R') # father
        seq2 = seq.replace('-','').replace('B','A').replace('M','A').replace('F','R') # mother 

        masked_sequence1, target_labels_masked1, target_labels1, _, _ = self.transform(seq1)
        masked_sequence2, target_labels_masked2, target_labels2, _, _ = self.transform(seq2)

        masked_sequence = torch.vstack((masked_sequence1, masked_sequence2))
        seg_label = torch.vstack((torch.tensor(seg_label), torch.tensor(seg_label)))
        masked_sequence = (masked_sequence, seg_label)

        target_labels_masked = torch.vstack((target_labels_masked1, target_labels_masked2))
        target_labels = torch.vstack((target_labels1, target_labels2))
        seq = (seq1, seq2)

        if self.regression: 
            seq_expr = self.seq_df.iloc[idx].expr.astype(np.float32)
            return masked_sequence, target_labels_masked, target_labels, seq, seq_expr

        return masked_sequence, target_labels_masked, target_labels, seq
        

    def close(self):
        self.fasta.close()

class ExpressionCollator(object): 
    def __init__(self, haplotypes=False): 
        self.haplotypes = haplotypes

    def __call__(self, data): 
        return self.collate_fn(data)

    def collate_fn(self, data):
        """collate fn that adds expression values for each sequence.
        """ 
        #masked sequence
        if self.haplotypes:
            masked_sequence = [x[0][0] for x in data]
            masked_sequence = [torch.stack(torch.split(d, 3)) for d in masked_sequence] 
            masked_sequence = torch.concat(masked_sequence)
            #seg labels
            seg_labels = [x[0][1] for x in data]
            seg_labels = torch.concat(seg_labels).flatten()
            # target labels masked
            target_labels_masked = [x[1] for x in data]
            target_labels_masked = torch.concat(target_labels_masked)
            # target labels 
            target_labels = [x[2] for x in data]
            target_labels = torch.concat(target_labels)
            #seq
            seqs = [x[3] for x in data]
            seqs = tuple(chain.from_iterable(seqs))

            seg_expr = [x[4] for x in data]
            # repeat each element twice, once for each haplotype
            seg_expr = torch.Tensor(seg_expr).repeat_interleave(2)

            return (masked_sequence, seg_labels, seg_expr),target_labels_masked, target_labels, seqs
        else: 
            masked_sequence = [x[0][0] for x in data]
            masked_sequence = [torch.stack(torch.split(d, 3)) for d in masked_sequence] 
            masked_sequence = torch.concat(masked_sequence)

            seg_labels = torch.Tensor([x[0][1] for x in data]).type(torch.LongTensor)

            target_labels_masked = [x[1] for x in data]
            target_labels_masked = torch.vstack(target_labels_masked)

            target_labels = [x[2] for x in data]
            target_labels = torch.vstack(target_labels)

            seqs = [x[3] for x in data]
            
            seg_expr = torch.Tensor([x[4] for x in data])

            return (masked_sequence, seg_labels, seg_expr), target_labels_masked, target_labels, seqs
    




class ExpressionDataset(Dataset):
    def __init__(self, seq_df, use_haplotypes, transform, dataset_path, seq_len, max_augm_shift=0):

        if dataset_path.endswith('.fa'):
            self.fasta = pysam.FastaFile(dataset_path)
        else:
            self.fasta = None

        self.seq_df = seq_df
        self.transform = transform
        self.max_augm_shift = max_augm_shift
        self.seq_len = seq_len
        self.use_haplotypes = use_haplotypes


    def __len__(self):
        return len(self.seq_df)
    
    def __getitem__(self, idx):

        if self.fasta:
                seq = self.fasta.fetch(self.seq_df.iloc[idx].seq_name).upper()
        else:
            seq = self.seq_df.iloc[idx].seq.upper()
        shift = np.random.randint(self.max_augm_shift+1) #random shift at training, must be chunk_size-input_params.seq_len
        seq = seq[shift:shift+self.seq_len] #shift the sequence and limit its size
        seg_label = self.seq_df.iloc[idx].seg_label #label for segment-aware training

        if self.use_haplotypes:
            
            seq1 = seq.replace('-','').replace('B','A').replace('F','A').replace('M','R') # father
            seq2 = seq.replace('-','').replace('B','A').replace('M','A').replace('F','R') # mother 

            masked_sequence1, target_labels_masked1, target_labels1, _, _ = self.transform(seq1)
            masked_sequence2, target_labels_masked2, target_labels2, _, _ = self.transform(seq2)

            masked_sequence = torch.vstack((masked_sequence1, masked_sequence2))
            seg_label = torch.vstack((torch.tensor(seg_label), torch.tensor(seg_label)))
            masked_sequence = (masked_sequence, seg_label)

            target_labels_masked = torch.vstack((target_labels_masked1, target_labels_masked2))
            target_labels = torch.vstack((target_labels1, target_labels2))
            seq = (seq1, seq2)

            if self.regression: 
                seq_expr = self.seq_df.iloc[idx].expr.astype(np.float32)
                return masked_sequence, target_labels_masked, target_labels, seq, seq_expr

            return masked_sequence, target_labels_masked, target_labels, seq
        
        else: 
        
            #for given genotype, randomly choose a haplotype for training/testing
            if np.random.rand()>0.5:
                seq = seq.replace('-','').replace('B','A').replace('F','A').replace('M','R')
            else:
                seq = seq.replace('-','').replace('B','A').replace('M','A').replace('F','R')

            masked_sequence, target_labels_masked, target_labels, _, _ = self.transform(seq)
            masked_sequence = (masked_sequence, seg_label)

            seq_expr = self.seq_df.iloc[idx].expr.astype(np.float32)
            return masked_sequence, target_labels_masked, target_labels, seq, seq_expr

            
        