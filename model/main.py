#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle

import re

import os
import gc

import pysam

import torch
from torch.utils.data import DataLoader, Dataset

#from tqdm.notebook import tqdm
from tqdm import tqdm


# In[2]:


from encoding_utils import sequence_encoders, sequence_utils

import helpers.train_eval as train_eval    #train and evaluation
import helpers.misc as misc                #miscellaneous functions
from helpers.metrics import MaskedAccuracy
from helpers.temperature_scaling import ModelWithTemperature


from models.spec_dss import DSSResNet, DSSResNetEmb, SpecAdd


import argparse

parser = argparse.ArgumentParser("main.py")

parser.add_argument("--dataset", help = "FASTA or parquet file", type = str, required = True)
parser.add_argument("--output_dir", help = "dir to save predictions and model/optimizer weights", type = str, required = True)
parser.add_argument("--model_weight", help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--optimizer_weight", help = "initialization weight of the optimizer, use only to resume training", type = str, default = None, required = False)
parser.add_argument("--split_mask", help = "split mask into 80% zeros 10% random 10% same", type=misc.str2bool, default = True, required = False)
parser.add_argument("--mask_rate", help = "single float or 2 floats for reference and alternative", type=float, nargs='+', default = [0.2], required = False)
parser.add_argument("--masking", help = "stratified_maf or none", type = str, default = 'none', required = False)
parser.add_argument("--validate_every", help = "validate every N epochs", type = int,  default = 1, required = False)
parser.add_argument("--test", help = "model to inference mode", action='store_true', default = False, required = False)
parser.add_argument("--get_embeddings", help = "save embeddings at test", action='store_true', default = False, required = False)
parser.add_argument("--agnostic", help = "use a pecies agnostic version", default = False, type=misc.str2bool, required = False)
parser.add_argument("--mask_at_test", help = "test on masked sequences", action='store_true', default = True, required = False)
parser.add_argument("--seq_len", help = "max sequence length", type = int, default = 5000, required = False)
parser.add_argument("--train_splits", help = "split each epoch into N epochs", type = int, default = 1, required = False)
parser.add_argument("--fold", help = "current fold", type = int, default = None, required = False)
parser.add_argument("--Nfolds", help = "total number of folds", type = int, default = None, required = False)
parser.add_argument("--tot_epochs", help = "total number of training epochs, (after splitting)", type = int, default = 100, required = False)
parser.add_argument("--d_model", help = "model dimensions", type = int, default = 256, required = False)
parser.add_argument("--n_layers", help = "number of layers", type = int, default = 16, required = False)
parser.add_argument("--batch_size", help = "batch size", type = int, default = 16, required = False)
parser.add_argument("--learning_rate", help = "learning rate", type = float, default = 1e-3, required = False)
parser.add_argument("--dropout", help = "model dropout", type = float, default = 0., required = False)
parser.add_argument("--weight_decay", help = "Adam weight decay", type = float, default = 0., required = False)
parser.add_argument("--save_at", help = "epochs to save model/optimizer weights, 1-based", nargs='+', type = str, default = [], required = False)

input_params = vars(parser.parse_args())

input_params = misc.dotdict(input_params)

input_params.save_at = misc.list2range(input_params.save_at)

if len(input_params.mask_rate)==1:
    input_params.mask_rate = input_params.mask_rate[0]

for param_name in ['output_dir', '\\',
'dataset', 'agnostic', '\\',
'Nfolds', 'fold', '\\',
'test', 'mask_at_test', 'get_embeddings', '\\',
'seq_len', '\\',
'mask_rate', 'split_mask', '\\',
'tot_epochs', 'save_at', 'train_splits', '\\',
'validate_every', '\\',
'd_model', 'n_layers','dropout', '\\',
'model_weight', 'optimizer_weight', '\\',
'batch_size', 'learning_rate', 'weight_decay', '\\',
]:

    if param_name == '\\':
        print()
    else:
        print(f'{param_name.upper()}: {input_params[param_name]}')



class SeqDataset(Dataset):

    def __init__(self, seq_df, transform, max_augm_shift=0, 
                 mode='train'):

        if input_params.dataset.endswith('.fa'):
            self.fasta = pysam.FastaFile(input_params.dataset)
        else:
            self.fasta = None

        self.seq_df = seq_df
        self.transform = transform
        self.max_augm_shift = max_augm_shift
        self.mode = mode

    def __len__(self):

        return len(self.seq_df)

    def __getitem__(self, idx):

        if self.fasta:
            seq = self.fasta.fetch(self.seq_df.iloc[idx].seq_name).upper()
        else:
            seq = self.seq_df.iloc[idx].seq.upper()

        shift = np.random.randint(self.max_augm_shift+1) #random shift at training, must be chunk_size-input_params.seq_len

        seq = seq[shift:shift+input_params.seq_len] #shift the sequence and limit its size

        seg_label = self.seq_df.iloc[idx].seg_label #label for segment-aware training

        #for given genotype, randomly choose a haplotype for training/testing
        if np.random.rand()>0.5:
            seq = seq.replace('-','').replace('B','A').replace('F','A').replace('M','R')
        else:
            seq = seq.replace('-','').replace('B','A').replace('M','A').replace('F','R')

        #if input_params.masking == 'stratified_maf' and not input_params.test:
        #    #select mask for the sequence depending on sequence coordinates w.r.t. contig
        #    seg_name = self.seq_df.iloc[idx].seg_name
        #    seq_mask = meta.loc[seg_name].MASK.values
        #    masked_sequence, target_labels_masked, target_labels, _, _ = self.transform(seq, mask = seq_mask)
        #else:
        #    masked_sequence, target_labels_masked, target_labels, _, _ = self.transform(seq)

        masked_sequence, target_labels_masked, target_labels, _, _ = self.transform(seq)

        masked_sequence = (masked_sequence, seg_label)

        return masked_sequence, target_labels_masked, target_labels, seq

    def close(self):
        self.fasta.close()


# In[5]:


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('\nCUDA device: GPU\n')
else:
    device = torch.device('cpu')
    print('\nCUDA device: CPU\n')
    #raise Exception('CUDA is not found')



gc.collect()
torch.cuda.empty_cache()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

if input_params.dataset.endswith('.fa'):
    seq_df = pd.read_csv(input_params.dataset + '.fai', header=None, sep='\t', usecols=[0], names=['seq_name'])
elif input_params.dataset.endswith('.parquet'):
    seq_df = pd.read_parquet(input_params.dataset).reset_index()

seq_df[['split','sample_id','seg_name']] =  seq_df['seq_name'].str.split(':',expand=True)



#for each contig, mark first and lask chunks
#seq_df[['split','sample_id','chrom','pos']] =  seq_df['seq_name'].str.split(':',expand=True)

#seq_df.pos = seq_df.pos.astype(int)

#df = seq_df.sort_values(by=['sample_id','chrom','pos']).copy()
#df['last_chunk'] = ~df.duplicated(subset=['sample_id','chrom'],keep='last')
#df['first_chunk'] = ~df.duplicated(subset=['sample_id','chrom'],keep='first')

#seq_df = seq_df.merge(df[['seq_name','first_chunk','last_chunk']])


# In[9]:


#from sklearn.model_selection import train_test_split
#
#meta = pd.read_csv(os.path.dirname(input_params.dataset) + '/meta.csv.gz')[['GENE_ID','AF']]
#meta['MAF'] = meta.AF.apply(lambda x: min(x,1-x)) #minor allele frequency
#
#meta['MAF_bin'] = np.digitize(meta.MAF, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5], right=True) #bin indices for MAF
#
#meta.set_index('GENE_ID', inplace=True)
#
#def get_random_mask():
#    '''
#    randomly mask positions at mask_rate
#    stratify by MAF distribution
#    '''
#    mask = np.zeros(len(meta.MAF_bin))
#    masked_idx, _ = train_test_split(np.arange(len(meta.MAF_bin)), train_size=input_params.mask_rate,
#                    shuffle=True, stratify=meta.MAF_bin)
#    mask[masked_idx] = 1
#    meta['MASK'] = mask
#
#    return meta


# In[10]:


if not input_params.agnostic:
    #for segment-aware model, assign a label to each segment
    seg_name = seq_df.seq_name.apply(lambda x:':'.join(x.split(':')[2:]))
    segment_encoding = seg_name.drop_duplicates().reset_index(drop=True)
    segment_encoding = {seg_name:idx for idx,seg_name in segment_encoding.items()}
    seq_df['seg_label'] = seg_name.map(segment_encoding)
else:
    seq_df['seg_label'] = 0


if input_params.test:
    seq_df = seq_df[seq_df.split=='test']
else:
    seq_df = seq_df[seq_df.split!='test']
    
    
test_df = None

if not input_params.test:

    #Train and Validate

    seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = input_params.seq_len, total_len = input_params.seq_len,
                                                      mask_rate = input_params.mask_rate, split_mask = input_params.split_mask)

    #N_train = int(len(seq_df)*(1-input_params.val_fraction))

    if input_params.fold is not None:
        
        samples = seq_df.sample_id.unique()
        val_samples = samples[input_params.fold::input_params.Nfolds] 
        
        train_df = seq_df[~seq_df.sample_id.isin(val_samples)] 
        test_df = seq_df[seq_df.sample_id.isin(val_samples)]
        
        test_dataset = SeqDataset(test_df, transform = seq_transform, mode='eval')
        test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size, num_workers = 0, collate_fn = None, shuffle = False)
    
    else:
        train_df = seq_df
        
        #train_df = seq_df[seq_df.split=='train']
        #test_df = seq_df[seq_df.split=='val']
  
    N_train = len(train_df)

    train_fold = np.repeat(list(range(input_params.train_splits)),repeats = N_train // input_params.train_splits + 1 )
    train_df['train_fold'] = train_fold[:N_train]

    train_dataset = SeqDataset(train_df, transform = seq_transform,  mode='train')
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = input_params.batch_size, num_workers = 2, collate_fn = None, shuffle = False)


elif input_params.get_embeddings:

    if input_params.mask_at_test:
        seq_transform = sequence_encoders.RollingMasker(mask_stride = 50, frame = 0)
    else:
        seq_transform = sequence_encoders.PlainOneHot(frame = 0, padding = 'none')

    test_dataset = SeqDataset(seq_df, transform = seq_transform, mode='eval')

    test_dataloader = DataLoader(dataset = test_dataset, batch_size = 1, num_workers = 1, collate_fn = None, shuffle = False)

else:

    #Test

    seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = input_params.seq_len, total_len = input_params.seq_len,
                                                      mask_rate = input_params.mask_rate, split_mask = input_params.split_mask)

    test_dataset = SeqDataset(seq_df, transform = seq_transform, mode='eval')
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = input_params.batch_size, num_workers = 2, collate_fn = None, shuffle = False)


# In[13]:


seg_encoder = SpecAdd(embed = True, encoder = 'label', Nsegments=seq_df.seg_label.nunique(), d_model = input_params.d_model)

model = DSSResNetEmb(d_input = 3, d_output = 3, d_model = input_params.d_model, n_layers = input_params.n_layers,
                     dropout = input_params.dropout, embed_before = True, species_encoder = seg_encoder)

model = model.to(device)

model_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(model_params, lr = input_params.learning_rate, weight_decay = input_params.weight_decay)


# In[14]:


last_epoch = 0

if input_params.model_weight:

    if torch.cuda.is_available():
        #load on gpu
        model.load_state_dict(torch.load(input_params.model_weight))
        if input_params.optimizer_weight:
            optimizer.load_state_dict(torch.load(input_params.optimizer_weight))
    else:
        #load on cpu
        model.load_state_dict(torch.load(input_params.model_weight, map_location=torch.device('cpu')))
        if input_params.optimizer_weight:
            optimizer.load_state_dict(torch.load(input_params.optimizer_weight, map_location=torch.device('cpu')))

    last_epoch = int(input_params.model_weight.split('_')[-3]) #infer previous epoch from input_params.model_weight

weights_dir = os.path.join(input_params.output_dir, 'weights') #dir to save model weights at save_at epochs

if input_params.save_at:
    os.makedirs(weights_dir, exist_ok = True)

#lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#        milestones=input_params.lr_sch_milestones, gamma=input_params.lr_sch_gamma, verbose=False)


# In[15]:


def metrics_to_str(metrics):
    loss, accuracy, masked_acc, masked_recall, masked_IQS = metrics
    return f'loss: {loss:.4}, acc: {accuracy:.4}, masked acc: {masked_acc:.4}, {misc.print_class_recall(masked_recall, "masked recall: ")}, masked IQS: {masked_IQS:.4}'




from helpers.misc import print    #print function that displays time

if not input_params.test:

    for epoch in range(last_epoch+1, input_params.tot_epochs+1):

        print(f'EPOCH {epoch}: Training...')

        if input_params.masking == 'stratified_maf':

            meta = get_random_mask()

        train_dataset.seq_df = train_df[train_df.train_fold == (epoch-1) % input_params.train_splits]
        print(f'using train samples: {list(train_dataset.seq_df.index[[0,-1]])}')

        train_metrics = train_eval.model_train(model, optimizer, train_dataloader, device,
                            silent = 1)

        print(f'epoch {epoch} - train, {metrics_to_str(train_metrics)}')

        if epoch in input_params.save_at: #save model weights

            misc.save_model_weights(model, optimizer, weights_dir, epoch)

        if test_df is not None and ( epoch==input_params.tot_epochs or
                            (input_params.validate_every and epoch%input_params.validate_every==0)):

            print(f'EPOCH {epoch}: Validating...')

            val_metrics, *_ =  train_eval.model_eval(model, optimizer, test_dataloader, device,
                    silent = 1)

            print(f'epoch {epoch} - validation, {metrics_to_str(val_metrics)}')

        #lr_scheduler.step()
else:

    print(f'EPOCH {last_epoch}: Test/Inference...')

    test_metrics, test_embeddings, motif_probas =  train_eval.model_eval(model, optimizer, test_dataloader, device,
                                                          get_embeddings = input_params.get_embeddings,
                                                          silent = 0)



    print(f'epoch {last_epoch} - test, {metrics_to_str(test_metrics)}')

    if input_params.get_embeddings:

        os.makedirs(input_params.output_dir, exist_ok = True)

        with open(input_params.output_dir + '/embeddings.pickle', 'wb') as f:
            #test_embeddings = np.vstack(test_embeddings)
            #np.save(f,test_embeddings)
            pickle.dump(test_embeddings,f)
            #pickle.dump(seq_df.seq_name.tolist(),f)

print()
print(f'peak GPU memory allocation: {round(torch.cuda.max_memory_allocated(device)/1024/1024)} Mb')
print('Done')
