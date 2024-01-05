# hydra-style train

import torch
import gc
import hydra
import os
import numpy as np
import pickle
import omegaconf
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from models.model.expression.datasets import ExpressionCollator



def train(cfg: DictConfig) -> None:
    print("training...")

    # load the data

    if cfg.dataset_path.endswith('.fa'):
        seq_df = pd.read_csv(input_params.dataset + '.fai', header=None, sep='\t', usecols=[0], names=['seq_name'])
    elif cfg.dataset_path.endswith('.parquet'):
        seq_df = pd.read_parquet(cfg.dataset_path).reset_index()
        
    seq_df[['split','sample_id','seg_name']] =  seq_df['seq_name'].str.split(':',expand=True)

    if not cfg.agnostic:
        #for segment-aware model, assign a label to each segment
        seg_name = seq_df.seq_name.apply(lambda x:':'.join(x.split(':')[2:]))
        segment_encoding = seg_name.drop_duplicates().reset_index(drop=True)
        segment_encoding = {seg_name:idx for idx,seg_name in segment_encoding.items()}
        seq_df['seg_label'] = seg_name.map(segment_encoding)
    else:
        seq_df['seg_label'] = 0

    # load the expression data and take samples that have expression values
    print("Loading expression data...")
    expression_data = pd.read_csv(cfg.expression_data, sep="\t")\
        .rename(columns=lambda x: re.sub("\..*",'',x))\
        .melt(id_vars = ["TargetID", "Gene_Symbol", "Chr", "Coord"], var_name = "sample_id", value_name = "expr")[["Gene_Symbol", "sample_id", "expr"]]
    seq_expression_df = pd.merge(seq_df, expression_data, left_on=["sample_id", "seg_name"], right_on=["sample_id", "Gene_Symbol"])

    # define test_set
    test_indices = np.random.randint(0, len(seq_expression_df), size=int(0.1*len(seq_expression_df)))
    save_path = os.path.join(
        cfg.output_dir, 
        cfg.run_name, 
        "test_set_indices.pkl"
    )

    if not os.path.exists(os.path.dirname(save_path)): 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # save the test indices
    with open(save_path, "wb") as f:
        pickle.dump(test_indices, f)

    seq_expression_df = seq_expression_df.loc[~seq_expression_df.index.isin(test_indices)]


    # select device
    print("CUDA IS AVAILABLE: ", torch.cuda.is_available())
    print("INPUT_PARAMS.DEVICE: ", input_params.device)
    if torch.cuda.is_available() and input_params.device=='cuda':
        device = torch.device('cuda')
        print('\nCUDA device: GPU\n')
    else:
        device = torch.device('cpu')
        print('\nCUDA device: CPU\n')
    gc.collect()
    torch.cuda.empty_cache()

    collator = ExpressionCollator(haplotypes=cfg.dataset.use_haplotypes)

    # instantiating the dataset
    dataset_factory = hydra.utils.instantiate(cfg.dataset)
    dataset = dataset_factory(
        seq_df=, 
        transform=, 
        dataset_path=, 
    )

    # instantiating the model
    model = hydra.utils.instantiate(cfg.model)

    # freeze certain layers: 


    # CONTINUE HERE, FINISH main function


    # instantiating the optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())





@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.pretty())
    train(cfg)
