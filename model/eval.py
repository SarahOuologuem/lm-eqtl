import torch
import gc
import hydra
import os
import numpy as np
import pickle
import omegaconf
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import datetime
import re
import rootutils
import wandb
from torch.utils.data import DataLoader, Dataset
import scipy
import polars as pl


rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from model.models.regression.datasets import ExpressionCollator
from model.encoding_utils import sequence_encoders
import model.helpers.train_eval_a as train_eval    #train and evaluation
import model.helpers.misc as misc                #miscellaneous functions
from models.spec_dss import DSSResNetEmb, SpecAdd, DSSResNetExpression
from model.helpers import misc
from model.models.regression.train_eval import train_reg_model, eval_reg_model
from model.models.regression.datasets import GenoDataset, HaploDataset, ExpressionCollator



def eval(cfg: DictConfig): 
    print("Evaluation...")

    if cfg.dataset_path.endswith('.fa'):
        seq_df = pd.read_csv(cfg.dataset + '.fai', header=None, sep='\t', usecols=[0], names=['seq_name'])
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
    seq_df = pl.from_pandas(seq_df)
    genes_in_seq_df = seq_df["seg_name"].unique()
    seq_expression_df = pl.read_csv(os.path.join(cfg.expression_data), separator="\t")\
        .filter(pl.col("Gene_Symbol").is_in(list(genes_in_seq_df)))\
        .melt(id_vars = ["TargetID", "Gene_Symbol", "Chr", "Coord"], variable_name = "sample_id", value_name = "expr")[["Gene_Symbol", "sample_id", "expr"]]\
        .with_columns(patient = pl.col("sample_id").str.replace("\..*", ""))\
        .with_columns(sample_id = pl.col("sample_id").str.replace("\.M.*", ""))\
        .join(seq_df, left_on=["patient", "Gene_Symbol"], right_on=["sample_id", "seg_name"]).to_pandas()
    seq_df = seq_df.to_pandas()

    collator = ExpressionCollator(haplotypes=cfg.dataset.use_haplotypes)
    seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = cfg.dataset.seq_len, total_len = cfg.dataset.seq_len,
                                                      mask_rate = cfg.mask_rate, split_mask = cfg.split_mask)
    
    dataset_factory = hydra.utils.instantiate(cfg.dataset)

    run_path = os.path.join(cfg.output_dir, cfg.proj_name, cfg.run_name)
    if not os.path.exists(run_path):
        os.makedirs(run_path, exist_ok=True)

    if cfg.split_df:
        print("Using fold dataframe for fold: ", cfg.eval_fold + 1)
        split_df = pl.read_csv(cfg.split_df, separator="\t")

        seq_expression_df = pl.from_pandas(seq_expression_df).join(split_df, left_on=["patient", "sample_id", "Gene_Symbol"], right_on=["patient", "sample_id", "gene_id"])
        print("seq_expression_df: ", seq_expression_df)
        val_df = seq_expression_df.filter(pl.col("test_split") == cfg.eval_fold + 1).to_pandas()
        train_df = seq_expression_df.filter(~pl.col("test_split").eq(cfg.eval_fold + 1)).to_pandas()
        print("Val df shape: ", val_df.shape)
    else:
        print("Val df shape: ", val_df.shape)
        inds_path = os.path.join(run_path, "val_indices.npy")
        val_indices = np.load(inds_path)
        val_df = seq_expression_df.iloc[val_indices]

    val_dataset = dataset_factory(seq_df=val_df, transform=seq_transform)
    val_dataloader = DataLoader(val_dataset, cfg.batch_size, shuffle=False, collate_fn=collator)
    # train_df = seq_expression_df.iloc[~seq_expression_df.index.isin(val_indices)]
    train_dataset = dataset_factory(seq_df=train_df, transform=seq_transform)
    train_dataloader = DataLoader(train_dataset, cfg.batch_size, shuffle=False, collate_fn=collator)

    print("CUDA IS AVAILABLE: ", torch.cuda.is_available())
    print("cfg.DEVICE: ", cfg.device)
    if torch.cuda.is_available() and cfg.device=='cuda':
        print("GPU MODEL: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        device = torch.device('cuda')
        print('\nDevice: GPU\n')
    else:
        device = torch.device('cpu')
        print('\nDevice: CPU\n')
    gc.collect()
    torch.cuda.empty_cache()

    # load model parameters
    if cfg.weights_path is None: 
        with open(os.path.join(run_path, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        weights_path = os.path.join(run_path, "weights", f"fold_{cfg.eval_fold}", f"epoch_{cfg.weights_epoch}_weights_model.pt")
    else: 
        with open(os.path.join("results", "example_model.pkl"), "rb") as f: 
            model = pickle.load(f)
        print("Loading weights from specific path...")
        weights_path = cfg.weights_path

    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.to(device)

    if cfg.eval_train_set: 
        print("\nEvaluating on train set...\n")
        train_metrics, train_embeddings = eval_reg_model(
            model, 
            train_dataloader, 
            cfg.dataset.use_haplotypes, 
            device, 
            silent=False, 
            eval_until=cfg.eval_until_batch
        )
    else: 
        train_metrics = None
        train_embeddings = None
    print("\nEvaluating on validation set...\n")
    val_metrics, val_embeddings = eval_reg_model(
        model, 
        val_dataloader, 
        cfg.dataset.use_haplotypes, 
        device, 
        silent=False, 
        eval_until=cfg.eval_until_batch
    )

    print("Eval done.")

    print("Saving metrics...")
    metrics_path = os.path.join(run_path, f"fold_{cfg.eval_fold}_epoch_{cfg.weights_epoch}_metrics.pkl")
    metrics = {
        "val": val_metrics,
        "train": train_metrics, 
        "train_embeddings": train_embeddings, 
        "val_embeddings": val_embeddings
    }

    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

@hydra.main(version_base=None, config_path="configs/", config_name="eval")
def main(cfg: DictConfig) -> None:
    print("Main called")
    print(OmegaConf.to_yaml(cfg))
    eval(cfg)



if __name__ == "__main__":
    main()