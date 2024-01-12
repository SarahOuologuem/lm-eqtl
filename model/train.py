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
import datetime
import re
import rootutils
import wandb
from torch.utils.data import DataLoader, Dataset
import scipy


rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from model.models.regression.datasets import ExpressionCollator
from model.encoding_utils import sequence_encoders
import model.helpers.train_eval_a as train_eval    #train and evaluation
import model.helpers.misc as misc                #miscellaneous functions
from models.spec_dss import DSSResNetEmb, SpecAdd, DSSResNetExpression
from model.helpers import misc
from model.models.regression.train_eval import train_reg_model, eval_reg_model
from model.models.regression.datasets import GenoDataset, HaploDataset, ExpressionCollator



def train(cfg: DictConfig) -> None:
    print("training...")

    print("CURRENT WORKING DIRECTORY: ", os.getcwd())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LOAD DATA
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    expression_data = pd.read_csv(cfg.expression_data, sep="\t")\
        .rename(columns=lambda x: re.sub("\..*",'',x))\
        .melt(id_vars = ["TargetID", "Gene_Symbol", "Chr", "Coord"], var_name = "sample_id", value_name = "expr")[["Gene_Symbol", "sample_id", "expr"]]
    seq_expression_df = pd.merge(seq_df, expression_data, left_on=["sample_id", "seg_name"], right_on=["sample_id", "Gene_Symbol"])

    # define test_set
    test_indices = np.random.randint(0, len(seq_expression_df), size=int(0.1*len(seq_expression_df)))
    save_path = os.path.join(
        cfg.output_dir, 
        cfg.proj_name,
        cfg.run_name, 
        "test_set_indices.pkl"
    )

    if not os.path.exists(os.path.dirname(save_path)): 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # save the test indices
    with open(save_path, "wb") as f:
        pickle.dump(test_indices, f)

    seq_expression_df = seq_expression_df.loc[~seq_expression_df.index.isin(test_indices)]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SELECT DEVICE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("CUDA IS AVAILABLE: ", torch.cuda.is_available())
    print("cfg.DEVICE: ", cfg.device)
    if torch.cuda.is_available() and cfg.device=='cuda':
        device = torch.device('cuda')
        print('\nCUDA device: GPU\n')
    else:
        device = torch.device('cpu')
        print('\nCUDA device: CPU\n')
    gc.collect()
    torch.cuda.empty_cache()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DATASET SETUP
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\n Setting up datasets... \n")
    collator = ExpressionCollator(haplotypes=cfg.dataset.use_haplotypes)
    seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = cfg.dataset.seq_len, total_len = cfg.dataset.seq_len,
                                                      mask_rate = cfg.mask_rate, split_mask = cfg.split_mask)
    # instantiating the dataset
    dataset_factory = hydra.utils.instantiate(cfg.dataset)

    if cfg.fold is not None: 
        print("Using fold: ", cfg.fold)
        samples = seq_expression_df.sample_id.unique()
        val_samples = samples[cfg.fold::cfg.Nfolds] 
        train_df = seq_expression_df[~seq_expression_df.sample_id.isin(val_samples)] 
        test_df = seq_expression_df[seq_expression_df.sample_id.isin(val_samples)]
        test_dataset = dataset_factory(
            seq_df=test_df, 
            transform = seq_transform
        )
        test_dataloader = DataLoader(dataset = test_dataset, batch_size = cfg.batch_size, num_workers = 0, collate_fn = collator, shuffle = False)

        # save the dataloader
        with open(
            os.path.join(
                cfg.output_dir, 
                cfg.proj_name,
                cfg.run_name, 
                f"test_dataloader_fold_{cfg.fold}.pkl"
            ), "wb"
        ) as pkl: 
            pickle.dump(test_dataloader, pkl)

    else: 
        print("No fold specified, using all data for training")
        train_df = seq_expression_df


    N_train = len(train_df)
    train_fold = np.repeat(list(range(cfg.train_splits)),repeats = N_train // cfg.train_splits + 1 )
    train_df['train_fold'] = train_fold[:N_train]

    # create training dataset & dataloader 
    train_dataset = dataset_factory(
        train_df, 
        transform = seq_transform
    )
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = cfg.batch_size, num_workers = 2, collate_fn = collator, shuffle = False)

    check_batch = next(iter(train_dataloader))
    seqs = check_batch[-1]
    print("LENGTH OF SEQ", len(seqs[0]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MODEL SETUP
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\n Setting up model... \n")
    seg_encoder = SpecAdd(embed = True, encoder = 'label', Nsegments=seq_df.seg_label.nunique(), d_model = cfg.model.d_model)

    # instantiating the model
    model_factory = hydra.utils.instantiate(cfg.model)
    model = model_factory(
        species_encoder = seg_encoder
    )

    with open(os.path.join(cfg.output_dir, cfg.proj_name, cfg.run_name, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    model = model.to(device)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Freeze layers
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\n Freezing layers... \n")
    for param_name, param in model.named_parameters(): 
        if not param_name.startswith("regression"):
            param.requires_grad = False

    model_params = [p for p in model.parameters() if p.requires_grad]

    print(f"TOTAL PARAMS: {len(list(model.parameters()))}")
    print(f"TRAINING {len(model_params)} PARAMS")

    optimizer = torch.optim.Adam(model_params, lr = cfg.learning_rate, weight_decay = cfg.weight_decay)

    weights_dir = os.path.join(cfg.output_dir, cfg.proj_name, 'weights') #dir to save model weights at save_at epochs


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LOAD MODEL WEIGHTS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    print("Sample Model weights before loading: ")
    print(list(model.parameters())[0][0][0][:10])

    print("\n Loading model weights... \n")
    if cfg.model_weights:
        if torch.cuda.is_available():
            #load on gpu
            model.load_state_dict(torch.load(cfg.model_weights), strict=False)
            if cfg.optimizer_weight:
                optimizer.load_state_dict(torch.load(cfg.optimizer_weight))
        else:
            #load on cpu
            model.load_state_dict(torch.load(cfg.model_weights, map_location=torch.device('cpu')), strict=False)
            if cfg.optimizer_weight:
                optimizer.load_state_dict(torch.load(cfg.optimizer_weight, map_location=torch.device('cpu')))

    print("Sample Model weights after loading: ")
    print(list(model.parameters())[0][0][0][:10])


    # setup wandb
    if cfg.log_wandb:
        wandb.init(project=cfg.proj_name, name=cfg.run_name) # seq_len, batch_size, geno or haplo, loss (full = reg + masked)
        wandb.watch(model, log_freq=1)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TRAINING LOOP
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("Total samples in train set: ", len(train_df))
    last_epoch = 0
    
    print("\n Training model... \n")
    for epoch in range(last_epoch+1, cfg.tot_epochs+1):

        print(f'EPOCH {epoch}: Training...')

        #if cfg.masking == 'stratified_maf':

        #    meta = get_random_mask()

        train_dataset.seq_df = train_df[train_df.train_fold == (epoch-1) % cfg.train_splits]

        train_metrics = train_reg_model(model, optimizer, train_dataloader, device,
                            silent = False, log_wandb=cfg.log_wandb)
            
        # print(f'epoch {epoch} - train, {metrics_to_str(train_metrics)}')

        if epoch in cfg.save_at: #save model weights
            misc.save_model_weights(model, optimizer, weights_dir, epoch)
    print("Training complete.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # EVALUATION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if cfg.fold is not None: 
        print("\n Testing model... \n")
        test_metrics = eval_reg_model(model, test_dataloader, device, silent=False)
        print("\nTest metrics: \n")
        print("avg_loss:", test_metrics["avg_loss"])
        print("avg_reg_loss:", test_metrics["avg_reg_loss"])
        print("avg_masked_loss:", test_metrics["avg_masked_loss"])
        print("accuracy:", test_metrics["accuracy"])
        print("masked_accuracy:", test_metrics["masked_accuracy"])
        print("masked_recall:", test_metrics["masked_recall"])
        print("masked_IQS:", test_metrics["masked_IQS"])
        print()

        expr_pred = test_metrics["expr_pred_dict"]["expr_pred"]
        expr_truth = test_metrics["expr_pred_dict"]["expr"]
        slope, _, r_value, p_value, _ = scipy.stats.linregress(expr_truth, expr_pred)

        print("R^2: ", r_value**2)
        print("Slope: ", slope)
        print("P-value: ", p_value)



@hydra.main(version_base=None, config_path="configs/", config_name="train")
def main(cfg: DictConfig) -> None:
    print("Main called")
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
