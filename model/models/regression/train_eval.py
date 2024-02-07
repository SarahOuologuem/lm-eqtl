
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
import numpy as np
import sys
import scipy
import torchmetrics

if ".." not in sys.path: 
    sys.path.append("..")

from helpers.metrics import MeanRecall, MaskedAccuracy, IQS
from helpers.misc import EMA, print_class_recall



def print_gpu_mb(device="cuda"): 
    """print gpu memory usage in Mb"""
    if (device == "cuda" or device == torch.device(type="cuda")) and torch.cuda.is_available(): 
        used = torch.cuda.memory_allocated(device)/1024/1024
        return f"{used:.2f} Mb"
    else: 
        return "N/A"


def train_reg_model(model, optimizer, dataloader, device, silent=False, log_wandb=True, haplotypes=None):
    """train function with added regression loss"""

    criterion = torch.nn.CrossEntropyLoss(reduction = "mean")

    accuracy = MaskedAccuracy(smooth=True).to(device)
    masked_recall = MeanRecall().to(device)
    masked_accuracy = MaskedAccuracy(smooth=True).to(device)
    masked_IQS = IQS().to(device)

    reg_criterion = nn.MSELoss().to(device)
    # reg_criterion = nn.L1Loss(reduction="mean").to(device)
    # reg_criterion = torchmetrics.regression.MeanAbsolutePercentageError().to(device)
    # reg_criterion = torchmetrics.regression.WeightedMeanAbsolutePercentageError().to(device)
    
    model.train() #model to train mode

    if not silent:
        tot_itr = len(dataloader.dataset)//dataloader.batch_size #total train iterations
        pbar = tqdm(total = tot_itr, ncols=750) #progress bar

    loss_EMA = EMA()

    expr_pred_dict = {
        "expr": [],
        "expr_pred": []
    }


    for itr_idx, ((masked_sequence, species_label, expr), targets_masked, targets, _) in enumerate(dataloader):

        masked_sequence = masked_sequence.to(device)
        species_label = species_label.to(device)
        targets_masked = targets_masked.to(device)
        targets = targets.to(device)

        # training on genotypes, regression output will be batch_size / 2
        if not haplotypes: 
            print("Training on genotypes, regression output will be batch_size / 2")
            expr = expr[::2]
        
        # training on haplotypes regression output will match batch_size
        # running_epoch_expr.append(expr.detach().clone().cpu().numpy().flatten())
        expr = expr.to(device).flatten()

        logits, embs, expr_pred = model(masked_sequence, species_label)
        # running_epoch_preds.append(expr_pred.detach().clone().cpu().numpy().flatten())

        print("expr_pred: ", expr_pred[:5])
        print("expr: ", expr[:5])

        masked_loss = criterion(logits, targets_masked)


        expr_loss = reg_criterion(expr_pred.flatten(), expr)


        optimizer.zero_grad()

        loss = expr_loss #+ masked_loss

        loss.backward()

        #if max_abs_grad:
        #    torch.nn.utils.clip_grad_value_(model.parameters(), max_abs_grad)

        optimizer.step()


        smoothed_loss = loss_EMA.update(loss.item())

        preds = torch.argmax(logits, dim=1)

        accuracy.update(preds, targets)
        masked_recall.update(preds, targets_masked)
        masked_accuracy.update(preds, targets_masked)
        masked_IQS.update(preds, targets_masked)

        expr_pred_dict["expr"].append(expr.detach().cpu().numpy().flatten())
        expr_pred_dict["expr_pred"].append(expr_pred.detach().cpu().numpy().flatten())

        running_expr_truth = np.concatenate(expr_pred_dict["expr"])
        running_expr_pred = np.concatenate(expr_pred_dict["expr_pred"])
        running_r_value = scipy.stats.linregress(running_expr_truth, running_expr_pred)[2]

        expr_pred = expr_pred.detach().cpu().numpy().flatten()
        expr_truth = expr.detach().cpu().numpy().flatten()

        # epoch_preds = np.concatenate(running_epoch_preds)
        # epoch_expr = np.concatenate(running_epoch_expr)
        _, _, r_value, _, _ = scipy.stats.linregress(expr_pred, expr_truth)

        if log_wandb:
            wandb.log({"expr_loss": expr_loss, "masked_loss": masked_loss, "total_loss": loss,
                    "masked_acc": masked_accuracy.compute(), "masked_recall": masked_recall.compute(), "masked_IQS": masked_IQS.compute(), 
                    "expr_pred": wandb.Histogram(expr_pred), 
                    "expr_truth": wandb.Histogram(expr_truth), 
                    "expr_pearsonr": running_r_value, 
                    "batch_mean_expr": np.mean(expr_truth)}) 
        
        if not silent:
            pbar.update(1)
            pbar.set_description(f'reg_loss: {expr_loss:.4f}, masked_loss: {masked_loss}, total_loss: {smoothed_loss:.4f}, reg_r_value: {r_value:.4f}, acc: {accuracy.compute():.4}, {print_class_recall(masked_recall.compute(), "masked recall: ")}, masked acc: {masked_accuracy.compute():.4}, masked IQS: {masked_IQS.compute():.4}')
    
    if not silent:
        pbar.reset()
        del pbar


    return expr_loss, loss, accuracy, masked_accuracy, masked_recall, masked_IQS



def eval_reg_model(model, dataloader, haplotypes, device, silent=False, eval_until=None):

    criterion = torch.nn.CrossEntropyLoss(reduction = "mean")

    accuracy = MaskedAccuracy(smooth=True).to(device)
    masked_recall = MeanRecall().to(device)
    masked_accuracy = MaskedAccuracy(smooth=True).to(device)
    masked_IQS = IQS().to(device)

    reg_criterion = nn.MSELoss()

    model.to(device)
    model.eval() #model to train mode

    if not silent:
        tot_itr = len(dataloader.dataset)//dataloader.batch_size #total train iterations
        pbar = tqdm(total = tot_itr, ncols=750) #progress bar

    avg_loss = 0.
    avg_reg_loss = 0.
    avg_masked_loss = 0.

    motif_probas = []

    expr_pred_dict = {
        "expr": [],
        "expr_pred": []
    }

    with torch.no_grad():
        
        if eval_until is None: 
            print("Taking all batches for evaluation...")

        embeddings = []

        for itr_idx, ((masked_sequence, species_label, expr), targets_masked, targets, _) in enumerate(dataloader):

            if eval_until is not None and itr_idx == eval_until:
                break

            masked_sequence = masked_sequence.to(device)
            species_label = species_label.long().to(device)
            targets_masked = targets_masked.to(device)
            targets = targets.to(device)
            expr = expr.to(device).flatten()

            if not haplotypes: 
                expr = expr[::2]

            logits, _, expr_pred = model(masked_sequence, species_label)
            masked_loss = criterion(logits, targets_masked)
            expr_loss = reg_criterion(expr_pred, expr)
            loss = expr_loss + masked_loss

            avg_loss += loss.item()
            avg_reg_loss += expr_loss.item()
            avg_masked_loss += masked_loss.item()

            logits, embs, expr_pred = model(masked_sequence, species_label)
            embs = embs.detach().cpu().numpy()
            embeddings.append(embs)

            loss = criterion(logits, targets_masked)

            avg_loss += loss.item()

            masked_preds = torch.argmax(logits, dim=1)
            accuracy.update(masked_preds, targets)
            masked_recall.update(masked_preds, targets_masked)
            masked_accuracy.update(masked_preds, targets_masked)
            masked_IQS.update(masked_preds, targets_masked)

            expr_pred_dict["expr"].append(expr.detach().cpu().numpy().flatten())
            expr_pred_dict["expr_pred"].append(expr_pred.detach().cpu().numpy().flatten())

            running_expr_truth = np.concatenate(expr_pred_dict["expr"])
            running_expr_pred = np.concatenate(expr_pred_dict["expr_pred"])
            print("Running expr truth: ", running_expr_truth.shape)
            print("Running expr pred: ", running_expr_pred.shape)
            running_r_value = scipy.stats.linregress(running_expr_truth, running_expr_pred)[2]

            if not silent:
                pbar.update(1)
                pbar.set_description(f'expr_loss: {expr_loss:.4f}, running_R: {running_r_value:.4f}, acc: {accuracy.compute():.4}, {print_class_recall(masked_recall.compute(), "masked recall: ")}, masked acc: {masked_accuracy.compute():.4}, masked IQS: {masked_IQS.compute():.4}, loss: {avg_loss/(itr_idx+1):.4}')

    if not silent:
        del pbar

    expr_pred_dict["expr"] = np.concatenate(expr_pred_dict["expr"])
    expr_pred_dict["expr_pred"] = np.concatenate(expr_pred_dict["expr_pred"])

    embeddings = np.concatenate(embeddings)

    return {
        "avg_loss": avg_loss/(itr_idx+1),
        "avg_reg_loss": avg_reg_loss/(itr_idx+1),
        "avg_masked_loss": avg_masked_loss/(itr_idx+1),
        "accuracy": accuracy.cpu().compute(),
        "masked_accuracy": masked_accuracy.cpu().compute(),
        "masked_recall": masked_recall.cpu().compute(),
        "masked_IQS": masked_IQS.cpu().compute(),
        "expr_pred_dict": expr_pred_dict
    }, embeddings

