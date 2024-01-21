
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
import numpy as np
import sys

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


def train_reg_model(model, optimizer, dataloader, device, silent=False, log_wandb=True):
    """train function with added regression loss"""

    criterion = torch.nn.CrossEntropyLoss(reduction = "mean")

    accuracy = MaskedAccuracy(smooth=True).to(device)
    masked_recall = MeanRecall().to(device)
    masked_accuracy = MaskedAccuracy(smooth=True).to(device)
    masked_IQS = IQS().to(device)

    reg_criterion = nn.MSELoss(reduction="mean")
    
    model.train() #model to train mode

    if not silent:
        tot_itr = len(dataloader.dataset)//dataloader.batch_size #total train iterations
        pbar = tqdm(total = tot_itr, ncols=750) #progress bar

    loss_EMA = EMA()

    for itr_idx, ((masked_sequence, species_label, expr), targets_masked, targets, _) in enumerate(dataloader):

        masked_sequence = masked_sequence.to(device)
        species_label = species_label.to(device)
        targets_masked = targets_masked.to(device)
        targets = targets.to(device)
        expr = expr.reshape(-1, 1).to(device)

        logits, _, expr_pred = model(masked_sequence, species_label)




        masked_loss = criterion(logits, targets_masked)
        expr_loss = reg_criterion(expr_pred, expr)

        optimizer.zero_grad()

        loss = expr_loss + masked_loss

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

        if log_wandb:

            expr_pred = expr_pred.detach().cpu().numpy().flatten()
            expr_truth = expr.detach().cpu().numpy().flatten()
            table = wandb.Table(data=np.stack((expr_truth, expr_pred), axis=1), columns=["expr_truth", "expr_pred"])
            wandb.log({'my_histogram': wandb.plot.histogram(table, "scores",
                title="Prediction Score Distribution")})


            if itr_idx % 10 == 0:
                wandb.log({"expr_loss": expr_loss, "masked_loss": masked_loss, "total_loss": loss,
                        "masked_acc": masked_accuracy.compute(), "masked_recall": masked_recall.compute(), "masked_IQS": masked_IQS.compute(), 
                        "batch_expr_pred": wandb.plot.histogram(table, "expr_pred", title="Expr Prediction Distribution"), 
                        "batch_expr_truth": wandb.plot.histogram(table, "expr_truth", title="Expr Truth Distribution")})
        
        if not silent:
            pbar.update(1)
            pbar.set_description(f'Mem: {print_gpu_mb(device)}, reg_loss: {expr_loss:.4f}, masked_loss: {masked_loss}, total_loss: {smoothed_loss:.4f}, acc: {accuracy.compute():.4}, {print_class_recall(masked_recall.compute(), "masked recall: ")}, masked acc: {masked_accuracy.compute():.4}, masked IQS: {masked_IQS.compute():.4}')

    if not silent:
        pbar.reset()
        del pbar

    return expr_loss, loss, accuracy, masked_accuracy, masked_recall, masked_IQS



def eval_reg_model(model, dataloader, device, silent=False):

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
        "expr": np.array([]),
        "expr_pred": np.array([])
    }

    with torch.no_grad():

        for itr_idx, ((masked_sequence, species_label, expr), targets_masked, targets, _) in enumerate(dataloader):

            masked_sequence = masked_sequence.to(device)
            species_label = species_label.long().to(device)
            targets_masked = targets_masked.to(device)
            targets = targets.to(device)
            expr = expr.reshape(-1, 1).to(device)

            logits, _, expr_pred = model(masked_sequence, species_label)
            masked_loss = criterion(logits, targets_masked)
            expr_loss = reg_criterion(expr_pred, expr)
            loss = expr_loss + masked_loss

            avg_loss += loss.item()
            avg_reg_loss += expr_loss.item()
            avg_masked_loss += masked_loss.item()

            logits, _, expr_pred = model(masked_sequence, species_label)
            loss = criterion(logits, targets_masked)

            avg_loss += loss.item()

            masked_preds = torch.argmax(logits, dim=1)
            accuracy.update(masked_preds, targets)
            masked_recall.update(masked_preds, targets_masked)
            masked_accuracy.update(masked_preds, targets_masked)
            masked_IQS.update(masked_preds, targets_masked)

            expr_pred_dict["expr"] = np.concatenate((expr_pred_dict["expr"], expr.cpu().numpy().flatten()))
            expr_pred_dict["expr_pred"] = np.concatenate((expr_pred_dict["expr_pred"], expr_pred.cpu().numpy().flatten()))

            if not silent:
                pbar.update(1)
                pbar.set_description(f'acc: {accuracy.compute():.4}, {print_class_recall(masked_recall.compute(), "masked recall: ")}, masked acc: {masked_accuracy.compute():.4}, masked IQS: {masked_IQS.compute():.4}, loss: {avg_loss/(itr_idx+1):.4}')

    if not silent:
        del pbar

    print(expr_pred_dict)

    return {
        "avg_loss": avg_loss/(itr_idx+1),
        "avg_reg_loss": avg_reg_loss/(itr_idx+1),
        "avg_masked_loss": avg_masked_loss/(itr_idx+1),
        "accuracy": accuracy.compute(),
        "masked_accuracy": masked_accuracy.compute(),
        "masked_recall": masked_recall.compute(),
        "masked_IQS": masked_IQS.compute(),
        "motif_probas": motif_probas,
        "expr_pred_dict": expr_pred_dict
    }
