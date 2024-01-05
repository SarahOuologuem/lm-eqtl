from helpers.metrics import MeanRecall, MaskedAccuracy, IQS
from helpers.misc import EMA, print_class_recall
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb



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

    reg_criterion = nn.MSELoss()
    
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

        if log_wandb:
            if itr_idx % 10 == 0:
                wandb.log({"expr_loss": expr_loss, "masked_loss": masked_loss, "total_loss": loss,
                        "masked_acc": masked_accuracy.compute(), "masked_recall": masked_recall.compute(), "masked_IQS": masked_IQS.compute()})

        smoothed_loss = loss_EMA.update(loss.item())

        preds = torch.argmax(logits, dim=1)

        accuracy.update(preds, targets)
        masked_recall.update(preds, targets_masked)
        masked_accuracy.update(preds, targets_masked)
        masked_IQS.update(preds, targets_masked)
        
        if not silent:
            pbar.update(1)
            pbar.set_description(f'Mem: {print_gpu_mb(device)}, reg_loss: {expr_loss:.4f}, masked_loss: {masked_loss}, total_loss: {smoothed_loss:.4f}, acc: {accuracy.compute():.4}, {print_class_recall(masked_recall.compute(), "masked recall: ")}, masked acc: {masked_accuracy.compute():.4}, masked IQS: {masked_IQS.compute():.4}')

    if not silent:
        pbar.reset()
        del pbar

    return expr_loss, loss, accuracy, masked_accuracy, masked_recall, masked_IQS
