import torch

import numpy as np

from torch import nn

# from tqdm.notebook import tqdm
from tqdm import tqdm

from helpers.metrics import MeanRecall, MaskedAccuracy, IQS

from helpers.misc import EMA, print_class_recall

from torch.nn.functional import log_softmax


def model_train(model, optimizer, dataloader, device, silent=False):
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    accuracy = MaskedAccuracy(smooth=True).to(device)
    masked_recall = MeanRecall().to(device)
    masked_accuracy = MaskedAccuracy(smooth=True).to(device)
    masked_IQS = IQS().to(device)

    model.train()  # model to train mode

    if not silent:
        tot_itr = len(dataloader.dataset) // dataloader.batch_size  # total train iterations
        pbar = tqdm(total=tot_itr, ncols=750)  # progress bar

    loss_EMA = EMA()

    for itr_idx, ((masked_sequence, species_label), targets_masked, targets, _) in enumerate(dataloader):
        masked_sequence = masked_sequence.to(device)
        species_label = species_label.to(device)
        targets_masked = targets_masked.to(device)
        targets = targets.to(device)

        logits, _ = model(masked_sequence, species_label)

        loss = criterion(logits, targets_masked)

        optimizer.zero_grad()

        loss.backward()

        # if max_abs_grad:
        #    torch.nn.utils.clip_grad_value_(model.parameters(), max_abs_grad)

        optimizer.step()

        smoothed_loss = loss_EMA.update(loss.item())

        preds = torch.argmax(logits, dim=1)

        accuracy.update(preds, targets)
        masked_recall.update(preds, targets_masked)
        masked_accuracy.update(preds, targets_masked)
        masked_IQS.update(preds, targets_masked)

        if not silent:
            pbar.update(1)
            pbar.set_description(
                f'acc: {accuracy.compute():.4}, {print_class_recall(masked_recall.compute(), "masked recall: ")}, masked acc: {masked_accuracy.compute():.4}, masked IQS: {masked_IQS.compute():.4}, loss: {smoothed_loss:.4}'
            )

    if not silent:
        del pbar

    return smoothed_loss, accuracy.compute(), masked_accuracy.compute(), masked_recall.compute(), masked_IQS.compute()


def model_eval_diploid(model, dataloader, device, get_embeddings=False, temperature=None, silent=False):
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    accuracy = MaskedAccuracy().to(device)
    masked_recall = MeanRecall(Nclasses=4).to(device)
    masked_accuracy = MaskedAccuracy().to(device)
    masked_IQS = IQS(Nclasses=4).to(device)

    model.eval()  # model to train mode

    if not silent:
        tot_itr = len(dataloader.dataset) // dataloader.batch_size  # total train iterations
        pbar = tqdm(total=tot_itr, ncols=750)  # progress bar

    avg_loss = 0.0

    all_embeddings = []

    motif_probas = []

    with torch.no_grad():
        for itr_idx, ((masked_sequence, species_label), targets_masked, targets, seq) in enumerate(dataloader):
            if get_embeddings:
                if dataloader.batch_size != 1:
                    raise ValueError(
                        f"For genotypes and get_embeddings=True, the batch size must be 1 and not {dataloader.batch_size}!"
                    )
                # with batch size = 1, one batch contains the two sequences of one sample
                # let's extract both sequences and get the predictions of both
                # afterwards compute the scores for the combined predictions
                # masked sequence
                masked_sequence1 = torch.split(masked_sequence, split_size_or_sections=1, dim=0)[0][0]
                masked_sequence2 = torch.split(masked_sequence, split_size_or_sections=1, dim=0)[1][0]
                # targets_masked
                targets_masked1 = torch.split(targets_masked, split_size_or_sections=1, dim=0)[0][0]
                targets_masked2 = torch.split(targets_masked, split_size_or_sections=1, dim=0)[1][0]
                # targets
                targets1 = torch.split(targets, split_size_or_sections=1, dim=0)[0][0]
                targets2 = torch.split(targets, split_size_or_sections=1, dim=0)[1][0]
                species_label = species_label[0]
                species_label = species_label.tile((len(masked_sequence1),))

                masked_sequence1 = masked_sequence1.to(device)
                masked_sequence2 = masked_sequence2.to(device)
                targets_masked1 = targets_masked1.to(device)
                targets_masked2 = targets_masked2.to(device)
                targets1 = targets1.to(device)
                targets2 = targets2.to(device)
                species_label = species_label.long().to(device)

                logits1, embeddings1 = model(masked_sequence1, species_label)
                if temperature:
                    logits1 /= temperature
                loss1 = criterion(logits1, targets_masked1)
                avg_loss += loss1.item()
                preds1 = torch.argmax(logits1, dim=1)

                logits2, embeddings2 = model(masked_sequence2, species_label)
                if temperature:
                    logits2 /= temperature
                loss2 = criterion(logits2, targets_masked1)
                avg_loss += loss2.item()
                preds2 = torch.argmax(logits2, dim=1)

                #  combine the predictions and compute the metrics

                # "notation":
                # 0 = R
                # 1 = M
                # 2 = F
                # 3 = B
                # -100 = masked

                # combine preds
                combined_preds = preds1 + preds2
                combined_preds = torch.where(combined_preds == 2, combined_preds + 1, 0)  # == 3 if both, 0 otherwise
                temp = combined_preds + preds1  # == 4 if both, 1 if father
                combined_preds = torch.where(
                    temp == 1, temp + 1, combined_preds
                )  # == 3 if both, 2 if father, otherwise 0
                temp = combined_preds + preds2  # == 4 if both, 2 if father, 1 if mother
                combined_preds = torch.where(
                    temp == 1, temp, combined_preds
                )  # == 3 if both, 2 if father, 1 if mother, otherwise 0

                # combine targets
                combined_targets = targets1 + targets2
                combined_targets = torch.where(
                    combined_targets == 2, combined_targets + 1, 0
                )  # == 3 if both, 0 otherwise
                temp = combined_targets + targets1  # == 4 if both, 1 if father
                combined_targets = torch.where(
                    temp == 1, temp + 1, combined_targets
                )  # == 3 if both, 2 if father, otherwise 0
                temp = combined_targets + targets2  # == 4 if both, 2 if father, 1 if mother
                combined_targets = torch.where(
                    temp == 1, temp, combined_targets
                )  # == 3 if both, 2 if father, 1 if mother, otherwise 0

                # combine masked targets
                combined_targets_masked = targets_masked1 + targets_masked2
                combined_targets_masked = torch.where(
                    combined_targets_masked == 2, combined_targets_masked + 1, 0
                )  # == 3 if both, 0 otherwise
                temp = combined_targets_masked + targets_masked1  # == 4 if both, 1 if father
                combined_targets_masked = torch.where(
                    temp == 1, temp + 1, combined_targets_masked
                )  # == 3 if both, 2 if father, otherwise 0
                temp = combined_targets_masked + targets_masked2  # == 4 if both, 2 if father, 1 if mother
                combined_targets_masked = torch.where(
                    temp == 1, temp, combined_targets_masked
                )  # == 3 if both, 2 if father, 1 if mother, otherwise 0
                combined_targets_masked = torch.where(
                    temp == -100, temp, combined_targets_masked
                )  # == 3 if both, 2 if father, 1 if mother, -100 if masked, otherwise 0

                accuracy.update(combined_preds, combined_targets)
                masked_recall.update(combined_preds, combined_targets_masked)
                masked_accuracy.update(combined_preds, combined_targets_masked)
                masked_IQS.update(combined_preds, combined_targets_masked)

                seq_name = dataloader.dataset.seq_df.iloc[itr_idx].seq_name  # extract sequence ID

                # get embeddings of the masked nucleotide
                sequence_embedding1 = embeddings1["seq_embedding"]
                sequence_embedding1 = sequence_embedding1.transpose(-1, -2)[targets_masked1 != -100]
                sequence_embedding2 = embeddings2["seq_embedding"]
                sequence_embedding2 = sequence_embedding2.transpose(-1, -2)[targets_masked2 != -100]

                sequence_embedding1 = sequence_embedding1.mean(dim=0)  # if we mask
                sequence_embedding2 = sequence_embedding2.mean(dim=0)  # if we mask
                sequence_embedding1 = sequence_embedding1.detach().cpu().numpy()
                sequence_embedding2 = sequence_embedding2.detach().cpu().numpy()

                logits1 = torch.permute(logits1, (2, 0, 1)).reshape(-1, masked_sequence1.shape[1]).detach()
                targets_masked1 = targets_masked1.T.flatten()
                masked_targets1 = targets_masked1[targets_masked1 != -100].cpu()
                logits1 = logits1[targets_masked1 != -100].cpu()

                logits2 = torch.permute(logits2, (2, 0, 1)).reshape(-1, masked_sequence2.shape[1]).detach()
                targets_masked2 = targets_masked2.T.flatten()
                masked_targets2 = targets_masked2[targets_masked2 != -100].cpu()
                logits2 = logits2[targets_masked2 != -100].cpu()

                logprobs1 = log_softmax(logits1, dim=1).numpy()
                ground_truth_logprobs1 = np.array([logprobs1[idx, base] for idx, base in enumerate(masked_targets1)])
                all_embeddings.append((seq_name + ":father", sequence_embedding1, ground_truth_logprobs1))

                logprobs2 = log_softmax(logits2, dim=1).numpy()
                ground_truth_logprobs2 = np.array([logprobs2[idx, base] for idx, base in enumerate(masked_targets2)])
                all_embeddings.append((seq_name + ":mother", sequence_embedding2, ground_truth_logprobs2))

            else:
                masked_sequence = masked_sequence.to(device)
                species_label = species_label.long().to(device)
                targets_masked = targets_masked.to(device)
                targets = targets.to(device)
                logits, embeddings = model(masked_sequence, species_label)
                if temperature:
                    logits /= temperature
                loss = criterion(logits, targets_masked)
                avg_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                #  combine the predictions and compute the metrics
                # "notation":
                # 0 = R
                # 1 = M
                # 2 = F
                # 3 = B
                # -100 = masked
                targets = torch.split(
                    targets, split_size_or_sections=2, dim=0
                )  # each tensor contains predictions of both haplotypes
                targets_masked = torch.split(targets_masked, split_size_or_sections=2, dim=0)
                preds = torch.split(preds, split_size_or_sections=2, dim=0)
                combined_targets_all = []
                combined_targets_masked_all = []
                combined_preds_all = []
                for target, target_masked, pred in zip(targets, targets_masked, preds):
                    # targets
                    combined_targets = target[0] + target[1]
                    combined_targets = torch.where(
                        combined_targets == 2, combined_targets + 1, 0
                    )  # == 3 if both, 0 otherwise
                    temp = combined_targets + target[0]  # == 4 if both, 1 if father
                    combined_targets = torch.where(
                        temp == 1, temp + 1, combined_targets
                    )  # == 3 if both, 2 if father, otherwise 0
                    temp = combined_targets + target[1]  # == 4 if both, 2 if father, 1 if mother
                    combined_targets = torch.where(
                        temp == 1, temp, combined_targets
                    )  # == 3 if both, 2 if father, 1 if mother, otherwise 0
                    combined_targets_all.append(combined_targets)
                    # combine masked targets
                    combined_targets_masked = target_masked[0] + target_masked[1]
                    combined_targets_masked = torch.where(
                        combined_targets_masked == 2, combined_targets_masked + 1, 0
                    )  # == 3 if both, 0 otherwise
                    temp = combined_targets_masked + target_masked[0]  # == 4 if both, 1 if father
                    combined_targets_masked = torch.where(
                        temp == 1, temp + 1, combined_targets_masked
                    )  # == 3 if both, 2 if father, otherwise 0
                    temp = combined_targets_masked + target_masked[1]  # == 4 if both, 2 if father, 1 if mother
                    combined_targets_masked = torch.where(
                        temp == 1, temp, combined_targets_masked
                    )  # == 3 if both, 2 if father, 1 if mother, otherwise 0
                    combined_targets_masked = torch.where(
                        temp == -100, temp, combined_targets_masked
                    )  # == 3 if both, 2 if father, 1 if mother, -100 if masked, otherwise 0
                    combined_targets_masked_all.append(combined_targets_masked)
                    # combine preds
                    combined_preds = pred[0] + pred[1]
                    combined_preds = torch.where(combined_preds == 2, combined_preds + 1, 0)  # == 3 if both, 0 otherwise
                    temp = combined_preds + pred[0]  # == 4 if both, 1 if father
                    combined_preds = torch.where(
                        temp == 1, temp + 1, combined_preds
                    )  # == 3 if both, 2 if father, otherwise 0
                    temp = combined_preds + pred[1]  # == 4 if both, 2 if father, 1 if mother
                    combined_preds = torch.where(
                        temp == 1, temp, combined_preds
                    )  # == 3 if both, 2 if father, 1 if mother, otherwise 0
                    combined_preds_all.append(combined_preds)

                accuracy.update(torch.stack(combined_preds_all), torch.stack(combined_targets_all))
                masked_recall.update(torch.stack(combined_preds_all), torch.stack(combined_targets_masked_all))
                masked_accuracy.update(torch.stack(combined_preds_all), torch.stack(combined_targets_masked_all))
                masked_IQS.update(torch.stack(combined_preds_all), torch.stack(combined_targets_masked_all))

            if not silent:
                pbar.update(1)
                pbar.set_description(
                    f'acc: {accuracy.compute():.4}, {print_class_recall(masked_recall.compute(), "masked recall: ")}, masked acc: {masked_accuracy.compute():.4}, masked IQS: {masked_IQS.compute():.4}, loss: {avg_loss/(itr_idx+1):.4}'
                )

    if not silent:
        del pbar

    return (
        (
            avg_loss / (itr_idx + 1),
            accuracy.compute(),
            masked_accuracy.compute(),
            masked_recall.compute(),
            masked_IQS.compute(),
        ),
        all_embeddings,
        motif_probas,
    )


def model_eval(model, dataloader, device, diploid=False, get_embeddings=False, temperature=None, silent=False):
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    if diploid:
        return model_eval_diploid(
            model=model,
            dataloader=dataloader,
            device=device,
            get_embeddings=get_embeddings,
            temperature=temperature,
            silent=silent,
        )

    accuracy = MaskedAccuracy().to(device)
    masked_recall = MeanRecall().to(device)
    masked_accuracy = MaskedAccuracy().to(device)
    masked_IQS = IQS().to(device)

    model.eval()  # model to train mode

    if not silent:
        tot_itr = len(dataloader.dataset) // dataloader.batch_size  # total train iterations
        pbar = tqdm(total=tot_itr, ncols=750)  # progress bar

    avg_loss = 0.0

    all_embeddings = []

    motif_probas = []

    with torch.no_grad():
        for itr_idx, ((masked_sequence, species_label), targets_masked, targets, seq) in enumerate(dataloader):
            if get_embeddings:
                # batches are generated by transformation in the dataset,
                # so remove extra batch dimension added by dataloader
                masked_sequence, targets_masked, targets = masked_sequence[0], targets_masked[0], targets[0]
                species_label = species_label.tile((len(masked_sequence),))

            masked_sequence = masked_sequence.to(device)
            species_label = species_label.long().to(device)
            targets_masked = targets_masked.to(device)
            targets = targets.to(device)

            logits, embeddings = model(masked_sequence, species_label)

            if temperature:
                logits /= temperature

            loss = criterion(logits, targets_masked)

            avg_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            accuracy.update(preds, targets)
            masked_recall.update(preds, targets_masked)
            masked_accuracy.update(preds, targets_masked)
            masked_IQS.update(preds, targets_masked)

            if get_embeddings:
                seq_name = dataloader.dataset.seq_df.iloc[itr_idx].seq_name

                # only get embeddings of the masked nucleotide
                sequence_embedding = embeddings["seq_embedding"]
                sequence_embedding = sequence_embedding.transpose(-1, -2)[targets_masked != -100]
                # shape # B, L, dim  to L,dim, left with only masked nucleotide embeddings
                # average over sequence
                # print(sequence_embedding.shape)
                sequence_embedding = sequence_embedding.mean(dim=0)  # if we mask
                # sequence_embedding = sequence_embedding[0].mean(dim=-1) # no mask

                sequence_embedding = sequence_embedding.detach().cpu().numpy()

                logits = torch.permute(logits, (2, 0, 1)).reshape(-1, masked_sequence.shape[1]).detach()

                targets_masked = targets_masked.T.flatten()

                masked_targets = targets_masked[targets_masked != -100].cpu()
                logits = logits[targets_masked != -100].cpu()

                logprobs = log_softmax(logits, dim=1).numpy()

                # mapping = {'A':0,'C':1,'G':2,'T':3}
                # ground_truth_logprobs = np.array([logprobs[idx,mapping[base]] for idx,base in enumerate(seq[0])])

                ground_truth_logprobs = np.array([logprobs[idx, base] for idx, base in enumerate(masked_targets)])

                all_embeddings.append((seq_name, sequence_embedding, ground_truth_logprobs))

            if not silent:
                pbar.update(1)
                pbar.set_description(
                    f'acc: {accuracy.compute():.4}, {print_class_recall(masked_recall.compute(), "masked recall: ")}, masked acc: {masked_accuracy.compute():.4}, masked IQS: {masked_IQS.compute():.4}, loss: {avg_loss/(itr_idx+1):.4}'
                )

    if not silent:
        del pbar

    return (
        (
            avg_loss / (itr_idx + 1),
            accuracy.compute(),
            masked_accuracy.compute(),
            masked_recall.compute(),
            masked_IQS.compute(),
        ),
        all_embeddings,
        motif_probas,
    )
