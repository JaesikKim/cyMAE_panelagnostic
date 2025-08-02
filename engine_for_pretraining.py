# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from einops import rearrange

def simple_label_mapper(labels):
    cell_group_map = {
        # B Cells
        "Plasmablast": "Plasmablast",

        "IgDposMemB": "Mem B",
        "IgDnegMemB": "Mem B",

        "NaiveB": "NaiveB",
        # CD4+ T Cells
        "Th2/activated": "Th2",
        "Treg/activated": "Treg",
        "Treg": "Treg",
        "CD4Naive": "CD4Naive",
        "Th2": "Th2",
        "Th17": "Th17",
        "nnCD4CXCR5pos/activated": "CD4+ T",
        "Th1": "Th1",
        "Th1/activated": "Th1",
        "CD4Naive/activated": "CD4Naive",
        "Th17/activated": "Th17",
        "nnCD4CXCR5pos": "CD4+ T",

        # CD8+ T Cells
        "CD8Naive": "CD8Naive",
        "CD8TEM2": "CD8TEM",
        "CD8Naive/activated": "CD8Naive",
        "CD8TEMRA/activated": "CD8TEMRA",
        "CD8TEM3/activated": "CD8TEM",
        "CD8TEM2/activated": "CD8TEM",
        "CD8TEM1/activated": "CD8TEM",
        "CD8TEMRA": "CD8TEMRA",
        "CD8TCM/activated": "CD8TCM",
        "CD8TEM1": "CD8TEM",
        "CD8TEM3": "CD8TEM",
        "CD8TCM": "CD8TCM",
        # Other T Cells
        "DPT": "Other T",
        "MAITNKT": "Other T",
        "gdT": "Other T",
        "DNT": "Other T",
        "DNT/activated": "Other T",
        "DPT/activated": "Other T",

        # NK & ILC
        "EarlyNK": "NK & ILC",
        "LateNK": "NK & ILC",
        "ILC": "NK & ILC",

        # Monocytes & Dendritic Cells
        "pDC": "Dendritic Cell",
        "mDC": "Dendritic Cell",
        "ClassicalMono": "Monocyte",
        "TotalMonocyte": "Monocyte",

        # Granulocytes
        "CD66bnegCD45lo": "Other Granulocyte", # Or Other/Debris
        "CD45hiCD66bpos": "Other Granulocyte",

        "Basophil": "Basophil",
        "Eosinophil": "Eosinophil",
        "Neutrophil": "Neutrophil",
    }

    return [cell_group_map[l] for l in labels]

# label_to_idx = {label: idx for idx, label in enumerate(["Plasmablast", "Th2/activated", "Treg/activated", "CD8Naive", "Treg", "EarlyNK", "CD66bnegCD45lo", "CD4Naive", "Th2", "CD8TEM2", "Th17", "IgDposMemB", "CD8Naive/activated", "CD8TEMRA/activated", "Eosinophil", "CD8TEM3/activated", "DPT", "MAITNKT", "gdT", "CD8TEM2/activated", "nnCD4CXCR5pos/activated", "IgDnegMemB", "CD45hiCD66bpos", "LateNK", "Neutrophil", "DNT", "Basophil", "pDC", "CD8TEM1/activated", "mDC", "Th1", "DNT/activated", "Th1/activated", "CD8TEMRA", "CD8TCM/activated", "CD8TEM1", "CD4Naive/activated", "NaiveB", "ILC", "CD8TEM3", "Th17/activated", "CD8TCM", "ClassicalMono", "DPT/activated", "nnCD4CXCR5pos", "TotalMonocyte"])}
label_to_idx = {label: idx for idx, label in enumerate(["Plasmablast", "Mem B", "NaiveB","Th2", "Treg", "CD4Naive", "Th17", "Th1", "CD4+ T", "CD8Naive", "CD8TEM", "CD8TEMRA", "CD8TCM", "Other T", "NK & ILC", "Dendritic Cell", "Monocyte", "Other Granulocyte", "Basophil", "Eosinophil", "Neutrophil"])}

def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        data, labels, marker_indices = batch 

        data = data.to(device, non_blocking=True).to(torch.float)

        if args.mode == 'train': 
            with torch.amp.autocast('cuda'):
                loss = model(data, marker_indices)
        elif args.mode == 'linear_probing_train':
            numeric_labels = [[label_to_idx[l] for l in simple_label_mapper(ls)] for ls in labels]
            labels = torch.tensor(numeric_labels).to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                loss = model.module.forward_linear_probing_train(data, marker_indices, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        loss_scale_value = loss_scaler.state_dict()["scale"]

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    metric_logger.synchronize_between_processes()

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



# def inference_one_epoch(args, model, data_loader, device):
#     model.eval()

#     all_original_data = []
#     all_cell_embeddings = []
#     all_pooled_embeddings = []
#     all_panel_labels = []
#     all_cell_labels = []
#     all_cell_labels_pred = []
#     with torch.no_grad():
#         for step, batch in enumerate(data_loader):
#             print(step)
#             data, labels, marker_indices = batch
#             panel_label = ','.join(np.array(args.union_marker_list)[marker_indices[0]])
#             data = data.to(device, non_blocking=True).to(torch.float)
            
#             if args.mode == 'inference':
#                 with torch.amp.autocast('cuda'):
#                     cell_embeddings, pooled_embeddings = model.forward_inference(data, marker_indices)
#                 all_original_data.append(data[0,:,:].cpu())
#                 all_cell_embeddings.append(cell_embeddings.cpu())
#                 all_pooled_embeddings.append(pooled_embeddings.cpu())

#             elif args.mode == 'linear_probing_inference':
#                 with torch.amp.autocast('cuda'):
#                     pred = model.forward_linear_probing_inference(data, marker_indices)
#                 all_cell_labels_pred += pred

#             all_cell_labels += labels
#             all_panel_labels.append(panel_label)
            
#     return all_original_data, all_cell_embeddings, all_pooled_embeddings, all_panel_labels, all_cell_labels, all_cell_labels_pred
    