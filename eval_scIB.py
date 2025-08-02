import numpy as np
import pandas as pd
import scanpy as sc
import scib
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import read_file
from run_mae_pretraining import get_model

import os

def metrics_for_cytometry(adata, adata_int, batch_key, label_key, **kwargs):
    """All metrics

    :Biological conservation:
        - HVG overlap :func:`~scib.metrics.hvg_overlap`
        + Cell type ASW :func:`~scib.metrics.silhouette`
        + Isolated label ASW :func:`~scib.metrics.isolated_labels`
        + Isolated label F1 :func:`~scib.metrics.isolated_labels`
        + NMI cluster/label :func:`~scib.metrics.nmi`
        + ARI cluster/label :func:`~scib.metrics.ari`
        - Cell cycle conservation :func:`~scib.metrics.cell_cycle`
        + cLISI (cell type Local Inverse Simpson's Index) :func:`~scib.metrics.clisi_graph`
        - Trajectory conservation :func:`~scib.metrics.trajectory_conservation`

    :Batch correction:
        + Graph connectivity :func:`~scib.metrics.graph_connectivity`
        + Batch ASW :func:`~scib.metrics.silhouette_batch`
        + Principal component regression :func:`~scib.metrics.pcr_comparison`
        + kBET (k-nearest neighbour batch effect test) :func:`~scib.metrics.kBET`
        + iLISI (integration Local Inverse Simpson's Index) :func:`~scib.metrics.ilisi_graph`

    :param adata: unintegrated, preprocessed anndata object
    :param adata_int: integrated anndata object
    :param batch_key: name of batch column in adata.obs and adata_int.obs
    :param label_key: name of biological label (cell type) column in adata.obs and adata_int.obs
    :param kwargs:
        Parameters to pass on to :func:`~scib.metrics.metrics` function:

            + ``embed``
            + ``cluster_key``
            + ``cluster_nmi``
            + ``nmi_method``
            + ``nmi_dir``
            + ``si_metric``
            + ``organism``
            + ``n_isolated``
            + ``subsample``
            + ``type_``
    """
    return scib.metrics.metrics(
        adata,
        adata_int,
        batch_key,
        label_key,
        isolated_labels_asw_=True,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=False,
        pcr_=False,
        isolated_labels_f1_=True,
        trajectory_=False,
        nmi_=True,
        ari_=True,
        cell_cycle_=False,
        kBET_=False,
        ilisi_=False,
        clisi_=False,
        **kwargs,
    )


def metrics(original: np.array, embeddings: np.array, batch: list[str], label: list[str]) -> tuple[float, float]:
    """
    Calculate batch and biological conservation scores using scib.

    :param original: np.array of input data (feature matrix)
    :param embeddings: np.array of integrated or embedded data (embedding matrix)
    :param batch: Batch labels as a list of integers
    :param label: Biological labels as a list of integers

    :return: Batch and biological conservation scores as a tuple
    """

    # Step 1: Create AnnData object for original data with batch and label
    adata = sc.AnnData(original)
    adata.obs["batch"] = pd.Categorical(batch)
    adata.obs["label"] = pd.Categorical(label)

    # Step 2: Create AnnData object for integrated data (embedding space)
    adata_int = sc.AnnData(original)
    adata_int.obs["batch"] = pd.Categorical(batch)
    adata_int.obs["label"] = pd.Categorical(label)
    adata_int.obsm["X_emb"] = embeddings  # Store embeddings in obsm for metrics to use

    # Step 3: Generate kNN graph on embedding space for graph-based metrics
    sc.pp.neighbors(adata_int, use_rep="X_emb")

    # Step 4: Calculate metrics
    return metrics_for_cytometry(adata, adata_int, batch_key="batch", label_key="label", embed="X_emb")


from scgraph import scGraph

def scGraph_metrics(original: np.array, embeddings: np.array, batch: list[str], label: list[str]) -> tuple[float, float]:
    """
    Calculate batch and biological conservation scores using scib.

    :param original: np.array of input data (feature matrix)
    :param embeddings: np.array of integrated or embedded data (embedding matrix)
    :param batch: Batch labels as a list of integers
    :param label: Biological labels as a list of integers

    :return: Batch and biological conservation scores as a tuple
    """

    # Step 1: Create AnnData object for original data with batch and label
    adata = sc.AnnData(original)
    adata.obs["batch"] = pd.Categorical(batch)
    adata.obs["label"] = pd.Categorical(label)
    adata.obsm["X_emb"] = embeddings  # Store embeddings in obsm for metrics to use

    adata.write('data.h5ad')

    # Initialize the graph analyzer
    scgraph = scGraph(
        adata_path="data.h5ad",   # Path to AnnData object
        batch_key="batch",                     # Column name for batch information
        label_key="label",                 # Column name for cell type labels
        trim_rate=0.05,                        # Trim rate for robust mean calculation
        thres_batch=100,                       # Minimum number of cells per batch
        thres_celltype=10,                      # Minimum number of cells per cell type
        only_umap=True,                        # Only evaluate 2D embeddings (mostly umaps)
    )

    # Run the analysis, return a pandas dataframe
    results = scgraph.main()
    return results


def undersample_above_target(labels, target_count=None):
    """
    - target_count 미지정 시: 클래스별 개수의 최소값(min)을 목표로 삼음
    - 각 클래스 count > target_count 인 경우에만 replace=False 로 undersample
    - count <= target_count 인 경우, 모든 샘플을 사용
    """
    labels_arr = np.array(labels)
    unique_labels, counts = np.unique(labels_arr, return_counts=True)
    
    # target_count 결정
    if target_count is None:
        target_count = int(np.min(counts))
        print(target_count)
    
    selected_idx = []
    for lbl, cnt in zip(unique_labels, counts):
        idx = np.where(labels_arr == lbl)[0]
        if cnt > target_count:
            # undersample: replace=False
            pick = rng.choice(idx, size=target_count, replace=False)
        else:
            # 그대로 모두 사용
            pick = idx

        selected_idx.append(pick)
    
    selected_idx = np.concatenate(selected_idx)
    rng.shuffle(selected_idx)  # 순서 섞기

    return selected_idx

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
        "CD8TEM2": "CD8TEM2",
        "CD8Naive/activated": "CD8Naive",
        "CD8TEMRA/activated": "CD8TEMRA",
        "CD8TEM3/activated": "CD8TEM3",
        "CD8TEM2/activated": "CD8TEM2",
        "CD8TEM1/activated": "CD8TEM1",
        "CD8TEMRA": "CD8TEMRA",
        "CD8TCM/activated": "CD8TCM",
        "CD8TEM1": "CD8TEM1",
        "CD8TEM3": "CD8TEM3",
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



# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.0_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.5_maxstep_1_celllambda_0.0_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_1.0_maxstep_1_celllambda_0.0_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_2.0_maxstep_1_celllambda_0.0_lr_0.005_checkpoint-6000.pth"

# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_-0.5_maxstep_1_celllambda_0.0_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_-1.0_maxstep_1_celllambda_0.0_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_-2.0_maxstep_1_celllambda_0.0_lr_0.005_checkpoint-6000.pth"

# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.5_maxstep_1_celllambda_0.5_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_-0.5_maxstep_1_celllambda_0.5_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_1.0_maxstep_1_celllambda_0.5_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_-1.0_maxstep_1_celllambda_0.5_lr_0.005_checkpoint-6000.pth"


# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.0_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.05_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.1_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.2_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.5_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_1.0_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_2.0_lr_0.005_checkpoint-6000.pth"
# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_5.0_lr_0.005_checkpoint-6000.pth"

ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_cumul_masking_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.0_lr_0.005_checkpoint-6000.pth"


device = 'cuda'

checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
ckpt_args = checkpoint['args']
ckpt_args.is_cumul_masking = True

union_marker_to_index = {
    marker: idx for idx, marker in enumerate(ckpt_args.union_marker_list)
}

model = get_model(ckpt_args)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()


panel_A = ["CD45", "CD123", "CD19", "CD4", "CD8a",
            "CD11c", "CD16", "CD161", "CD57", "CD38",
            "CD56", "CD294", "CD14", "CD3", "CD20",
            "CD66b", "HLA-DR", "IgD", "TCRgd", "CD45RA"]

panel_B = ["CD45", "CD196", "CD4", "CD8a", "CD11c", 
            "CD161", "CD45RO", "CD45RA", "CD194", "CD25",
            "CD27", "CD57", "CD183", "CD185", "CD38",
            "CD294", "CD197", "CD14", "CD3", "HLA-DR", 
            "TCRgd"]

intersection_AB = list(set(panel_A) & set(panel_B))

panel_C = ["CD45", "CD196", "CD123", "CD19", "CD4", "CD8a",
            "CD11c", "CD16", "CD45RO", "CD45RA", "CD161", 
            "CD194", "CD25", "CD27",  "CD57", "CD183", "CD185",
            "CD38", "CD56", "TCRgd", "CD294", "CD197", "CD14", "CD3", "CD20",
            "CD66b", "HLA-DR", "IgD"]


panel_D = ["CD45", "CD196", "CD19", "CD4", "CD8a", 
            "CD25", "CD27", "CD183", 
            "CD56", "CD3", "CD20", "IgD",
            "CD16", "CD11c", "CD14", "CD45RO"] 

test_ori_scores = []
test_ori_avgbios = []
test_cymae_scores = []
test_cymae_avgbios = []
ext_ori_scores = []
ext_ori_avgbios = []
ext_cymae_scores = []
ext_cymae_avgbios = []
combined_ori_scores = []
combined_ori_avgbios = []
combined_ori_avgbatch = []
combined_cymae_scores = []
combined_cymae_avgbios = []
combined_cymae_avgbatch = []
for seed in [42,43,44,45,46]:
    rng = np.random.default_rng(seed=seed)

    # running on original data
    path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/test_panel_C/'

    test_filenames = os.listdir(path)

    test_ori_data = []
    test_cymae_embeddings = []
    test_labels = []

    for filename in test_filenames:
        data, labels, marker_list = read_file(path+filename)
        labels = simple_label_mapper(labels)

        # selected_idx = undersample_above_target(labels, target_count=50)
        selected_idx = rng.choice(len(labels), size=1000, replace=False)
        data = data[selected_idx]
        labels = np.array(labels)[selected_idx].tolist()

        # panel A
        marker_order = [i for i,m in enumerate(marker_list) if m in panel_A]
        data = data[:, marker_order]
        marker_list = [marker_list[i] for i in marker_order]

        # # panel B
        # marker_order = [i for i,m in enumerate(marker_list) if m in panel_B]
        # data = data[:, marker_order]
        # marker_list = [marker_list[i] for i in marker_order]

        # # # panel C
        # data = data
        # marker_list = marker_list

        # # panel D
        # marker_order = [i for i,m in enumerate(marker_list) if m in panel_D]
        # data = data[:, marker_order]
        # marker_list = [marker_list[i] for i in marker_order]


        test_ori_data.append(data)
        test_labels += labels

        marker_indices = []
        for marker in marker_list:
            if marker in ckpt_args.union_marker_list:
                marker_indices.append(union_marker_to_index[marker])

        data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                cell_embeddings, pooled_embeddings = model.forward_inference(data, marker_indices)

        test_cymae_embeddings.append(cell_embeddings.cpu())


    test_ori_data = torch.cat(test_ori_data)
    test_cymae_embeddings = torch.cat(test_cymae_embeddings)
    print(test_ori_data.shape, test_cymae_embeddings.shape, len(test_labels))


    path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/ext_panel_C/'
    test_filenames = os.listdir(path)

    ext_ori_data = []
    ext_cymae_embeddings = []
    ext_labels = []
    for filename in test_filenames:
        data, labels, marker_list = read_file(path+filename)
        labels = simple_label_mapper(labels)

        # selected_idx = undersample_above_target(labels, target_count=50)
        selected_idx = rng.choice(len(labels), size=1000, replace=False)
        data = data[selected_idx]
        labels = np.array(labels)[selected_idx].tolist()

        # panel A
        marker_order = [i for i,m in enumerate(marker_list) if m in panel_A]
        data = data[:, marker_order]
        marker_list = [marker_list[i] for i in marker_order]

        # # panel B
        # marker_order = [i for i,m in enumerate(marker_list) if m in panel_B]
        # data = data[:, marker_order]
        # marker_list = [marker_list[i] for i in marker_order]

        # # panel C
        # data = data
        # marker_list = marker_list

        # # panel D
        # marker_order = [i for i,m in enumerate(marker_list) if m in panel_D]
        # data = data[:, marker_order]
        # marker_list = [marker_list[i] for i in marker_order]


        ext_ori_data.append(data)
        ext_labels += labels

        marker_indices = []
        for marker in marker_list:
            if marker in ckpt_args.union_marker_list:
                marker_indices.append(union_marker_to_index[marker])

        data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                cell_embeddings, pooled_embeddings = model.forward_inference(data, marker_indices)

        ext_cymae_embeddings.append(cell_embeddings.cpu())

    ext_ori_data = torch.cat(ext_ori_data)
    ext_cymae_embeddings = torch.cat(ext_cymae_embeddings)
    print(ext_ori_data.shape, ext_cymae_embeddings.shape, len(ext_labels))

    combined_ori_data = torch.cat((test_ori_data, ext_ori_data))
    combined_labels = test_labels + ext_labels
    combined_cymae_embeddings = torch.cat((test_cymae_embeddings, ext_cymae_embeddings))
    print(combined_ori_data.shape, combined_cymae_embeddings.shape, len(combined_labels))

    combined_batches = ['test']*len(test_labels) + ['ext']*len(ext_labels)

    # results = scGraph_metrics(test_cymae_embeddings_balanced.numpy(),test_cymae_embeddings_balanced.numpy(),['batch' for _ in range(len(test_cymae_labels_balanced))], test_cymae_labels_balanced)
    # print(results)

    # score = metrics(test_ori_data.numpy(), test_ori_data.numpy(),['batch' for _ in range(len(test_labels))], test_labels)
    # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
    # test_ori_scores.append(score)
    # test_ori_avgbios.append(AvgBIO)
    # print("test_ori", AvgBIO)

    score = metrics(test_cymae_embeddings.numpy(), test_cymae_embeddings.numpy(),['batch' for _ in range(len(test_labels))], test_labels)
    AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
    test_cymae_scores.append(score)
    test_cymae_avgbios.append(AvgBIO)
    print("test_cymae", AvgBIO)

    # score = metrics(ext_ori_data.numpy(), ext_ori_data.numpy(),['batch' for _ in range(len(ext_labels))], ext_labels)
    # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
    # ext_ori_scores.append(score)
    # ext_ori_avgbios.append(AvgBIO)
    # print("ext_ori", AvgBIO)

    score = metrics(ext_cymae_embeddings.numpy(), ext_cymae_embeddings.numpy(),['batch' for _ in range(len(ext_labels))], ext_labels)
    AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
    ext_cymae_scores.append(score)
    ext_cymae_avgbios.append(AvgBIO)
    print("ext_cymae", AvgBIO)

    # score = metrics(combined_ori_data.numpy(), combined_ori_data.numpy(),combined_batches, combined_labels)
    # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label', 'isolated_label_F1', 'isolated_label_silhouette']].mean(axis=0)
    # AvgBatch = score.loc['ASW_label/batch']
    # combined_ori_scores.append(score)
    # combined_ori_avgbios.append(AvgBIO)
    # combined_ori_avgbatch.append(AvgBatch)
    # print("combined_ori", AvgBIO, AvgBatch)

    score = metrics(combined_cymae_embeddings.numpy(), combined_cymae_embeddings.numpy(), combined_batches, combined_labels)
    AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label', 'isolated_label_F1', 'isolated_label_silhouette']].mean(axis=0)
    AvgBatch = score.loc['ASW_label/batch']
    combined_cymae_scores.append(score)
    combined_cymae_avgbios.append(AvgBIO)
    combined_cymae_avgbatch.append(AvgBatch)
    print("combined_cymae", AvgBIO, AvgBatch)

# test_ori_scores = pd.concat(test_ori_scores, axis=1)
# print(test_ori_scores.mean(axis=1))

test_cymae_scores = pd.concat(test_cymae_scores, axis=1)
print(test_cymae_scores.mean(axis=1))

# ext_ori_scores = pd.concat(ext_ori_scores, axis=1)
# print(ext_ori_scores.mean(axis=1))

ext_cymae_scores = pd.concat(ext_cymae_scores, axis=1)
print(ext_cymae_scores.mean(axis=1))

# combined_ori_scores = pd.concat(combined_ori_scores, axis=1)
# print(combined_ori_scores.mean(axis=1))

combined_cymae_scores = pd.concat(combined_cymae_scores, axis=1)
print(combined_cymae_scores.mean(axis=1))

# print(np.mean(test_ori_avgbios))
print(np.mean(test_cymae_avgbios))
# print(np.mean(ext_ori_avgbios))
print(np.mean(ext_cymae_avgbios))
# print(np.mean(combined_ori_avgbios), np.mean(combined_ori_avgbatch))
print(np.mean(combined_cymae_avgbios), np.mean(combined_cymae_avgbatch))










# test_anchor_scores = []
# test_anchor_avgbios = []
# test_cymae_scores = []
# test_cymae_avgbios = []
# ext_anchor_scores = []
# ext_anchor_avgbios = []
# ext_cymae_scores = []
# ext_cymae_avgbios = []
# combined_anchor_scores = []
# combined_anchor_avgbios = []
# combined_anchor_avgbatch = []
# combined_cymae_scores = []
# combined_cymae_avgbios = []
# combined_cymae_avgbatch = []
# for seed in [42,43,44,45,46]:
#     rng = np.random.default_rng(seed=seed)

#     # running on original data
#     path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/test_panel_C/'

#     test_panel_A_filenames = os.listdir(path)[:2]
#     test_panel_B_filenames = os.listdir(path)[2:]

#     test_anchor_data = []
#     test_cymae_embeddings = []
#     test_labels = []

#     for filename in test_panel_A_filenames:
#         data, labels, marker_list = read_file(path+filename)
#         labels = simple_label_mapper(labels)

#         mask = [i for i,l in enumerate(labels) if l in ["Th2", "Treg", "Th17", "Th1", "CD4Naive", "CD4+ T", "CD8Naive", "CD8TEM1", "CD8TEM2", "CD8TEM3", "CD8TEMRA", "CD8TCM"]]
#         data = data[mask, :]
#         labels = [labels[i] for i in mask]

#         selected_idx = undersample_above_target(labels, target_count=50)
#         # selected_idx = rng.choice(len(labels), size=1000, replace=False)
#         data = data[selected_idx]
#         labels = np.array(labels)[selected_idx].tolist()

#         # anchor markers
#         marker_order = [i for i,m in enumerate(marker_list) if m in intersection_AB]
#         test_anchor_data.append(data[:, marker_order])
#         test_labels += labels

#         # panel A
#         marker_order = [i for i,m in enumerate(marker_list) if m in panel_A]
#         data = data[:, marker_order]
#         marker_list = [marker_list[i] for i in marker_order]

#         marker_indices = []
#         for marker in marker_list:
#             if marker in ckpt_args.union_marker_list:
#                 marker_indices.append(union_marker_to_index[marker])

#         data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#         with torch.no_grad():
#             with torch.amp.autocast('cuda'):
#                 cell_embeddings, pooled_embeddings = model.forward_inference(data, marker_indices)

#         test_cymae_embeddings.append(cell_embeddings.cpu())

#     for filename in test_panel_B_filenames:
#         data, labels, marker_list = read_file(path+filename)
#         labels = simple_label_mapper(labels)

#         mask = [i for i,l in enumerate(labels) if l in ["Th2", "Treg", "Th17", "Th1", "CD4Naive", "CD4+ T", "CD8Naive", "CD8TEM1", "CD8TEM2", "CD8TEM3", "CD8TEMRA", "CD8TCM"]]
#         data = data[mask, :]
#         labels = [labels[i] for i in mask]
        
#         selected_idx = undersample_above_target(labels, target_count=50)
#         # selected_idx = rng.choice(len(labels), size=1000, replace=False)
#         data = data[selected_idx]
#         labels = np.array(labels)[selected_idx].tolist()

#         # anchor markers
#         marker_order = [i for i,m in enumerate(marker_list) if m in intersection_AB]
#         test_anchor_data.append(data[:, marker_order])
#         test_labels += labels

#         # panel B
#         marker_order = [i for i,m in enumerate(marker_list) if m in panel_B]
#         data = data[:, marker_order]
#         marker_list = [marker_list[i] for i in marker_order]

#         marker_indices = []
#         for marker in marker_list:
#             if marker in ckpt_args.union_marker_list:
#                 marker_indices.append(union_marker_to_index[marker])

#         data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#         with torch.no_grad():
#             with torch.amp.autocast('cuda'):
#                 cell_embeddings, pooled_embeddings = model.forward_inference(data, marker_indices)

#         test_cymae_embeddings.append(cell_embeddings.cpu())


#     test_anchor_data = torch.cat(test_anchor_data)
#     test_cymae_embeddings = torch.cat(test_cymae_embeddings)
#     print(test_anchor_data.shape, test_cymae_embeddings.shape, len(test_labels))


#     path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/ext_panel_C/'

#     ext_panel_A_filenames = os.listdir(path)[:2]
#     ext_panel_B_filenames = os.listdir(path)[2:]

#     ext_anchor_data = []
#     ext_cymae_embeddings = []
#     ext_labels = []

#     for filename in ext_panel_A_filenames:
#         data, labels, marker_list = read_file(path+filename)
#         labels = simple_label_mapper(labels)

#         mask = [i for i,l in enumerate(labels) if l in ["Th2", "Treg", "Th17", "Th1", "CD4Naive", "CD4+ T", "CD8Naive", "CD8TEM1", "CD8TEM2", "CD8TEM3", "CD8TEMRA", "CD8TCM"]]
#         data = data[mask, :]
#         labels = [labels[i] for i in mask]

#         selected_idx = undersample_above_target(labels, target_count=50)
#         # selected_idx = rng.choice(len(labels), size=1000, replace=False)
#         data = data[selected_idx]
#         labels = np.array(labels)[selected_idx].tolist()

#         # anchor markers
#         marker_order = [i for i,m in enumerate(marker_list) if m in intersection_AB]
#         ext_anchor_data.append(data[:, marker_order])
#         ext_labels += labels

#         # panel A
#         marker_order = [i for i,m in enumerate(marker_list) if m in panel_A]
#         data = data[:, marker_order]
#         marker_list = [marker_list[i] for i in marker_order]

#         marker_indices = []
#         for marker in marker_list:
#             if marker in ckpt_args.union_marker_list:
#                 marker_indices.append(union_marker_to_index[marker])

#         data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#         with torch.no_grad():
#             with torch.amp.autocast('cuda'):
#                 cell_embeddings, pooled_embeddings = model.forward_inference(data, marker_indices)

#         ext_cymae_embeddings.append(cell_embeddings.cpu())

#     for filename in ext_panel_B_filenames:
#         data, labels, marker_list = read_file(path+filename)
#         labels = simple_label_mapper(labels)

#         mask = [i for i,l in enumerate(labels) if l in ["Th2", "Treg", "Th17", "Th1", "CD4Naive", "CD4+ T", "CD8Naive", "CD8TEM1", "CD8TEM2", "CD8TEM3", "CD8TEMRA", "CD8TCM"]]
#         data = data[mask, :]
#         labels = [labels[i] for i in mask]
        
#         selected_idx = undersample_above_target(labels, target_count=50)
#         # selected_idx = rng.choice(len(labels), size=1000, replace=False)
#         data = data[selected_idx]
#         labels = np.array(labels)[selected_idx].tolist()

#         # anchor markers
#         marker_order = [i for i,m in enumerate(marker_list) if m in intersection_AB]
#         ext_anchor_data.append(data[:, marker_order])
#         ext_labels += labels

#         # panel B
#         marker_order = [i for i,m in enumerate(marker_list) if m in panel_B]
#         data = data[:, marker_order]
#         marker_list = [marker_list[i] for i in marker_order]

#         marker_indices = []
#         for marker in marker_list:
#             if marker in ckpt_args.union_marker_list:
#                 marker_indices.append(union_marker_to_index[marker])

#         data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#         with torch.no_grad():
#             with torch.amp.autocast('cuda'):
#                 cell_embeddings, pooled_embeddings = model.forward_inference(data, marker_indices)

#         ext_cymae_embeddings.append(cell_embeddings.cpu())


#     ext_anchor_data = torch.cat(ext_anchor_data)
#     ext_cymae_embeddings = torch.cat(ext_cymae_embeddings)
#     print(ext_anchor_data.shape, ext_cymae_embeddings.shape, len(ext_labels))



#     combined_anchor_data = torch.cat((test_anchor_data, ext_anchor_data))
#     combined_labels = test_labels + ext_labels
#     combined_cymae_embeddings = torch.cat((test_cymae_embeddings, ext_cymae_embeddings))
#     print(combined_anchor_data.shape, combined_cymae_embeddings.shape, len(combined_labels))

#     combined_batches = ['test']*len(test_labels) + ['ext']*len(ext_labels)

#     # results = scGraph_metrics(test_cymae_embeddings_balanced.numpy(),test_cymae_embeddings_balanced.numpy(),['batch' for _ in range(len(test_cymae_labels_balanced))], test_cymae_labels_balanced)
#     # print(results)

#     score = metrics(test_anchor_data.numpy(), test_anchor_data.numpy(),['batch' for _ in range(len(test_labels))], test_labels)
#     AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     test_anchor_scores.append(score)
#     test_anchor_avgbios.append(AvgBIO)
#     print("test_anchor", AvgBIO)

#     score = metrics(test_cymae_embeddings.numpy(), test_cymae_embeddings.numpy(),['batch' for _ in range(len(test_labels))], test_labels)
#     AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     test_cymae_scores.append(score)
#     test_cymae_avgbios.append(AvgBIO)
#     print("test_cymae", AvgBIO)

#     score = metrics(ext_anchor_data.numpy(), ext_anchor_data.numpy(),['batch' for _ in range(len(ext_labels))], ext_labels)
#     AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     ext_anchor_scores.append(score)
#     ext_anchor_avgbios.append(AvgBIO)
#     print("ext_anchor", AvgBIO)

#     score = metrics(ext_cymae_embeddings.numpy(), ext_cymae_embeddings.numpy(),['batch' for _ in range(len(ext_labels))], ext_labels)
#     AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     ext_cymae_scores.append(score)
#     ext_cymae_avgbios.append(AvgBIO)
#     print("ext_cymae", AvgBIO)

#     score = metrics(combined_anchor_data.numpy(), combined_anchor_data.numpy(),combined_batches, combined_labels)
#     AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label', 'isolated_label_F1', 'isolated_label_silhouette']].mean(axis=0)
#     AvgBatch = score.loc['ASW_label/batch']
#     combined_anchor_scores.append(score)
#     combined_anchor_avgbios.append(AvgBIO)
#     combined_anchor_avgbatch.append(AvgBatch)
#     print("combined_anchor", AvgBIO, AvgBatch)

#     score = metrics(combined_cymae_embeddings.numpy(), combined_cymae_embeddings.numpy(), combined_batches, combined_labels)
#     AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label', 'isolated_label_F1', 'isolated_label_silhouette']].mean(axis=0)
#     AvgBatch = score.loc['ASW_label/batch']
#     combined_cymae_scores.append(score)
#     combined_cymae_avgbios.append(AvgBIO)
#     combined_cymae_avgbatch.append(AvgBatch)
#     print("combined_cymae", AvgBIO, AvgBatch)

# test_anchor_scores = pd.concat(test_anchor_scores, axis=1)
# print(test_anchor_scores.mean(axis=1))

# test_cymae_scores = pd.concat(test_cymae_scores, axis=1)
# print(test_cymae_scores.mean(axis=1))

# ext_anchor_scores = pd.concat(ext_anchor_scores, axis=1)
# print(ext_anchor_scores.mean(axis=1))

# ext_cymae_scores = pd.concat(ext_cymae_scores, axis=1)
# print(ext_cymae_scores.mean(axis=1))

# combined_anchor_scores = pd.concat(combined_anchor_scores, axis=1)
# print(combined_anchor_scores.mean(axis=1))

# combined_cymae_scores = pd.concat(combined_cymae_scores, axis=1)
# print(combined_cymae_scores.mean(axis=1))

# print(np.mean(test_anchor_avgbios))
# print(np.mean(test_cymae_avgbios))
# print(np.mean(ext_anchor_avgbios))
# print(np.mean(ext_cymae_avgbios))
# print(np.mean(combined_anchor_avgbios), np.mean(combined_anchor_avgbatch))
# print(np.mean(combined_cymae_avgbios), np.mean(combined_cymae_avgbatch))

















# # simulation simple

# ckpt = "./ckpts/simulation_simple/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_1.0_maxstep_2_celllambda_0.5_lr_0.005_checkpoint-5000.pth"
# device = 'cuda'

# checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
# ckpt_args = checkpoint['args']
# union_marker_to_index = {
#     marker: idx for idx, marker in enumerate(ckpt_args.union_marker_list)
# }

# model = get_model(ckpt_args)
# model.load_state_dict(checkpoint['model'])
# model.to(device)
# model.eval()


# rng = np.random.default_rng(seed=42)




# # running on original data
# path = '/project/kimgroup_immune_health/data/pan_panel/simulation_simple/test/'
# test_filenames = ['MDIPA_AALC_18_V1_subset.h5', 'MDIPA_AALC_19_V1_subset.h5']

# test_ori_data = []
# test_ori_labels = []
# for filename in test_filenames:
#     data, labels, marker_list = read_file(path+filename)

#     test_ori_data.append(data)
#     test_ori_labels += labels
# test_ori_data = torch.cat(test_ori_data)
# test_ori_labels = np.array(test_ori_labels)
# print(test_ori_data.shape, len(test_ori_labels))


# # running on knn imputation
# path = '/project/kimgroup_immune_health/data/pan_panel/simulation_simple/knn_imputed_test/'
# test_filenames = ['MDIPA_AALC_18_V1_subset_imputed.h5', 'MDIPA_AALC_19_V1_subset_imputed.h5']

# test_knn_data = []
# test_knn_labels = []
# for filename in test_filenames:
#     data, labels, marker_list = read_file(path+filename)

#     test_knn_data.append(data)
#     test_knn_labels += labels
# test_knn_data = torch.cat(test_knn_data)
# test_knn_labels = np.array(test_knn_labels)
# print(test_knn_data.shape, len(test_knn_labels))


# # # running on cyCombine imputation
# # path = '/project/kimgroup_immune_health/data/pan_panel/simulation/cyCombine_imputed_test/'
# # test_filenames = ['MDIPA_AALC_18_V1_subsetC_imputed.h5', 'MDIPA_AALC_19_V1_subsetC_imputed.h5']

# # test_cycombine_data = []
# # test_cycombine_labels = []
# # for filename in test_filenames:
# #     data, labels, marker_list = read_file(path+filename)

# #     test_cycombine_data.append(data)
# #     test_cycombine_labels += labels
# # test_cycombine_data = torch.cat(test_cycombine_data)
# # test_cycombine_labels = np.array(test_cycombine_labels)
# # print(test_cycombine_data.shape, len(test_cycombine_labels))


# # running on cyMAE
# path = '/project/kimgroup_immune_health/data/pan_panel/simulation_simple/test/'
# test_filenames = ['MDIPA_AALC_18_V1_subset.h5', 'MDIPA_AALC_19_V1_subset.h5']

# test_cymae_embeddings = []
# test_cymae_labels = []

# for filename in test_filenames:
#     full_data, full_labels, marker_list = read_file(path+filename)


#     data = full_data
#     labels = full_labels

#     test_cymae_labels += labels

#     new_marker_indices = []
#     for marker in marker_list:
#         if marker in ckpt_args.union_marker_list:
#             new_marker_indices.append(union_marker_to_index[marker])

#     data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#     with torch.no_grad():
#         with torch.amp.autocast('cuda'):
#             cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)

#     test_cymae_embeddings.append(cell_embeddings.cpu())
# test_cymae_embeddings = torch.cat(test_cymae_embeddings)
# test_cymae_labels = np.array(test_cymae_labels)
# print(test_cymae_embeddings.shape, len(test_cymae_labels))

# for seed in [42,43,44]:
#     rng = np.random.default_rng(seed=seed)
#     selected_idx = undersample_above_target(test_cymae_labels, target_count=1000)
#     # selected_idx = rng.choice(len(test_cymae_labels), size=16387, replace=False)


#     test_ori_data_balanced       = test_ori_data[selected_idx]
#     test_ori_labels_balanced     = test_ori_labels[selected_idx].tolist()
#     test_knn_data_balanced       = test_knn_data[selected_idx]
#     test_knn_labels_balanced     = test_knn_labels[selected_idx].tolist() 
#     # test_cycombine_data_balanced       = test_cycombine_data[selected_idx]
#     # test_cycombine_labels_balanced     = test_cycombine_labels[selected_idx].tolist()  
#     test_cymae_embeddings_balanced       = test_cymae_embeddings[selected_idx]
#     test_cymae_labels_balanced     = test_cymae_labels[selected_idx].tolist()
#     print(test_ori_data_balanced.shape, len(test_ori_labels_balanced))
#     print(test_knn_data_balanced.shape, len(test_knn_labels_balanced))
#     # print(test_cycombine_data_balanced.shape, len(test_cycombine_labels_balanced))
#     print(test_cymae_embeddings_balanced.shape, len(test_cymae_labels_balanced))

#     score = metrics(test_ori_data_balanced.numpy(), test_ori_data_balanced.numpy(),['batch' for _ in range(len(test_ori_labels_balanced))], test_ori_labels_balanced)
#     AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     print(score)
#     print(AvgBIO)

#     score = metrics(test_cymae_embeddings_balanced.numpy(), test_cymae_embeddings_balanced.numpy(),['batch' for _ in range(len(test_cymae_labels_balanced))], test_cymae_labels_balanced)
#     AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     print(score)
#     print(AvgBIO)

#     score = metrics(test_knn_data_balanced.numpy(), test_knn_data_balanced.numpy(),['batch' for _ in range(len(test_knn_labels_balanced))], test_knn_labels_balanced)
#     AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     print(score)
#     print(AvgBIO)

#     # score = metrics(test_cycombine_data_balanced.numpy(), test_cycombine_data_balanced.numpy(),['batch' for _ in range(len(test_cycombine_labels_balanced))], test_cycombine_labels_balanced)
#     # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     # print(score)
#     # print(AvgBIO)










# # simulation complex
# # ckpt = "./ckpts/simulation/dmodel_8_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_celllambda_0.0_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_16_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_celllambda_0.0_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_celllambda_0.0_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_64_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_celllambda_0.0_lr_0.005_checkpoint-5000.pth"

# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_1.0_celllambda_0.0_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_2.0_celllambda_0.0_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_5.0_celllambda_0.0_lr_0.005_checkpoint-5000.pth"

# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_1_celllambda_0.05_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_1_celllambda_0.1_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_1_celllambda_0.2_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_1_celllambda_0.5_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_1_celllambda_1.0_lr_0.005_checkpoint-5000.pth"

# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_2_celllambda_0.05_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_2_celllambda_0.1_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_2_celllambda_0.2_lr_0.005_checkpoint-5000.pth"
# ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_2_celllambda_0.5_lr_0.005_checkpoint-5000.pth"
# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_2_celllambda_1.0_lr_0.005_checkpoint-5000.pth"



# # ckpt = "./ckpts/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_1.0_celllambda_0.0_lr_0.005_checkpoint-5000.pth"

# # ckpt = "./ckpts/simulation/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_1.0_maxstep_2_celllambda_0.2_lr_0.005_checkpoint-5000.pth"
# device = 'cuda'

# checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
# ckpt_args = checkpoint['args']
# ckpt_args.max_step_k = 2
# union_marker_to_index = {
#     marker: idx for idx, marker in enumerate(ckpt_args.union_marker_list)
# }

# model = get_model(ckpt_args)
# model.load_state_dict(checkpoint['model'])
# model.to(device)
# model.eval()


# rng = np.random.default_rng(seed=42)




# # # running on original data
# # path = '/project/kimgroup_immune_health/data/pan_panel/simulation/test/'
# # test_filenames = ['MDIPA_AALC_18_V1_subsetC.h5', 'MDIPA_AALC_19_V1_subsetC.h5']
# # # test_filenames = ['MDIPA_AALC_18_V1_subsetD.h5', 'MDIPA_AALC_19_V1_subsetD.h5']

# # test_ori_data = []
# # test_ori_labels = []
# # for filename in test_filenames:
# #     data, labels, marker_list = read_file(path+filename)

# #     test_ori_data.append(data)
# #     test_ori_labels += labels
# # test_ori_data = torch.cat(test_ori_data)
# # test_ori_labels = np.array(test_ori_labels)
# # print(test_ori_data.shape, len(test_ori_labels))


# # # running on knn imputation
# # path = '/project/kimgroup_immune_health/data/pan_panel/simulation/knn_imputed_test/'
# # test_filenames = ['MDIPA_AALC_18_V1_subsetC_imputed.h5', 'MDIPA_AALC_19_V1_subsetC_imputed.h5']
# # # test_filenames = ['MDIPA_AALC_18_V1_subsetD.h5', 'MDIPA_AALC_19_V1_subsetD.h5']

# # test_knn_data = []
# # test_knn_labels = []
# # for filename in test_filenames:
# #     data, labels, marker_list = read_file(path+filename)

# #     test_knn_data.append(data)
# #     test_knn_labels += labels
# # test_knn_data = torch.cat(test_knn_data)
# # test_knn_labels = np.array(test_knn_labels)
# # print(test_knn_data.shape, len(test_knn_labels))


# # # running on cyCombine imputation
# # path = '/project/kimgroup_immune_health/data/pan_panel/simulation/cyCombine_imputed_test/'
# # test_filenames = ['MDIPA_AALC_18_V1_subsetC_imputed.h5', 'MDIPA_AALC_19_V1_subsetC_imputed.h5']
# # # test_filenames = ['MDIPA_AALC_18_V1_subsetD.h5', 'MDIPA_AALC_19_V1_subsetD.h5']

# # test_cycombine_data = []
# # test_cycombine_labels = []
# # for filename in test_filenames:
# #     data, labels, marker_list = read_file(path+filename)

# #     test_cycombine_data.append(data)
# #     test_cycombine_labels += labels
# # test_cycombine_data = torch.cat(test_cycombine_data)
# # test_cycombine_labels = np.array(test_cycombine_labels)
# # print(test_cycombine_data.shape, len(test_cycombine_labels))


# # running on cyMAE
# path = '/project/kimgroup_immune_health/data/pan_panel/simulation/test/'
# test_filenames = ['MDIPA_AALC_18_V1_subsetC.h5', 'MDIPA_AALC_19_V1_subsetC.h5']
# # test_filenames = ['MDIPA_AALC_18_V1_subsetD.h5', 'MDIPA_AALC_19_V1_subsetD.h5']

# test_cymae_embeddings = []
# test_cymae_labels = []

# for filename in test_filenames:
#     full_data, full_labels, marker_list = read_file(path+filename)


#     data = full_data
#     labels = full_labels

#     test_cymae_labels += labels

#     new_marker_indices = []
#     for marker in marker_list:
#         if marker in ckpt_args.union_marker_list:
#             new_marker_indices.append(union_marker_to_index[marker])

#     data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#     with torch.no_grad():
#         with torch.amp.autocast('cuda'):
#             cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)

#     test_cymae_embeddings.append(cell_embeddings.cpu())
# test_cymae_embeddings = torch.cat(test_cymae_embeddings)
# test_cymae_labels = np.array(test_cymae_labels)
# print(test_cymae_embeddings.shape, len(test_cymae_labels))

# scores = []
# avgbios = []
# for seed in [42,43,44]:
#     rng = np.random.default_rng(seed=seed)
#     selected_idx = undersample_above_target(test_cymae_labels, target_count=1000)
#     # selected_idx = rng.choice(len(test_cymae_labels), size=16387, replace=False)


#     # test_ori_data_balanced       = test_ori_data[selected_idx]
#     # test_ori_labels_balanced     = test_ori_labels[selected_idx].tolist()
#     # test_knn_data_balanced       = test_knn_data[selected_idx]
#     # test_knn_labels_balanced     = test_knn_labels[selected_idx].tolist() 
#     # test_cycombine_data_balanced       = test_cycombine_data[selected_idx]
#     # test_cycombine_labels_balanced     = test_cycombine_labels[selected_idx].tolist()  
#     test_cymae_embeddings_balanced       = test_cymae_embeddings[selected_idx]
#     test_cymae_labels_balanced     = test_cymae_labels[selected_idx].tolist()
#     # print(test_ori_data_balanced.shape, len(test_ori_labels_balanced))
#     # print(test_knn_data_balanced.shape, len(test_knn_labels_balanced))
#     # print(test_cycombine_data_balanced.shape, len(test_cycombine_labels_balanced))
#     print(test_cymae_embeddings_balanced.shape, len(test_cymae_labels_balanced))


#     results = scGraph_metrics(test_cymae_embeddings_balanced.numpy(),test_cymae_embeddings_balanced.numpy(),['batch' for _ in range(len(test_cymae_labels_balanced))], test_cymae_labels_balanced)
#     print(results)

#     # score = metrics(test_ori_data_balanced.numpy(), test_ori_data_balanced.numpy(),['batch' for _ in range(len(test_ori_labels_balanced))], test_ori_labels_balanced)
#     # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     # print(score)
#     # print(AvgBIO)

#     # score = metrics(test_cymae_embeddings_balanced.numpy(), test_cymae_embeddings_balanced.numpy(),['batch' for _ in range(len(test_cymae_labels_balanced))], test_cymae_labels_balanced)
#     # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     # scores.append(score)
#     # avgbios.append(AvgBIO)
#     # print(score)
#     # print(AvgBIO)

#     # score = metrics(test_knn_data_balanced.numpy(), test_knn_data_balanced.numpy(),['batch' for _ in range(len(test_knn_labels_balanced))], test_knn_labels_balanced)
#     # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     # print(score)
#     # print(AvgBIO)

#     # score = metrics(test_cycombine_data_balanced.numpy(), test_cycombine_data_balanced.numpy(),['batch' for _ in range(len(test_cycombine_labels_balanced))], test_cycombine_labels_balanced)
#     # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     # print(score)
#     # print(AvgBIO)

# # scores = pd.concat(scores, axis=1)
# # print(scores.mean(axis=1))
# # print(np.mean(avgbios))


# # reducer = umap.UMAP(n_components=2, random_state=42)
# # umap_result = reducer.fit_transform(test_ori_data_balanced)


# # df = pd.DataFrame({
# #     "UMAP1": umap_result[:, 0],
# #     "UMAP2": umap_result[:, 1],
# #     "Label": test_ori_labels_balanced,
# # })

# # # 2) define a consistent color palette for labels
# # unique_labels = np.sort(df["Label"].unique())
# # palette_label = dict(zip(
# #     unique_labels,
# #     sns.color_palette("hls", len(unique_labels))
# # ))

# # # 4) plot
# # plt.figure(figsize=(12,12))
# # sns.scatterplot(
# #     data=df,
# #     x="UMAP1", y="UMAP2",
# #     hue="Label",
# #     palette=palette_label,
# #     hue_order=list(unique_labels),
# #     s=10,
# #     alpha=0.5,
# #     legend=True
# # )

# # plt.xlabel("UMAP Dimension 1")
# # plt.ylabel("UMAP Dimension 2")
# # plt.tight_layout()
# # plt.savefig("figs/test_ori.png")
# # plt.close()


# # reducer = umap.UMAP(n_components=2, random_state=42)
# # umap_result = reducer.fit_transform(test_cymae_embeddings_balanced)


# # df = pd.DataFrame({
# #     "UMAP1": umap_result[:, 0],
# #     "UMAP2": umap_result[:, 1],
# #     "Label": test_cymae_labels_balanced,
# # })

# # # 2) define a consistent color palette for labels
# # unique_labels = np.sort(df["Label"].unique())
# # palette_label = dict(zip(
# #     unique_labels,
# #     sns.color_palette("hls", len(unique_labels))
# # ))

# # # 4) plot
# # plt.figure(figsize=(12,12))
# # sns.scatterplot(
# #     data=df,
# #     x="UMAP1", y="UMAP2",
# #     hue="Label",
# #     palette=palette_label,
# #     hue_order=list(unique_labels),
# #     s=10,
# #     alpha=0.5,
# #     legend=True
# # )

# # plt.xlabel("UMAP Dimension 1")
# # plt.ylabel("UMAP Dimension 2")
# # plt.tight_layout()
# # plt.savefig("figs/test_cymae.png")
# # plt.close()






# # reducer = umap.UMAP(n_components=2, random_state=42)
# # umap_result = reducer.fit_transform(test_cycombine_data_balanced)


# # df = pd.DataFrame({
# #     "UMAP1": umap_result[:, 0],
# #     "UMAP2": umap_result[:, 1],
# #     "Label": test_cycombine_labels_balanced,
# # })

# # # 2) define a consistent color palette for labels
# # unique_labels = np.sort(df["Label"].unique())
# # palette_label = dict(zip(
# #     unique_labels,
# #     sns.color_palette("hls", len(unique_labels))
# # ))

# # # 4) plot
# # plt.figure(figsize=(12,12))
# # sns.scatterplot(
# #     data=df,
# #     x="UMAP1", y="UMAP2",
# #     hue="Label",
# #     palette=palette_label,
# #     hue_order=list(unique_labels),
# #     s=10,
# #     alpha=0.5,
# #     legend=True
# # )

# # plt.xlabel("UMAP Dimension 1")
# # plt.ylabel("UMAP Dimension 2")
# # plt.tight_layout()
# # plt.savefig("figs/test_cycombine.png")
# # plt.close()



# # reducer = umap.UMAP(n_components=2, random_state=42)
# # umap_result = reducer.fit_transform(test_knn_data_balanced)


# # df = pd.DataFrame({
# #     "UMAP1": umap_result[:, 0],
# #     "UMAP2": umap_result[:, 1],
# #     "Label": test_knn_labels_balanced,
# # })

# # # 2) define a consistent color palette for labels
# # unique_labels = np.sort(df["Label"].unique())
# # palette_label = dict(zip(
# #     unique_labels,
# #     sns.color_palette("hls", len(unique_labels))
# # ))

# # # 4) plot
# # plt.figure(figsize=(12,12))
# # sns.scatterplot(
# #     data=df,
# #     x="UMAP1", y="UMAP2",
# #     hue="Label",
# #     palette=palette_label,
# #     hue_order=list(unique_labels),
# #     s=10,
# #     alpha=0.5,
# #     legend=True
# # )

# # plt.xlabel("UMAP Dimension 1")
# # plt.ylabel("UMAP Dimension 2")
# # plt.tight_layout()
# # plt.savefig("figs/test_knn.png")
# # plt.close()












# # real data

# # ckpt = "./ckpts/realdata/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_0.0_maxstep_0_celllambda_0.0_lr_0.005_checkpoint-5000.pth"
# ckpt = "./ckpts/realdata/dmodel_32_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.5_maxstep_1_celllambda_0.05_lr_0.005_checkpoint-10000.pth"
# device = 'cuda'

# checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
# ckpt_args = checkpoint['args']
# union_marker_to_index = {
#     marker: idx for idx, marker in enumerate(ckpt_args.union_marker_list)
# }

# model = get_model(ckpt_args)
# model.load_state_dict(checkpoint['model'])
# model.to(device)
# model.eval()


# rng = np.random.default_rng(seed=42)




# # running on original data
# path = '/project/kimgroup_immune_health/data/pan_panel/realdata/test/'
# # test_filenames = ['MDIPA_NeuExpV2_DORA_062_BASE.h5', 'MDIPA_NeuExpV2_DORA_063_BASE.h5', 'MDIPA_NeuExpV2_DORA_064_BASE.h5',
# #                   'MDIPA_NeuExpV2_DORA_065_BASE.h5', 'MDIPA_NeuExpV2_DORA_066_BASE.h5']
# # test_filenames = ['200616_MDIPAa4_PICR7363T1_debc.h5', '200616_MDIPAa4_PICR7378T1_debc.h5', '200616_MDIPAa4_PICR7378T4_debc.h5', '200616_MDIPAa4_PICR7378T8_debc.h5',
# #                   '200616_MDIPAa4_PICR7386T1_debc.h5', '200616_MDIPAa4_PICR7386T4_debc.h5', '200616_MDIPAa4_PICR8025_A_debc.h5', '200616_MDIPAa4_PICR8028_A_debc.h5',
# #                   '200616_MDIPAa4_PICR8029_A_debc.h5', '200616_MDIPAa4_PICR8035_A_debc.h5', '200616_MDIPAa4_PICR8104_A_debc.h5', '200616_MDIPAa4_PICR8107_A_debc.h5',
# #                   '200616_MDIPAa4_PICR8108_A_debc.h5', '200616_MDIPAa4_PICR8109_A_debc.h5']
# test_filenames = [
#     'MDIPA_MESSI_994933_D0.h5', 'MDIPA_MESSI_994591_D0.h5', 'MDIPA_MESSI_994570_D0.h5', 'MDIPA_MESSI_994586_D0.h5', 'MDIPA_MESSI_994955_D0.h5',
#     'MDIPA_MESSI_994749_D0.h5', 'MDIPA_MESSI_994938_D0.h5', 'MDIPA_MESSI_994945_D0.h5', 'MDIPA_MESSI_994942_D0.h5', 'MDIPA_MESSI_994588_D0.h5',
#     'MDIPA_MESSI_HD_1.h5', 'MDIPA_MESSI_HD_2.h5', 'MDIPA_MESSI_HD_3.h5', 'MDIPA_MESSI_HD_4.h5',
# ]

# test_ori_data = []
# test_ori_labels = []
# for filename in test_filenames:
#     data, labels, marker_list = read_file(path+filename)

#     test_ori_data.append(data)
#     test_ori_labels += labels
# test_ori_data = torch.cat(test_ori_data)
# test_ori_labels = np.array(test_ori_labels)
# print(test_ori_data.shape, len(test_ori_labels))


# # running on knn imputation
# path = '/project/kimgroup_immune_health/data/pan_panel/realdata/knn_imputed_test/'
# # test_filenames = ['MDIPA_NeuExpV2_DORA_062_BASE_imputed.h5', 'MDIPA_NeuExpV2_DORA_063_BASE_imputed.h5', 'MDIPA_NeuExpV2_DORA_064_BASE_imputed.h5',
# #                   'MDIPA_NeuExpV2_DORA_065_BASE_imputed.h5', 'MDIPA_NeuExpV2_DORA_066_BASE_imputed.h5']
# # test_filenames = ['200616_MDIPAa4_PICR7363T1_debc_imputed.h5', '200616_MDIPAa4_PICR7378T1_debc_imputed.h5', '200616_MDIPAa4_PICR7378T4_debc_imputed.h5', '200616_MDIPAa4_PICR7378T8_debc_imputed.h5',
# #                   '200616_MDIPAa4_PICR7386T1_debc_imputed.h5', '200616_MDIPAa4_PICR7386T4_debc_imputed.h5', '200616_MDIPAa4_PICR8025_A_debc_imputed.h5', '200616_MDIPAa4_PICR8028_A_debc_imputed.h5',
# #                   '200616_MDIPAa4_PICR8029_A_debc_imputed.h5', '200616_MDIPAa4_PICR8035_A_debc_imputed.h5', '200616_MDIPAa4_PICR8104_A_debc_imputed.h5', '200616_MDIPAa4_PICR8107_A_debc_imputed.h5',
# #                   '200616_MDIPAa4_PICR8108_A_debc_imputed.h5', '200616_MDIPAa4_PICR8109_A_debc_imputed.h5']
# test_filenames = [
#     'MDIPA_MESSI_994933_D0_imputed.h5', 'MDIPA_MESSI_994591_D0_imputed.h5', 'MDIPA_MESSI_994570_D0_imputed.h5', 'MDIPA_MESSI_994586_D0_imputed.h5', 'MDIPA_MESSI_994955_D0_imputed.h5',
#     'MDIPA_MESSI_994749_D0_imputed.h5', 'MDIPA_MESSI_994938_D0_imputed.h5', 'MDIPA_MESSI_994945_D0_imputed.h5', 'MDIPA_MESSI_994942_D0_imputed.h5', 'MDIPA_MESSI_994588_D0_imputed.h5',
#     'MDIPA_MESSI_HD_1_imputed.h5', 'MDIPA_MESSI_HD_2_imputed.h5', 'MDIPA_MESSI_HD_3_imputed.h5', 'MDIPA_MESSI_HD_4_imputed.h5',
# ]


# test_knn_data = []
# test_knn_labels = []
# for filename in test_filenames:
#     data, labels, marker_list = read_file(path+filename)

#     test_knn_data.append(data)
#     test_knn_labels += labels
# test_knn_data = torch.cat(test_knn_data)
# test_knn_labels = np.array(test_knn_labels)
# print(test_knn_data.shape, len(test_knn_labels))


# # # running on cyCombine imputation
# # path = '/project/kimgroup_immune_health/data/pan_panel/realdata/cyCombine_imputed_test/'
# # test_filenames = ['MDIPA_NeuExpV2_DORA_062_BASE.h5', 'MDIPA_NeuExpV2_DORA_063_BASE.h5', 'MDIPA_NeuExpV2_DORA_064_BASE.h5',
# #                   'MDIPA_NeuExpV2_DORA_065_BASE.h5', 'MDIPA_NeuExpV2_DORA_066_BASE.h5']
# # # test_filenames = ['200616_MDIPAa4_PICR7363T1_debc.h5', '200616_MDIPAa4_PICR7378T1_debc.h5', '200616_MDIPAa4_PICR7378T4_debc.h5', '200616_MDIPAa4_PICR7378T8_debc.h5',
# # #                   '200616_MDIPAa4_PICR7386T1_debc.h5', '200616_MDIPAa4_PICR7386T4_debc.h5', '200616_MDIPAa4_PICR8025_A_debc.h5', '200616_MDIPAa4_PICR8028_A_debc.h5',
# # #                   '200616_MDIPAa4_PICR8029_A_debc.h5', '200616_MDIPAa4_PICR8035_A_debc.h5', '200616_MDIPAa4_PICR8104_A_debc.h5', '200616_MDIPAa4_PICR8107_A_debc.h5',
# # #                   '200616_MDIPAa4_PICR8108_A_debc.h5', '200616_MDIPAa4_PICR8109_A_debc.h5']
# # test_filenames = [
# #     'MDIPA_MESSI_994933_D0.h5', 'MDIPA_MESSI_994591_D0.h5', 'MDIPA_MESSI_994570_D0.h5', 'MDIPA_MESSI_994586_D0.h5', 'MDIPA_MESSI_994955_D0.h5',
# #     'MDIPA_MESSI_994749_D0.h5', 'MDIPA_MESSI_994938_D0.h5', 'MDIPA_MESSI_994945_D0.h5', 'MDIPA_MESSI_994942_D0.h5', 'MDIPA_MESSI_994588_D0.h5',
# #     'MDIPA_MESSI_HD_1.h5', 'MDIPA_MESSI_HD_2.h5', 'MDIPA_MESSI_HD_3.h5', 'MDIPA_MESSI_HD_4.h5',
# # ]

# # test_filenames = ['MDIPA_AALC_18_V1_subsetC_imputed.h5', 'MDIPA_AALC_19_V1_subsetC_imputed.h5']
# # # test_filenames = ['MDIPA_AALC_18_V1_subsetD.h5', 'MDIPA_AALC_19_V1_subsetD.h5']

# # test_cycombine_data = []
# # test_cycombine_labels = []
# # for filename in test_filenames:
# #     data, labels, marker_list = read_file(path+filename)

# #     test_cycombine_data.append(data)
# #     test_cycombine_labels += labels
# # test_cycombine_data = torch.cat(test_cycombine_data)
# # test_cycombine_labels = np.array(test_cycombine_labels)
# # print(test_cycombine_data.shape, len(test_cycombine_labels))


# # running on cyMAE
# path = '/project/kimgroup_immune_health/data/pan_panel/realdata/test/'
# # test_filenames = ['MDIPA_NeuExpV2_DORA_062_BASE.h5', 'MDIPA_NeuExpV2_DORA_063_BASE.h5', 'MDIPA_NeuExpV2_DORA_064_BASE.h5',
# #                   'MDIPA_NeuExpV2_DORA_065_BASE.h5', 'MDIPA_NeuExpV2_DORA_066_BASE.h5']
# # test_filenames = ['200616_MDIPAa4_PICR7363T1_debc.h5', '200616_MDIPAa4_PICR7378T1_debc.h5', '200616_MDIPAa4_PICR7378T4_debc.h5', '200616_MDIPAa4_PICR7378T8_debc.h5',
# #                   '200616_MDIPAa4_PICR7386T1_debc.h5', '200616_MDIPAa4_PICR7386T4_debc.h5', '200616_MDIPAa4_PICR8025_A_debc.h5', '200616_MDIPAa4_PICR8028_A_debc.h5',
# #                   '200616_MDIPAa4_PICR8029_A_debc.h5', '200616_MDIPAa4_PICR8035_A_debc.h5', '200616_MDIPAa4_PICR8104_A_debc.h5', '200616_MDIPAa4_PICR8107_A_debc.h5',
# #                   '200616_MDIPAa4_PICR8108_A_debc.h5', '200616_MDIPAa4_PICR8109_A_debc.h5']
# test_filenames = [
#     'MDIPA_MESSI_994933_D0.h5', 'MDIPA_MESSI_994591_D0.h5', 'MDIPA_MESSI_994570_D0.h5', 'MDIPA_MESSI_994586_D0.h5', 'MDIPA_MESSI_994955_D0.h5',
#     'MDIPA_MESSI_994749_D0.h5', 'MDIPA_MESSI_994938_D0.h5', 'MDIPA_MESSI_994945_D0.h5', 'MDIPA_MESSI_994942_D0.h5', 'MDIPA_MESSI_994588_D0.h5',
#     'MDIPA_MESSI_HD_1.h5', 'MDIPA_MESSI_HD_2.h5', 'MDIPA_MESSI_HD_3.h5', 'MDIPA_MESSI_HD_4.h5',
# ]


# test_cymae_embeddings = []
# test_cymae_labels = []

# for filename in test_filenames:
#     full_data, full_labels, marker_list = read_file(path+filename)


#     data = full_data
#     labels = full_labels

#     test_cymae_labels += labels

#     new_marker_indices = []
#     for marker in marker_list:
#         if marker in ckpt_args.union_marker_list:
#             new_marker_indices.append(union_marker_to_index[marker])

#     data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#     with torch.no_grad():
#         with torch.amp.autocast('cuda'):
#             cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)

#     test_cymae_embeddings.append(cell_embeddings.cpu())
# test_cymae_embeddings = torch.cat(test_cymae_embeddings)
# test_cymae_labels = np.array(test_cymae_labels)
# print(test_cymae_embeddings.shape, len(test_cymae_labels))

# for seed in [42,43,44]:
#     rng = np.random.default_rng(seed=seed)
#     selected_idx = undersample_above_target(test_cymae_labels, target_count=1000)
#     # selected_idx = rng.choice(len(test_cymae_labels), size=16387, replace=False)


#     test_ori_data_balanced       = test_ori_data[selected_idx]
#     test_ori_labels_balanced     = test_ori_labels[selected_idx].tolist()
#     test_knn_data_balanced       = test_knn_data[selected_idx]
#     test_knn_labels_balanced     = test_knn_labels[selected_idx].tolist() 
#     # test_cycombine_data_balanced       = test_cycombine_data[selected_idx]
#     # test_cycombine_labels_balanced     = test_cycombine_labels[selected_idx].tolist()  
#     test_cymae_embeddings_balanced       = test_cymae_embeddings[selected_idx]
#     test_cymae_labels_balanced     = test_cymae_labels[selected_idx].tolist()
#     print(test_ori_data_balanced.shape, len(test_ori_labels_balanced))
#     print(test_knn_data_balanced.shape, len(test_knn_labels_balanced))
#     # print(test_cycombine_data_balanced.shape, len(test_cycombine_labels_balanced))
#     print(test_cymae_embeddings_balanced.shape, len(test_cymae_labels_balanced))

#     # score = metrics(test_ori_data_balanced.numpy(), test_ori_data_balanced.numpy(),['batch' for _ in range(len(test_ori_labels_balanced))], test_ori_labels_balanced)
#     # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     # print(score)
#     # print(AvgBIO)

#     # score = metrics(test_cymae_embeddings_balanced.numpy(), test_cymae_embeddings_balanced.numpy(),['batch' for _ in range(len(test_cymae_labels_balanced))], test_cymae_labels_balanced)
#     # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     # print(score)
#     # print(AvgBIO)

#     # score = metrics(test_knn_data_balanced.numpy(), test_knn_data_balanced.numpy(),['batch' for _ in range(len(test_knn_labels_balanced))], test_knn_labels_balanced)
#     # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     # print(score)
#     # print(AvgBIO)

#     # score = metrics(test_cycombine_data_balanced.numpy(), test_cycombine_data_balanced.numpy(),['batch' for _ in range(len(test_cycombine_labels_balanced))], test_cycombine_labels_balanced)
#     # AvgBIO = score.loc[['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label']].mean(axis=0)
#     # print(score)
#     # print(AvgBIO)



# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(test_ori_data_balanced)


# df = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": test_ori_labels_balanced,
# })

# # 2) define a consistent color palette for labels
# unique_labels = np.sort(df["Label"].unique())
# palette_label = dict(zip(
#     unique_labels,
#     sns.color_palette("hls", len(unique_labels))
# ))

# # 4) plot
# plt.figure(figsize=(12,12))
# sns.scatterplot(
#     data=df,
#     x="UMAP1", y="UMAP2",
#     hue="Label",
#     palette=palette_label,
#     hue_order=list(unique_labels),
#     s=10,
#     alpha=0.5,
#     legend=True
# )

# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/test_ori.png")
# plt.close()


# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(test_cymae_embeddings_balanced)


# df = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": test_cymae_labels_balanced,
# })

# # 2) define a consistent color palette for labels
# unique_labels = np.sort(df["Label"].unique())
# palette_label = dict(zip(
#     unique_labels,
#     sns.color_palette("hls", len(unique_labels))
# ))

# # 4) plot
# plt.figure(figsize=(12,12))
# sns.scatterplot(
#     data=df,
#     x="UMAP1", y="UMAP2",
#     hue="Label",
#     palette=palette_label,
#     hue_order=list(unique_labels),
#     s=10,
#     alpha=0.5,
#     legend=True
# )

# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/test_cymae.png")
# plt.close()






# # reducer = umap.UMAP(n_components=2, random_state=42)
# # umap_result = reducer.fit_transform(test_cycombine_data_balanced)


# # df = pd.DataFrame({
# #     "UMAP1": umap_result[:, 0],
# #     "UMAP2": umap_result[:, 1],
# #     "Label": test_cycombine_labels_balanced,
# # })

# # # 2) define a consistent color palette for labels
# # unique_labels = np.sort(df["Label"].unique())
# # palette_label = dict(zip(
# #     unique_labels,
# #     sns.color_palette("hls", len(unique_labels))
# # ))

# # # 4) plot
# # plt.figure(figsize=(12,12))
# # sns.scatterplot(
# #     data=df,
# #     x="UMAP1", y="UMAP2",
# #     hue="Label",
# #     palette=palette_label,
# #     hue_order=list(unique_labels),
# #     s=10,
# #     alpha=0.5,
# #     legend=True
# # )

# # plt.xlabel("UMAP Dimension 1")
# # plt.ylabel("UMAP Dimension 2")
# # plt.tight_layout()
# # plt.savefig("figs/test_cycombine.png")
# # plt.close()



# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(test_knn_data_balanced)


# df = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": test_knn_labels_balanced,
# })

# # 2) define a consistent color palette for labels
# unique_labels = np.sort(df["Label"].unique())
# palette_label = dict(zip(
#     unique_labels,
#     sns.color_palette("hls", len(unique_labels))
# ))

# # 4) plot
# plt.figure(figsize=(12,12))
# sns.scatterplot(
#     data=df,
#     x="UMAP1", y="UMAP2",
#     hue="Label",
#     palette=palette_label,
#     hue_order=list(unique_labels),
#     s=10,
#     alpha=0.5,
#     legend=True
# )

# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/test_knn.png")
# plt.close()
