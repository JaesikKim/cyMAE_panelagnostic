import os
import glob
import torch
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from datasets import read_file
from run_mae_pretraining import get_model

import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

MDIPA = ['CD45', 'CD196', 'CD123', 'CD19', 'CD4', 'CD8a', 'CD11c', 'CD16', 'CD45RO', 'CD45RA', 'CD161', 'CD194', 'CD25', 'CD27', 'CD57', 'CD183', 'CD185', 'CD28', 'CD38', 'CD56', 'TCRgd', 'CD294', 'CD197', 'CD14', 'CD3', 'CD20', 'CD66b', 'HLA-DR', 'IgD', 'CD127']

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

def plot_umap(data, labels, filepath):
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = reducer.fit_transform(data)

    unique_labels = np.sort(np.unique(labels))  # 알파벳(또는 숫자) 순서 정렬
    palette1 = sns.color_palette("hls", len(unique_labels))
    palette_all = {label: palette1[i] for i, label in enumerate(unique_labels)}

    df_all = pd.DataFrame({
        "UMAP1": umap_result[:, 0],
        "UMAP2": umap_result[:, 1],
        "Label": labels
    })

    plt.figure(figsize=(10, 8))
    # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
    sns.scatterplot(
        data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
        palette=palette_all, s=10, alpha=0.7, legend="auto"
    )
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()



rng = np.random.default_rng(seed=42)


# simulation2 plot1: [knn] panel_A_acute vs panel_B_acute vs panel_C_vaccine vs panel_C_ISPY
# simulation2 plot2: [cymae] panel_A_acute vs panel_B_acute vs panel_C_vaccine vs panel_C_ISPY
# simulation2 plot3: panel_D_vaccine - original vs cymae vs knn vs cycombine
# simulation2 plot4 panel_D_ISPY - original vs cymae vs knn vs cycombine

ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.5_lr_0.005_checkpoint-6000.pth"

device = 'cuda'

checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
ckpt_args = checkpoint['args']
ckpt_args.is_cumul_masking = True

model = get_model(ckpt_args)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

union_marker_to_index = {
    marker: idx for idx, marker in enumerate(ckpt_args.union_marker_list)
}


path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/test_panel_C/'

test_panel_C_data = []
test_panel_C_cell_embeddings = []
test_panel_C_labels = []

for filename in os.listdir(path):
    # if filename not in ['43.T3_Normalized.h5', '23.T1_Normalized.h5', '15.T2_Normalized.h5', '45.T1_Normalized.h5']:
    #     continue
    full_data, full_labels, marker_list = read_file(path+filename)
    full_labels = simple_label_mapper(full_labels)

    selected_idx = undersample_above_target(full_labels, target_count=100)
    # selected_idx = rng.choice(len(full_labels), size=2000, replace=False)

    data = full_data[selected_idx]
    labels = np.array(full_labels)[selected_idx].tolist()
    
    test_panel_C_data.append(data.numpy())
    test_panel_C_labels += labels

    new_marker_indices = []
    for marker in marker_list:
        if marker in ckpt_args.union_marker_list:
            new_marker_indices.append(union_marker_to_index[marker])

    data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)
    test_panel_C_cell_embeddings.append(cell_embeddings.cpu().numpy())

test_panel_C_data = pd.DataFrame(np.concatenate(test_panel_C_data), columns=marker_list)
test_panel_C_cell_embeddings = np.concatenate(test_panel_C_cell_embeddings)

path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/test_panel_D/'

test_panel_D_data = []
test_panel_D_cell_embeddings = []
test_panel_D_labels = []

for filename in os.listdir(path):
    # if filename not in ['43.T3_Normalized.h5', '23.T1_Normalized.h5', '15.T2_Normalized.h5', '45.T1_Normalized.h5']:
    #     continue
    full_data, full_labels, marker_list = read_file(path+filename)
    full_labels = simple_label_mapper(full_labels)

    selected_idx = undersample_above_target(full_labels, target_count=100)
    # selected_idx = rng.choice(len(full_labels), size=2000, replace=False)

    data = full_data[selected_idx]
    labels = np.array(full_labels)[selected_idx].tolist()
    
    test_panel_D_data.append(data.numpy())
    test_panel_D_labels += labels

    new_marker_indices = []
    for marker in marker_list:
        if marker in ckpt_args.union_marker_list:
            new_marker_indices.append(union_marker_to_index[marker])

    data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)
    test_panel_D_cell_embeddings.append(cell_embeddings.cpu().numpy())

test_panel_D_data = pd.DataFrame(np.concatenate(test_panel_D_data), columns=marker_list)
test_panel_D_cell_embeddings = np.concatenate(test_panel_D_cell_embeddings)

plot_umap(test_panel_C_data, test_panel_C_labels, "figs/simulation2/umap_test_panel_C_data_celltype.png")
plot_umap(test_panel_C_cell_embeddings, test_panel_C_labels, "figs/simulation2/umap_test_panel_C_cymae_celltype.png")
plot_umap(test_panel_D_data, test_panel_D_labels, "figs/simulation2/umap_test_panel_D_data_celltype.png")
plot_umap(test_panel_D_cell_embeddings, test_panel_D_labels, "figs/simulation2/umap_test_panel_D_cymae_celltype.png")


path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/ext_panel_C/'

ext_panel_C_data = []
ext_panel_C_cell_embeddings = []
ext_panel_C_labels = []

for filename in os.listdir(path):
    # if filename not in ['43.T3_Normalized.h5', '23.T1_Normalized.h5', '15.T2_Normalized.h5', '45.T1_Normalized.h5']:
    #     continue
    full_data, full_labels, marker_list = read_file(path+filename)
    full_labels = simple_label_mapper(full_labels)

    selected_idx = undersample_above_target(full_labels, target_count=100)
    # selected_idx = rng.choice(len(full_labels), size=2000, replace=False)
    
    data = full_data[selected_idx]
    labels = np.array(full_labels)[selected_idx].tolist()
    
    ext_panel_C_data.append(data.numpy())
    ext_panel_C_labels += labels

    new_marker_indices = []
    for marker in marker_list:
        if marker in ckpt_args.union_marker_list:
            new_marker_indices.append(union_marker_to_index[marker])

    data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)
    ext_panel_C_cell_embeddings.append(cell_embeddings.cpu().numpy())

ext_panel_C_data = pd.DataFrame(np.concatenate(ext_panel_C_data), columns=marker_list)
ext_panel_C_cell_embeddings = np.concatenate(ext_panel_C_cell_embeddings)

path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/ext_panel_D/'

ext_panel_D_data = []
ext_panel_D_cell_embeddings = []
ext_panel_D_labels = []

for filename in os.listdir(path):
    # if filename not in ['43.T3_Normalized.h5', '23.T1_Normalized.h5', '15.T2_Normalized.h5', '45.T1_Normalized.h5']:
    #     continue
    full_data, full_labels, marker_list = read_file(path+filename)
    full_labels = simple_label_mapper(full_labels)

    selected_idx = undersample_above_target(full_labels, target_count=100)
    # selected_idx = rng.choice(len(full_labels), size=2000, replace=False)

    data = full_data[selected_idx]
    labels = np.array(full_labels)[selected_idx].tolist()
    
    ext_panel_D_data.append(data.numpy())
    ext_panel_D_labels += labels

    new_marker_indices = []
    for marker in marker_list:
        if marker in ckpt_args.union_marker_list:
            new_marker_indices.append(union_marker_to_index[marker])

    data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)
    ext_panel_D_cell_embeddings.append(cell_embeddings.cpu().numpy())

ext_panel_D_data = pd.DataFrame(np.concatenate(ext_panel_D_data), columns=marker_list)
ext_panel_D_cell_embeddings = np.concatenate(ext_panel_D_cell_embeddings)

plot_umap(ext_panel_C_data, ext_panel_C_labels, "figs/simulation2/umap_ext_panel_C_data_celltype.png")
plot_umap(ext_panel_C_cell_embeddings, ext_panel_C_labels, "figs/simulation2/umap_ext_panel_C_cymae_celltype.png")
plot_umap(ext_panel_D_data, ext_panel_D_labels, "figs/simulation2/umap_ext_panel_D_data_celltype.png")
plot_umap(ext_panel_D_cell_embeddings, ext_panel_D_labels, "figs/simulation2/umap_ext_panel_D_cymae_celltype.png")


path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/ext2_panel_C/'

ext_panel_C_data = []
ext_panel_C_cell_embeddings = []
ext_panel_C_labels = []

for filename in os.listdir(path):
    # if filename not in ['43.T3_Normalized.h5', '23.T1_Normalized.h5', '15.T2_Normalized.h5', '45.T1_Normalized.h5']:
    #     continue
    full_data, full_labels, marker_list = read_file(path+filename)
    full_labels = simple_label_mapper(full_labels)

    selected_idx = undersample_above_target(full_labels, target_count=100)
    # selected_idx = rng.choice(len(full_labels), size=2000, replace=False)
    
    data = full_data[selected_idx]
    labels = np.array(full_labels)[selected_idx].tolist()
    
    ext_panel_C_data.append(data.numpy())
    ext_panel_C_labels += labels

    new_marker_indices = []
    for marker in marker_list:
        if marker in ckpt_args.union_marker_list:
            new_marker_indices.append(union_marker_to_index[marker])

    data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)
    ext_panel_C_cell_embeddings.append(cell_embeddings.cpu().numpy())

ext_panel_C_data = pd.DataFrame(np.concatenate(ext_panel_C_data), columns=marker_list)
ext_panel_C_cell_embeddings = np.concatenate(ext_panel_C_cell_embeddings)

path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/ext2_panel_D/'

ext_panel_D_data = []
ext_panel_D_cell_embeddings = []
ext_panel_D_labels = []

for filename in os.listdir(path):
    # if filename not in ['43.T3_Normalized.h5', '23.T1_Normalized.h5', '15.T2_Normalized.h5', '45.T1_Normalized.h5']:
    #     continue
    full_data, full_labels, marker_list = read_file(path+filename)
    full_labels = simple_label_mapper(full_labels)

    selected_idx = undersample_above_target(full_labels, target_count=100)
    # selected_idx = rng.choice(len(full_labels), size=2000, replace=False)

    data = full_data[selected_idx]
    labels = np.array(full_labels)[selected_idx].tolist()
    
    ext_panel_D_data.append(data.numpy())
    ext_panel_D_labels += labels

    new_marker_indices = []
    for marker in marker_list:
        if marker in ckpt_args.union_marker_list:
            new_marker_indices.append(union_marker_to_index[marker])

    data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)
    ext_panel_D_cell_embeddings.append(cell_embeddings.cpu().numpy())

ext_panel_D_data = pd.DataFrame(np.concatenate(ext_panel_D_data), columns=marker_list)
ext_panel_D_cell_embeddings = np.concatenate(ext_panel_D_cell_embeddings)

plot_umap(ext_panel_C_data, ext_panel_C_labels, "figs/simulation2/umap_ext2_panel_C_data_celltype.png")
plot_umap(ext_panel_C_cell_embeddings, ext_panel_C_labels, "figs/simulation2/umap_ext2_panel_C_cymae_celltype.png")
plot_umap(ext_panel_D_data, ext_panel_D_labels, "figs/simulation2/umap_ext2_panel_D_data_celltype.png")
plot_umap(ext_panel_D_cell_embeddings, ext_panel_D_labels, "figs/simulation2/umap_ext2_panel_D_cymae_celltype.png")



# combined_panel_C_data = pd.concat((test_panel_C_data, ext_panel_C_data))
# combined_panel_C_cell_embeddings = np.concatenate((test_panel_C_cell_embeddings, ext_panel_C_cell_embeddings))
# combined_panel_C_labels = test_panel_C_labels + ext_panel_C_labels
# combined_panel_D_data = pd.concat((test_panel_D_data, ext_panel_D_data))
# combined_panel_D_cell_embeddings = np.concatenate((test_panel_D_cell_embeddings, ext_panel_D_cell_embeddings))
# combined_panel_D_labels = test_panel_D_labels + ext_panel_D_labels

# plot_umap(combined_panel_C_data, combined_panel_C_labels, "figs/simulation2/umap_combined_panel_C_data_celltype.png")
# plot_umap(combined_panel_C_cell_embeddings, combined_panel_C_labels, "figs/simulation2/umap_combined_panel_C_cymae_celltype.png")
# plot_umap(combined_panel_D_data, combined_panel_D_labels, "figs/simulation2/umap_combined_panel_D_data_celltype.png")
# plot_umap(combined_panel_D_cell_embeddings, combined_panel_D_labels, "figs/simulation2/umap_combined_panel_D_cymae_celltype.png")







# path = '/project/kimgroup_immune_health/data/pan_panel/simulation3/MDIPA_ISPY/'

# MDIPA_ISPY_data = []
# MDIPA_ISPY_cell_embeddings = []
# MDIPA_ISPY_labels = []

# for filename in os.listdir(path):
#     if filename not in ['10_Normalized.h5', '11_Normalized.h5', '12_Normalized.h5', '13_Normalized.h5', '14_Normalized.h5']:
#         continue

#     full_data, full_labels, marker_list = read_file(path+filename)

#     selected_idx = undersample_above_target(full_labels, target_count=100)
    
#     data = full_data[selected_idx]
#     labels = np.array(full_labels)[selected_idx].tolist()

    
#     MDIPA_ISPY_data.append(data.numpy())
#     MDIPA_ISPY_labels += labels

#     new_marker_indices = []
#     for marker in marker_list:
#         if marker in ckpt_args.union_marker_list:
#             new_marker_indices.append(union_marker_to_index[marker])

#     data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#     with torch.no_grad():
#         with torch.amp.autocast('cuda'):
#             cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)
#     MDIPA_ISPY_cell_embeddings.append(cell_embeddings.cpu().numpy())

# MDIPA_ISPY_data = pd.DataFrame(np.concatenate(MDIPA_ISPY_data), columns=marker_list)
# MDIPA_ISPY_cell_embeddings = np.concatenate(MDIPA_ISPY_cell_embeddings)


# MDIPA_combined_cell_embeddings = np.concatenate((dev_cell_embeddings, MDIPA_vaccine_cell_embeddings, MDIPA_ISPY_cell_embeddings))
# MDIPA_combined_knn_imputed = pd.concat((dev_knn_imputed, MDIPA_vaccine_data, MDIPA_ISPY_data))
# MDIPA_combined_labels = dev_labels + MDIPA_vaccine_labels + MDIPA_ISPY_labels
# MDIPA_combined_datasets = ['dev']*len(dev_labels) + ['external1']*len(MDIPA_vaccine_labels) + ['external2']*len(MDIPA_ISPY_labels)


# plot_umap(MDIPA_combined_knn_imputed, MDIPA_combined_labels, "figs/simulation3/umap_MDIPA_knn_celltype.png")
# plot_umap(MDIPA_combined_cell_embeddings, MDIPA_combined_labels, "figs/simulation3/umap_MDIPA_cymae_celltype.png")
# plot_umap(MDIPA_combined_knn_imputed, MDIPA_combined_datasets, "figs/simulation3/umap_MDIPA_knn_dataset.png")
# plot_umap(MDIPA_combined_cell_embeddings, MDIPA_combined_datasets, "figs/simulation3/umap_MDIPA_cymae_dataset.png")


# path = '/project/kimgroup_immune_health/data/pan_panel/simulation3/panel_C_vaccine/'
# knn_path = '/project/kimgroup_immune_health/data/pan_panel/simulation3/knn_imputed_panel_C_vaccine/'

# panel_C_vaccine_cell_embeddings = []
# panel_C_vaccine_knn_imputed = []
# panel_C_vaccine_labels = []

# for filename in os.listdir(path):
#     if filename not in ['43.T3_Normalized.h5', '23.T1_Normalized.h5', '15.T2_Normalized.h5', '45.T1_Normalized.h5']:
#         continue

#     full_data, full_labels, marker_list = read_file(path+filename)
#     full_knn_data, _, knn_marker_list = read_file(knn_path+filename[:-3]+"_imputed.h5")

#     selected_idx = undersample_above_target(full_labels, target_count=100)
    
#     data = full_data[selected_idx]
#     knn_data = full_knn_data[selected_idx]
#     labels = np.array(full_labels)[selected_idx].tolist()

#     panel_C_vaccine_knn_imputed.append(knn_data.numpy())
#     panel_C_vaccine_labels += labels

#     new_marker_indices = []
#     for marker in marker_list:
#         if marker in ckpt_args.union_marker_list:
#             new_marker_indices.append(union_marker_to_index[marker])

#     data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#     with torch.no_grad():
#         with torch.amp.autocast('cuda'):
#             cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)
#     panel_C_vaccine_cell_embeddings.append(cell_embeddings.cpu().numpy())

# panel_C_vaccine_cell_embeddings = np.concatenate(panel_C_vaccine_cell_embeddings)
# panel_C_vaccine_knn_imputed = pd.DataFrame(np.concatenate(panel_C_vaccine_knn_imputed), columns=knn_marker_list)
# print(panel_C_vaccine_knn_imputed.shape)


# path = '/project/kimgroup_immune_health/data/pan_panel/simulation3/panel_C_ISPY/'
# knn_path = '/project/kimgroup_immune_health/data/pan_panel/simulation3/knn_imputed_panel_C_ISPY/'

# panel_C_ISPY_cell_embeddings = []
# panel_C_ISPY_knn_imputed = []
# panel_C_ISPY_labels = []

# for filename in os.listdir(path):
#     if filename not in ['10_Normalized.h5', '11_Normalized.h5', '12_Normalized.h5', '13_Normalized.h5', '14_Normalized.h5']:
#         continue
#     full_data, full_labels, marker_list = read_file(path+filename)
#     full_knn_data, _, knn_marker_list = read_file(knn_path+filename[:-3]+"_imputed.h5")

#     selected_idx = undersample_above_target(full_labels, target_count=100)
    
#     data = full_data[selected_idx]
#     knn_data = full_knn_data[selected_idx]
#     labels = np.array(full_labels)[selected_idx].tolist()
    
#     panel_C_ISPY_knn_imputed.append(knn_data.numpy())
#     panel_C_ISPY_labels += labels

#     new_marker_indices = []
#     for marker in marker_list:
#         if marker in ckpt_args.union_marker_list:
#             new_marker_indices.append(union_marker_to_index[marker])

#     data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#     with torch.no_grad():
#         with torch.amp.autocast('cuda'):
#             cell_embeddings, pooled_embeddings = model.forward_inference(data, new_marker_indices)
#     panel_C_ISPY_cell_embeddings.append(cell_embeddings.cpu().numpy())

# panel_C_ISPY_cell_embeddings = np.concatenate(panel_C_ISPY_cell_embeddings)
# panel_C_ISPY_knn_imputed = pd.DataFrame(np.concatenate(panel_C_ISPY_knn_imputed), columns=knn_marker_list)
# print(panel_C_ISPY_knn_imputed.shape)



# panel_C_combined_cell_embeddings = np.concatenate((dev_cell_embeddings, panel_C_vaccine_cell_embeddings, panel_C_ISPY_cell_embeddings))
# panel_C_combined_knn_imputed = pd.concat((dev_knn_imputed, panel_C_vaccine_knn_imputed, panel_C_ISPY_knn_imputed))

# panel_C_combined_labels = dev_labels + panel_C_vaccine_labels + panel_C_ISPY_labels
# panel_C_combined_datasets = ['dev']*len(dev_labels) + ['external1']*len(panel_C_vaccine_labels) + ['external2']*len(panel_C_ISPY_labels)

# plot_umap(panel_C_combined_knn_imputed, panel_C_combined_labels, "figs/simulation3/umap_panel_C_knn_celltype.png")
# plot_umap(panel_C_combined_cell_embeddings, panel_C_combined_labels, "figs/simulation3/umap_panel_C_cymae_celltype.png")
# plot_umap(panel_C_combined_knn_imputed, panel_C_combined_datasets, "figs/simulation3/umap_panel_C_knn_dataset.png")
# plot_umap(panel_C_combined_cell_embeddings, panel_C_combined_datasets, "figs/simulation3/umap_panel_C_cymae_dataset.png")










# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(panel_C_vaccine_data)

# unique_labels = np.sort(np.unique(panel_C_vaccine_labels))  # 알파벳(또는 숫자) 순서 정렬
# palette1 = sns.color_palette("hls", len(unique_labels))
# palette_all = {label: palette1[i] for i, label in enumerate(unique_labels)}

# df_all = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": panel_C_vaccine_labels
# })

# plt.figure(figsize=(10, 8))
# # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
# sns.scatterplot(
#     data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
#     palette=palette_all, s=10, alpha=0.7, legend="auto"
# )
# plt.title("UMAP Projection of Cell Embeddings (Label)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/simulation2/umap_panel_C_vaccine_ori.png")
# plt.clf()

# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(panel_C_vaccine_cell_embeddings)

# df_all = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": panel_C_vaccine_labels
# })

# plt.figure(figsize=(10, 8))
# # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
# sns.scatterplot(
#     data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
#     palette=palette_all, s=10, alpha=0.7, legend="auto"
# )
# plt.title("UMAP Projection of Cell Embeddings (Label)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/simulation2/umap_panel_C_vaccine_cymae.png")
# plt.clf()



# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(panel_C_ISPY_data)

# unique_labels = np.sort(np.unique(panel_C_ISPY_labels))  # 알파벳(또는 숫자) 순서 정렬
# palette1 = sns.color_palette("hls", len(unique_labels))
# palette_all = {label: palette1[i] for i, label in enumerate(unique_labels)}

# df_all = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": panel_C_ISPY_labels
# })

# plt.figure(figsize=(10, 8))
# # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
# sns.scatterplot(
#     data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
#     palette=palette_all, s=10, alpha=0.7, legend="auto"
# )
# plt.title("UMAP Projection of Cell Embeddings (Label)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/simulation2/umap_panel_C_ISPY_ori.png")
# plt.clf()

# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(panel_C_ISPY_cell_embeddings)

# df_all = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": panel_C_ISPY_labels
# })

# plt.figure(figsize=(10, 8))
# # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
# sns.scatterplot(
#     data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
#     palette=palette_all, s=10, alpha=0.7, legend="auto"
# )
# plt.title("UMAP Projection of Cell Embeddings (Label)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/simulation2/umap_panel_C_ISPY_cymae.png")
# plt.clf()



# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(panel_D_vaccine_data)

# unique_labels = np.sort(np.unique(panel_D_vaccine_labels))  # 알파벳(또는 숫자) 순서 정렬
# palette1 = sns.color_palette("hls", len(unique_labels))
# palette_all = {label: palette1[i] for i, label in enumerate(unique_labels)}

# df_all = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": panel_D_vaccine_labels
# })

# plt.figure(figsize=(10, 8))
# # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
# sns.scatterplot(
#     data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
#     palette=palette_all, s=10, alpha=0.7, legend="auto"
# )
# plt.title("UMAP Projection of Cell Embeddings (Label)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/simulation2/umap_panel_D_vaccine_ori.png")
# plt.clf()

# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(panel_D_vaccine_cell_embeddings)

# df_all = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": panel_D_vaccine_labels
# })

# plt.figure(figsize=(10, 8))
# # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
# sns.scatterplot(
#     data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
#     palette=palette_all, s=10, alpha=0.7, legend="auto"
# )
# plt.title("UMAP Projection of Cell Embeddings (Label)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/simulation2/umap_panel_D_vaccine_cymae.png")
# plt.clf()

# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(panel_D_vaccine_knn_imputed)

# df_all = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": panel_D_vaccine_labels
# })

# plt.figure(figsize=(10, 8))
# # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
# sns.scatterplot(
#     data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
#     palette=palette_all, s=10, alpha=0.7, legend="auto"
# )
# plt.title("UMAP Projection of Cell Embeddings (Label)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/simulation2/umap_panel_D_vaccine_knn.png")
# plt.clf()



# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(panel_D_ISPY_data)

# unique_labels = np.sort(np.unique(panel_D_ISPY_labels))  # 알파벳(또는 숫자) 순서 정렬
# palette1 = sns.color_palette("hls", len(unique_labels))
# palette_all = {label: palette1[i] for i, label in enumerate(unique_labels)}

# df_all = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": panel_D_ISPY_labels
# })

# plt.figure(figsize=(10, 8))
# # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
# sns.scatterplot(
#     data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
#     palette=palette_all, s=10, alpha=0.7, legend="auto"
# )
# plt.title("UMAP Projection of Cell Embeddings (Label)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/simulation2/umap_panel_D_ISPY_ori.png")
# plt.clf()

# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(panel_D_ISPY_cell_embeddings)

# df_all = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": panel_D_ISPY_labels
# })

# plt.figure(figsize=(10, 8))
# # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
# sns.scatterplot(
#     data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
#     palette=palette_all, s=10, alpha=0.7, legend="auto"
# )
# plt.title("UMAP Projection of Cell Embeddings (Label)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/simulation2/umap_panel_D_ISPY_cymae.png")
# plt.clf()


# reducer = umap.UMAP(n_components=2, random_state=42)
# umap_result = reducer.fit_transform(panel_D_ISPY_knn_imputed)

# df_all = pd.DataFrame({
#     "UMAP1": umap_result[:, 0],
#     "UMAP2": umap_result[:, 1],
#     "Label": panel_D_ISPY_labels
# })

# plt.figure(figsize=(10, 8))
# # seaborn은 hue 인자로 범주형 데이터를 받아 색상 팔레트를 자동 생성합니다.
# sns.scatterplot(
#     data=df_all, x="UMAP1", y="UMAP2", hue="Label", hue_order=list(unique_labels),
#     palette=palette_all, s=10, alpha=0.7, legend="auto"
# )
# plt.title("UMAP Projection of Cell Embeddings (Label)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.savefig("figs/simulation2/umap_panel_D_ISPY_knn.png")
# plt.clf()



