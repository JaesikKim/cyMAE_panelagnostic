import torch
from datasets import read_file
from run_mae_pretraining import get_model

import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os

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

sns.set_context("talk", font_scale=1.0)  # 기본 폰트 크기를 1.2배로


ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.5_lr_0.005_checkpoint-6000.pth"
device = 'cuda'

checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
ckpt_args = checkpoint['args']
ckpt_args.is_cumul_masking = True
model = get_model(ckpt_args)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/test_panel_C/'
test_filenames = os.listdir(path)
union_marker_to_index = {
    marker: idx for idx, marker in enumerate(ckpt_args.union_marker_list)
}

rng = np.random.default_rng(seed=42)

panels = [["CD45", "CD4", "CD8a", "CD11c", "CD161", "CD3", "CD14", "CD38", "CD294", "CD45RA", "CD57", "HLA-DR", "TCRgd", "CD16", "CD19", "CD20", "CD56", "CD66b", "CD123", "IgD"], # subset A
          ["CD45", "CD4", "CD8a", "CD11c", "CD161", "CD3", "CD14", "CD38", "CD294", "CD45RA", "CD57", "HLA-DR", "TCRgd", "CD16", "CD19", "CD20", "CD56", "CD66b", "CD123", "IgD"]+["CD45RO", "CD197"], 
          ["CD45", "CD4", "CD8a", "CD11c", "CD161", "CD3", "CD14", "CD38", "CD294", "CD45RA", "CD57", "HLA-DR", "TCRgd", "CD16", "CD19", "CD20", "CD56", "CD66b", "CD123", "IgD"]+["CD45RO", "CD197", "CD183", "CD185", "CD196"], 
          ["CD45", "CD4", "CD8a", "CD11c", "CD161", "CD3", "CD14", "CD38", "CD294", "CD45RA", "CD57", "HLA-DR", "TCRgd", "CD16", "CD19", "CD20", "CD56", "CD66b", "CD123", "IgD"]+["CD45RO", "CD197", "CD25", "CD27", "CD183", "CD185", "CD194", "CD196", "CD197"], 
         ]

panel_names = [
    "panel A",
    "+ Naive/Memory markers",
    "+ Th markers",
    "full panel",
]

new_cell_embeddings = []
new_panel_identity = []
new_labels = []
with torch.no_grad():
    for filename in test_filenames:
        data, labels, marker_list = read_file(path+filename)
        labels = simple_label_mapper(labels)

        selected_idx = undersample_above_target(labels, target_count=50)

        data = data[selected_idx]
        labels = np.array(labels)[selected_idx].tolist()

        for panel, panel_name in zip(panels, panel_names):
            marker_order = [i for i,m in enumerate(marker_list) if m in panel]
            new_data = data[:, marker_order]
            new_marker_list = [marker_list[i] for i in marker_order]

            new_marker_indices = []
            for marker in new_marker_list:
                if marker in ckpt_args.union_marker_list:
                    new_marker_indices.append(union_marker_to_index[marker])

            new_data = new_data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)

            with torch.amp.autocast('cuda'):
                cell_embeddings, pooled_embeddings = model.forward_inference(new_data, new_marker_indices)

            # new_data = new_data.cpu().numpy()
            new_cell_embeddings.append(cell_embeddings.cpu().numpy())
            new_panel_identity += [panel_name for _ in range(new_data.shape[1])]
            new_labels += labels

new_cell_embeddings = np.concatenate(new_cell_embeddings, axis=0)
print(new_cell_embeddings.shape)
reducer = umap.UMAP(n_components=2, random_state=42)
umap_result = reducer.fit_transform(new_cell_embeddings)

df_all = pd.DataFrame({
    "UMAP1": umap_result[:, 0],
    "UMAP2": umap_result[:, 1],
    "Label": new_labels,
    "Panel": new_panel_identity
})

import matplotlib.patches as mpatches

unique_labels = np.sort(np.unique(df_all["Label"]))
panel_order = panel_names
n_labels = len(unique_labels)
n_panels = len(df_all["Panel"].unique())

# 서브플롯 그리드 생성
n_cols = 4
n_rows = math.ceil(n_labels / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

# 패널별 색상 팔레트
panel_palette = sns.color_palette("Blues", n_colors=n_panels)

print("Generating the combined UMAP plot with a single legend...")
for i, cell_type in enumerate(unique_labels):
    row, col = i // n_cols, i % n_cols
    ax = axes[row, col]

    # 전경: 강조할 세포 유형
    df_target = df_all[df_all["Label"] == cell_type]
    
    # ✨ 변경점 1: `legend=False` 추가하여 서브플롯 범례 생성 방지
    sns.scatterplot(
        data=df_target, x="UMAP1", y="UMAP2", hue="Panel",
        hue_order=panel_order, palette=panel_palette,
        s=50, alpha=0.5, ax=ax,
        legend=False # 이 인자를 추가합니다.
    )

    x_min, x_max = df_target["UMAP1"].min(), df_target["UMAP1"].max()
    y_min, y_max = df_target["UMAP2"].min(), df_target["UMAP2"].max()
    
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_title(f"Highlight: {cell_type}", fontweight='bold', fontsize=22)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    # ✨ 변경점 2: ax.legend() 호출 삭제
    # ax.legend(title="Panel") # 이 라인을 삭제합니다.

# 남는 플롯 숨기기
for i in range(n_labels, n_rows * n_cols):
    axes.flatten()[i].axis('off')

# --- ✨ 새로운 부분: 그림 전체에 대한 대표 범례 생성 ---
# 범례에 사용할 핸들(색상 패치)을 직접 만듭니다.
handles = [
    mpatches.Patch(color=color, label=name)
    for name, color in zip(panel_order, panel_palette)
]
# 그림의 특정 위치에 범례를 추가합니다.
fig.legend(handles=handles, title="Panel", loc=(0.78, 0.1), fontsize=20)

# ✨ 변경점 3: 범례 공간 확보를 위해 tight_layout 조정

plt.tight_layout() # 전체 제목과 겹치지 않도록 조정
plt.savefig("figs/simulation2/trajectory_from_panel_A.png", dpi=150)
plt.close(fig)






panels = [["CD45", "CD4", "CD8a", "CD11c", "CD161", "CD3", "CD14", "CD38", "CD294", "CD45RA", "CD57", "HLA-DR", "TCRgd", "CD25", "CD27", "CD183", "CD185", "CD194", "CD196", "CD197", "CD45RO"], # subset B
          ["CD45", "CD4", "CD8a", "CD11c", "CD161", "CD3", "CD14", "CD38", "CD294", "CD45RA", "CD57", "HLA-DR", "TCRgd", "CD25", "CD27", "CD183", "CD185", "CD194", "CD196", "CD197", "CD45RO"] + ["CD19", "CD20", "IgD"],
          ["CD45", "CD4", "CD8a", "CD11c", "CD161", "CD3", "CD14", "CD38", "CD294", "CD45RA", "CD57", "HLA-DR", "TCRgd", "CD25", "CD27", "CD183", "CD185", "CD194", "CD196", "CD197", "CD45RO"] + ["CD19", "CD20", "IgD", "CD66b", "CD16", "CD123"],
          ["CD45", "CD4", "CD8a", "CD11c", "CD161", "CD3", "CD14", "CD38", "CD294", "CD45RA", "CD57", "HLA-DR", "TCRgd", "CD25", "CD27", "CD183", "CD185", "CD194", "CD196", "CD197", "CD45RO"] + ["CD19", "CD20", "IgD", "CD66b", "CD16", "CD123", "CD56"],
         ]

panel_names = [
    "panel B",
    "+ B cell markers",
    "+ Granulocyte markers",
    "full panel",
]

new_cell_embeddings = []
new_panel_identity = []
new_labels = []
with torch.no_grad():
    for filename in test_filenames:
        data, labels, marker_list = read_file(path+filename)
        labels = simple_label_mapper(labels)

        selected_idx = undersample_above_target(labels, target_count=50)

        data = data[selected_idx]
        labels = np.array(labels)[selected_idx].tolist()

        for panel, panel_name in zip(panels, panel_names):
            marker_order = [i for i,m in enumerate(marker_list) if m in panel]
            new_data = data[:, marker_order]
            new_marker_list = [marker_list[i] for i in marker_order]

            new_marker_indices = []
            for marker in new_marker_list:
                if marker in ckpt_args.union_marker_list:
                    new_marker_indices.append(union_marker_to_index[marker])

            new_data = new_data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)

            with torch.amp.autocast('cuda'):
                cell_embeddings, pooled_embeddings = model.forward_inference(new_data, new_marker_indices)

            # new_data = new_data.cpu().numpy()
            new_cell_embeddings.append(cell_embeddings.cpu().numpy())
            new_panel_identity += [panel_name for _ in range(new_data.shape[1])]
            new_labels += labels

new_cell_embeddings = np.concatenate(new_cell_embeddings, axis=0)
print(new_cell_embeddings.shape)
reducer = umap.UMAP(n_components=2, random_state=42)
umap_result = reducer.fit_transform(new_cell_embeddings)

df_all = pd.DataFrame({
    "UMAP1": umap_result[:, 0],
    "UMAP2": umap_result[:, 1],
    "Label": new_labels,
    "Panel": new_panel_identity
})


import matplotlib.patches as mpatches

unique_labels = np.sort(np.unique(df_all["Label"]))
panel_order = panel_names
n_labels = len(unique_labels)
n_panels = len(df_all["Panel"].unique())

# 서브플롯 그리드 생성
n_cols = 4
n_rows = math.ceil(n_labels / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

# 패널별 색상 팔레트
panel_palette = sns.color_palette("Reds", n_colors=n_panels)

print("Generating the combined UMAP plot with a single legend...")
for i, cell_type in enumerate(unique_labels):
    row, col = i // n_cols, i % n_cols
    ax = axes[row, col]

    # 전경: 강조할 세포 유형
    df_target = df_all[df_all["Label"] == cell_type]
    
    # ✨ 변경점 1: `legend=False` 추가하여 서브플롯 범례 생성 방지
    sns.scatterplot(
        data=df_target, x="UMAP1", y="UMAP2", hue="Panel",
        hue_order=panel_order, palette=panel_palette,
        s=50, alpha=0.5, ax=ax,
        legend=False # 이 인자를 추가합니다.
    )

    x_min, x_max = df_target["UMAP1"].min(), df_target["UMAP1"].max()
    y_min, y_max = df_target["UMAP2"].min(), df_target["UMAP2"].max()
    
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_title(f"Highlight: {cell_type}", fontweight='bold', fontsize=22)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    # ✨ 변경점 2: ax.legend() 호출 삭제
    # ax.legend(title="Panel") # 이 라인을 삭제합니다.

# 남는 플롯 숨기기
for i in range(n_labels, n_rows * n_cols):
    axes.flatten()[i].axis('off')

# --- ✨ 새로운 부분: 그림 전체에 대한 대표 범례 생성 ---
# 범례에 사용할 핸들(색상 패치)을 직접 만듭니다.
handles = [
    mpatches.Patch(color=color, label=name)
    for name, color in zip(panel_order, panel_palette)
]
# 그림의 특정 위치에 범례를 추가합니다.
fig.legend(handles=handles, title="Panel", loc=(0.78, 0.1), fontsize=20)

plt.tight_layout() # 전체 제목과 겹치지 않도록 조정
plt.savefig("figs/simulation2/trajectory_from_panel_B.png", dpi=150)
plt.close(fig)



