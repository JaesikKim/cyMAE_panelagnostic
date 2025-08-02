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

# import ot
# def compute_emd(A,B):
#     # uniform weights assumed
#     n = A.shape[0]
#     M = torch.cdist(A, B).numpy()  # cost matrix
#     a = b = (1.0 / n) * torch.ones(n).numpy()

#     emd = ot.emd2(a, b, M)
#     return emd

# def gaussian_kernel(x, y, sigma=1.0):
#     beta = 1.0 / (2.0 * sigma**2)
#     dist = torch.cdist(x, y) ** 2
#     return torch.exp(-beta * dist)

# def compute_mmd(x, y, sigma=1.0):
#     Kxx = gaussian_kernel(x, x, sigma).mean()
#     Kyy = gaussian_kernel(y, y, sigma).mean()
#     Kxy = gaussian_kernel(x, y, sigma).mean()
#     return Kxx + Kyy - 2 * Kxy

sns.set_context("talk", font_scale=1.0)  # 기본 폰트 크기를 1.2배로


# ckpt = "./ckpts/dmodel_32_no_pred_rank_no_adv_loss/cyMAE2_maskingalpha_1.0_celllambda_0.0_lr_0.005_checkpoint-5000.pth"

ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.5_lr_0.005_checkpoint-6000.pth"
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


rng = np.random.default_rng(seed=42)


# panel identity on cyMAE
path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/test_panel_C/'
test_filenames = os.listdir(path)


panel_A = ["CD45", "CD123", "CD19", "CD4", "CD8a",
            "CD11c", "CD16", "CD161", "CD57", "CD38",
            "CD56", "CD294", "CD14", "CD3", "CD20",
            "CD66b", "HLA-DR", "IgD", "TCRgd", "CD45RA"]

panel_B = ["CD45", "CD196", "CD4", "CD8a", "CD11c", 
            "CD161", "CD45RO", "CD45RA", "CD194", "CD25",
            "CD27", "CD57", "CD183", "CD185", "CD38",
            "CD294", "CD197", "CD14", "CD3", "HLA-DR", 
            "TCRgd"]


test_cell_embeddings = []
test_panel_identity = []
test_labels = []

with torch.no_grad():
    for filename in test_filenames:
        data, labels, marker_list = read_file(path+filename)
        labels = simple_label_mapper(labels)

        # balanced sampling
        selected_idx = undersample_above_target(labels, target_count=50)
        data = data[selected_idx]
        labels = np.array(labels)[selected_idx].tolist()

        # panel A
        marker_order = [i for i,m in enumerate(marker_list) if m in panel_A]
        new_data = data[:, marker_order]
        new_marker_list = [marker_list[i] for i in marker_order]

        new_marker_indices = []
        for marker in new_marker_list:
            if marker in ckpt_args.union_marker_list:
                new_marker_indices.append(union_marker_to_index[marker])

        new_data = new_data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)

        with torch.amp.autocast('cuda'):
            cell_embeddings_panelA, _ = model.forward_inference(new_data, new_marker_indices)

        test_cell_embeddings.append(cell_embeddings_panelA.cpu())
        test_panel_identity += ['Panel A' for _ in range(new_data.shape[1])]
        test_labels += labels

        # panel B
        marker_order = [i for i,m in enumerate(marker_list) if m in panel_B]
        new_data = data[:, marker_order]
        new_marker_list = [marker_list[i] for i in marker_order]

        new_marker_indices = []
        for marker in new_marker_list:
            if marker in ckpt_args.union_marker_list:
                new_marker_indices.append(union_marker_to_index[marker])

        new_data = new_data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)

        with torch.amp.autocast('cuda'):
            cell_embeddings_panelB, _ = model.forward_inference(new_data, new_marker_indices)

        test_cell_embeddings.append(cell_embeddings_panelB.cpu())
        test_panel_identity += ['Panel B' for _ in range(new_data.shape[1])]
        test_labels += labels


test_cell_embeddings = torch.cat(test_cell_embeddings)
test_panel_identity = np.array(test_panel_identity)
test_labels = np.array(test_labels)


# for cell_type in np.unique(test_labels):
#     msk = test_labels == cell_type
#     embeddings_sub = test_cell_embeddings[test_labels == cell_type]
#     panel_sub = test_panel_identity[test_labels == cell_type]

#     embeddings_sub_panelA = embeddings_sub[panel_sub == 'Panel A']
#     embeddings_sub_panelB = embeddings_sub[panel_sub == 'Panel B']

#     emd = compute_emd(embeddings_sub_panelA, embeddings_sub_panelB)
#     print(cell_type, emd)







import matplotlib.patches as mpatches

reducer = umap.UMAP(n_components=2, random_state=42)
umap_result = reducer.fit_transform(test_cell_embeddings)

unique_labels = np.sort(np.unique(test_labels))  # 알파벳(또는 숫자) 순서 정렬
palette1 = sns.color_palette("hls", len(unique_labels))
palette_label = {label: palette1[i] for i, label in enumerate(unique_labels)}

df_all = pd.DataFrame({
    "UMAP1": umap_result[:, 0],
    "UMAP2": umap_result[:, 1],
    "Label": test_labels,
    "Panel": test_panel_identity
})

# 패널별 색상을 지정
panel_colors = {'Panel A': 'dodgerblue', 'Panel B': 'orangered'}

# --- 배경 세포를 포함하여 UMAP 시각화 (일관된 축 스케일) ---

unique_labels = np.sort(np.unique(test_labels))
n_labels = len(unique_labels)

# 서브플롯 그리드 설정
n_cols = 4
n_rows = math.ceil(n_labels / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

# 모든 서브플롯에 동일한 축 범위를 적용하기 위해 전체 데이터의 min/max를 계산
x_min, x_max = df_all["UMAP1"].min(), df_all["UMAP1"].max()
y_min, y_max = df_all["UMAP2"].min(), df_all["UMAP2"].max()
# 약간의 여백(padding)을 줌
x_pad = (x_max - x_min) * 0.05
y_pad = (y_max - y_min) * 0.05

print("Generating UMAP comparison plots with a single figure legend...")
for idx, label in enumerate(unique_labels):
    row, col = idx // n_cols, idx % n_cols
    ax = axes[row, col]

    # 1. 배경 세포 그리기
    df_background = df_all[df_all["Label"] != label]
    ax.scatter(
        df_background["UMAP1"], df_background["UMAP2"],
        color='lightgray', s=8, alpha=0.4, rasterized=True
    )

    # 2. 강조할 세포 유형 그리기
    df_target = df_all[df_all["Label"] == label]
    for panel, color in panel_colors.items():
        df_panel = df_target[df_target["Panel"] == panel]
        ax.scatter(
            df_panel["UMAP1"], df_panel["UMAP2"],
            label=panel, s=20, alpha=0.8, color=color
        )

    # 3. 이동 경로 그리기
    dfa = df_target[df_target['Panel'] == 'Panel A'].reset_index(drop=True)
    dfb = df_target[df_target['Panel'] == 'Panel B'].reset_index(drop=True)
    for i in range(min(len(dfa), len(dfb))):
        x0, y0 = dfa.loc[i, ['UMAP1','UMAP2']]
        x1, y1 = dfb.loc[i, ['UMAP1','UMAP2']]
        ax.plot([x0, x1], [y0, y1], color='black', alpha=0.25, linewidth=0.4)

    # 4. 축 범위 및 기타 설정
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title(f"Highlight: {label}", fontweight='bold')
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    # ax.legend()를 여기서 호출하지 않습니다.
    ax.grid(False)

# 비어있는 서브플롯 숨기기
for empty_idx in range(n_labels, n_rows * n_cols):
    axes.flatten()[empty_idx].axis('off')

# --- ✨ 새로운 부분: 그림 전체에 대한 대표 범례 생성 ---
# 범례에 사용할 핸들(색상 패치)을 직접 만듭니다.
handles = [
    mpatches.Patch(color=color, label=panel) for panel, color in panel_colors.items()
]
# 그림의 특정 위치에 범례를 추가합니다.
fig.legend(handles=handles, title="Panel", loc=(0.83, 0.1), fontsize=20)

# 범례 추가 후 레이아웃을 다시 조정하여 겹침을 방지합니다.
plt.tight_layout()
# plt.tight_layout(rect=[0, 0, 0.85, 1]) # rect를 조정하여 범례 공간 확보


# unique_labels = np.sort(np.unique(test_labels))
# n_labels = len(unique_labels)

# # 서브플롯 그리드 설정
# n_cols = 4
# n_rows = math.ceil(n_labels / n_cols)

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

# # 모든 서브플롯에 동일한 축 범위를 적용하기 위해 전체 데이터의 min/max를 계산
# x_min, x_max = df_all["UMAP1"].min(), df_all["UMAP1"].max()
# y_min, y_max = df_all["UMAP2"].min(), df_all["UMAP2"].max()
# # 약간의 여백(padding)을 줌
# x_pad = (x_max - x_min) * 0.05
# y_pad = (y_max - y_min) * 0.05

# print("Generating UMAP comparison plots with shared context...")
# for idx, label in enumerate(unique_labels):
#     row, col = idx // n_cols, idx % n_cols
#     ax = axes[row, col]

#     # 1. 강조할 세포 유형(label)을 제외한 나머지 모든 세포를 회색 배경으로 그리기
#     df_background = df_all[df_all["Label"] != label]
#     ax.scatter(
#         df_background["UMAP1"],
#         df_background["UMAP2"],
#         color='lightgray',
#         s=8,
#         alpha=0.4,
#         rasterized=True # 많은 점을 효율적으로 렌더링
#     )

#     # 2. 강조할 세포 유형만 패널별로 색상을 주어 위에 그리기
#     df_target = df_all[df_all["Label"] == label]
#     for panel, color in panel_colors.items():
#         df_panel = df_target[df_target["Panel"] == panel]
#         ax.scatter(
#             df_panel["UMAP1"], df_panel["UMAP2"],
#             label=panel,
#             s=20,  # 배경보다 점을 약간 크게
#             alpha=0.8,
#             color=color
#         )

#     # 3. 강조하는 세포 유형의 패널 A -> B 이동 경로 그리기
#     dfa = df_target[df_target['Panel'] == 'Panel A'].reset_index(drop=True)
#     dfb = df_target[df_target['Panel'] == 'Panel B'].reset_index(drop=True)
#     for i in range(min(len(dfa), len(dfb))):
#         x0, y0 = dfa.loc[i, ['UMAP1','UMAP2']]
#         x1, y1 = dfb.loc[i, ['UMAP1','UMAP2']]
#         ax.plot([x0, x1], [y0, y1], color='black', alpha=0.25, linewidth=0.4)

#     # 4. 모든 서브플롯의 축 범위를 동일하게 설정
#     ax.set_xlim(x_min - x_pad, x_max + x_pad)
#     ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
#     ax.set_title(f"Highlight: {label}", fontweight='bold')
#     ax.set_xlabel("UMAP1")
#     ax.set_ylabel("UMAP2")
#     ax.legend(title="Panel")
#     ax.grid(False)


# # 비어있는 서브플롯 숨기기
# for empty_idx in range(n_labels, n_rows * n_cols):
#     axes.flatten()[empty_idx].axis('off')

# plt.tight_layout()

fig.savefig(f"figs/simulation2/test_panel_comparison.png", dpi=150)
plt.close(fig)

print("\nSaved context-aware UMAP plots to figs/simulation2/test_panel_comparison_with_context.png")




all_distances = {}
mean_distances = {}

# 각 세포 유형별로 거리 계산
unique_cell_types = np.unique(test_labels)
print("Calculating pairwise distances for each cell type...")

for cell_type in unique_cell_types:
    # 현재 세포 유형에 해당하는 데이터만 필터링
    embeddings_sub = test_cell_embeddings[test_labels == cell_type]
    panel_sub = test_panel_identity[test_labels == cell_type]

    # 패널 A와 패널 B의 임베딩을 분리
    embeddings_sub_panelA = embeddings_sub[panel_sub == 'Panel A']
    embeddings_sub_panelB = embeddings_sub[panel_sub == 'Panel B']

    # 데이터 로딩 로직상 두 패널의 셀 개수는 동일해야 함
    # 안전을 위해 길이를 맞춤
    min_len = min(len(embeddings_sub_panelA), len(embeddings_sub_panelB))
    if min_len == 0:
        continue
        
    embeddings_sub_panelA = embeddings_sub_panelA[:min_len]
    embeddings_sub_panelB = embeddings_sub_panelB[:min_len]

    # 셀 쌍(pair) 간의 유클리드 거리 계산
    # torch.norm(a - b, dim=1)는 각 행(셀) 쌍의 L2 norm을 계산합니다.
    distances = torch.norm(embeddings_sub_panelA - embeddings_sub_panelB, p=2, dim=1).numpy()
    
    all_distances[cell_type] = distances
    mean_distances[cell_type] = np.mean(distances)
    
    print(f"  - {cell_type}: Mean Distance = {mean_distances[cell_type]:.4f} (n={len(distances)})")

# --- 히스토그램 시각화 ---


plt.style.use('seaborn-v0_8-whitegrid') # 깔끔한 스타일 적용
fig, ax = plt.subplots(figsize=(13, 8))

# 각 세포 유형에 고유한 색상을 할당하기 위한 팔레트 생성
unique_cell_types = sorted(all_distances.keys())
palette = sns.color_palette("hls", len(unique_cell_types))
color_map = {cell_type: color for cell_type, color in zip(unique_cell_types, palette)}

# 평균 거리가 작은 순서대로 정렬하여 플롯 (가독성 향상)
sorted_types_by_dist = sorted(mean_distances, key=mean_distances.get)

print("\nPlotting overlaid KDEs...")

for cell_type in sorted_types_by_dist:
    distances = all_distances[cell_type]
    mean_dist = mean_distances[cell_type]
    
    # sns.kdeplot을 사용하여 분포 곡선을 그림
    sns.kdeplot(
        distances,
        ax=ax,
        color=color_map[cell_type],
        label=f"{cell_type} (μ={mean_dist:.2f})", 
        linewidth=2.5,
        alpha=0.8
    )

# ax.set_title('Pairwise Embedding Distance Distributions (Panel A vs B)', fontsize=18, pad=15)
ax.set_xlabel('Pairwise Embedding Distance between Panel A and Panel B', fontsize=18)
ax.set_ylabel('Density', fontsize=18)

# 범례를 그래프 바깥 오른쪽에 배치
ax.legend(title='Cell Type (Mean Distance)', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=13)

# 레이아웃을 조정하여 범례가 잘리지 않도록 함
plt.tight_layout(rect=[0, 0, 0.85, 1])
fig.savefig(f"figs/simulation2/distance_kde_overlay.png", dpi=150)
plt.close(fig)

print("\nSaved overlaid KDE plot to figs/simulation2/distance_kde_overlay.png")

