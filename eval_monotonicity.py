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
from scipy.stats import ttest_1samp, linregress, ttest_rel
import math

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

subset_A = ["CD45", "CD123", "CD19", "CD4", "CD8a",
            "CD11c", "CD16", "CD161", "CD57", "CD38",
            "CD56", "CD294", "CD14", "CD3", "CD20",
            "CD66b", "HLA-DR", "IgD", "TCRgd", "CD45RA"]

## T cell focused panel
subset_B = ["CD45", "CD196", "CD4", "CD8a", "CD11c", 
            "CD161", "CD45RO", "CD45RA", "CD194", "CD25",
            "CD27", "CD57", "CD183", "CD185", "CD38",
            "CD294", "CD197", "CD14", "CD3", "HLA-DR", 
            "TCRgd"]

# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
rng = np.random.default_rng(seed=42)

def generate_random_paths(full_panel, num_paths, rng):
    """
    full_panel: list of markers (length ≥ 3)
    num_paths: how many [full, step1, step2] paths to generate
    rng: a np.random.Generator
    """
    paths = []
    for _ in range(num_paths):
        current = full_panel.copy()
        path = [current.copy()]

        # 1st removal
        max_remove1 = len(current) - 1
        remove_count1 = int(rng.integers(1, max_remove1+1))
        removed1 = list(rng.choice(current, size=remove_count1, replace=False))
        current = [m for m in current if m not in removed1]
        path.append(current.copy())

        # 2nd removal
        max_remove2 = len(current)
        remove_count2 = int(rng.integers(1, max_remove2+1))
        removed2 = list(rng.choice(current, size=remove_count2, replace=False))
        current = [m for m in current if m not in removed2]
        path.append(current.copy())

        # also record how many markers were removed at each step
        paths.append({
            'panels': path,
            'removed1': remove_count1,
            'removed2': remove_count2
        })
    return paths

# 2) Generate your paths
full_panel = sorted(set(subset_A) | set(subset_B))
random_paths = generate_random_paths(full_panel, num_paths=1000, rng=rng)


marker_diff = []
dist_diff   = []

cell_removed = []
cell_dist    = []
cell_labels  = []

for filename in test_filenames:
    full_data, full_labels, marker_list = read_file(path + filename)

    # full_labels: 길이 1000, 각 cell의 타입(문자열)
    labels1000 = full_labels[:100]

    for path_info in random_paths:
        panels = path_info['panels']
        rem1, rem2 = path_info['removed1'], path_info['removed2']

        # 단계별 embedding
        Es = []
        for panel in panels:
            data = full_data[:100]
            cols = [i for i,m in enumerate(marker_list) if m in panel]
            tensor = torch.tensor(data[:, cols]).unsqueeze(0).to(device).float()
            with torch.no_grad(), torch.amp.autocast('cuda'):
                emb, _ = model.forward_inference(
                    tensor,
                    [union_marker_to_index[m] for m in panel if m in ckpt_args.union_marker_list]
                )
            Es.append(emb.cpu().numpy())

        E0, E1, E2 = Es
        d1 = np.linalg.norm(E0 - E1, axis=1)
        d2 = np.linalg.norm(E0 - E2, axis=1)

        # per-cell diff
        dd = d2 - d1
        md = rem2

        dist_diff.extend(dd)
        marker_diff.extend([md] * len(dd))
        # 기록
        cell_removed.extend([rem1] * len(d1))
        cell_dist   .extend(d1)
        cell_labels .extend(labels1000)

        cell_removed.extend([rem1+rem2] * len(d2))
        cell_dist   .extend(d2)
        cell_labels .extend(labels1000)

# numpy array 로 변환
marker_diff = np.array(marker_diff)
dist_diff   = np.array(dist_diff)

# 2) DataFrame
df = pd.DataFrame({
    'marker_diff': marker_diff,
    'dist_diff':   dist_diff
})

# 양측검정
t_stat, p_two    = ttest_1samp(dist_diff, popmean=0)
# 단측검정 (μ>0)
p_one_if_pos = p_two/2 if t_stat>0 else 1 - p_two/2

print(f"t = {t_stat:.3f}, p(two-sided) = {p_two:.3e}")
print(f"p(one-sided, Δd>0) = {p_one_if_pos:.3e}")


bins   = [1, 3, 6, 9, 12, 15, 18, 21, np.inf]
labels = ['= [1,3)', '= [3,6)', '= [6,9)', '= [9,12)', '= [12,15)', '= [15,18)', '= [18,21)', '>= 21']
labels=[f'$m_{{large}} - m_{{small}}$ {bin}' for bin in labels]

df['md_bin'] = pd.cut(df['marker_diff'], bins=bins, labels=labels, right=False)

# 3) 시각화
plt.figure(figsize=(12, 6))
for lbl, color in zip(labels, sns.color_palette("tab10", len(labels))):
    subset = df.loc[df['md_bin'] == lbl, 'dist_diff']
    sns.kdeplot(
        subset,
        fill=True,
        alpha=0.4,
        linewidth=1.5,
        label=lbl,
        color=color
    )

plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel(r'$d(E_{full}-E_{large}) - d(E_{full}-E_{small})$')
plt.ylabel('Density')
plt.legend(title='Marker size diff', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig('figs/simulation2/histogram_embdistdiff.png')
plt.close()



# 4) numpy 변환
cell_removed = np.array(cell_removed)
cell_dist    = np.array(cell_dist)
cell_labels  = np.array(cell_labels)
print(cell_removed.shape, cell_dist.shape, cell_labels.shape)


df = pd.DataFrame({
    'removed': cell_removed,
    'distance': cell_dist
})

# Compute regression stats
slope, intercept, r_value, p_value, std_err = linregress(df['removed'], df['distance'])
r_squared = r_value**2

# 2) 평균 ± 표준편차 라인
sns.lineplot(
    x='removed', y='distance',
    data=df,
    estimator=np.mean,
    ci='sd',
    marker='o',
    linewidth=2,
    err_kws={'alpha':0.3},
    label='Mean ± SD'
)

# 3) 선형회귀 추세선 (regplot 쓰지 않고, 수식으로 직접)
x_vals = np.array([df['removed'].min(), df['removed'].max()])
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=2, label='Linear fit')

# 4) 통계치 annotation
plt.text(
    0.02, 0.98,
    f"Slope = {slope:.3f}\nR² = {r_squared:.2f}\np = {p_value:.1e}",
    transform=plt.gca().transAxes,
    va='top',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
)

plt.xlabel(r'$m_{full}-m_{subset}$')
plt.ylabel(r'$d(E_{full}, E_{subset}$')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figs/simulation2/linplot_markerdiff_embdist.png')
plt.close()
print(r_squared, p_value)


