import json
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'loss_log.jsonl' with the path to your log file

# log_file_path = "/home/jaesik/cyMAE_panelagnostic_light/ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_5.0_lr_0.005_log.txt"
# fig_path = "figs/loss/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss_maskingalpha_0.0_maxstep_1_celllambda_5.0_lr_0.005.png"

# log_file_path = "/home/jaesik/cyMAE_panelagnostic_light/ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_2.0_lr_0.005_log.txt"
# fig_path = "figs/loss/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss_maskingalpha_0.0_maxstep_1_celllambda_2.0_lr_0.005.png"

# log_file_path = "/home/jaesik/cyMAE_panelagnostic_light/ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_1.0_lr_0.005_log.txt"
# fig_path = "figs/loss/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss_maskingalpha_0.0_maxstep_1_celllambda_1.0_lr_0.005.png"

log_file_path = "/home/jaesik/cyMAE_panelagnostic_light/ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.0_lr_0.005_log.txt"
fig_path = "figs/loss/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss_maskingalpha_0.0_maxstep_1_celllambda_0.0_lr_0.005.png"





# log_file_path = "/home/jaesik/cyMAE_panelagnostic_light/ckpts/exp2_dmodel_32_subset_size_1000_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.0_lr_0.005_log.txt"
# fig_path = "figs/loss/exp2_dmodel_32_subset_size_1000_no_pred_rank_no_adv_loss_maskingalpha_0.0_maxstep_1_celllambda_0.0_lr_0.005.png"

# Read the JSONL log file
records = []
with open(log_file_path, 'r') as f:
    for line in f:
        records.append(json.loads(line.strip()))

# Create a DataFrame
df = pd.DataFrame(records)
df = df[df['epoch']>500]

# Plot the training loss curve
plt.figure(figsize=(12,3))
plt.plot(df['epoch'], df['train_loss'], marker='o')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss Curve')
plt.tight_layout()
plt.savefig(fig_path)
plt.close()



