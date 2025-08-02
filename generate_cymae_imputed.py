
# import os
# import glob
# import torch
# import pandas as pd
# import numpy as np

# import torch
# from datasets import read_file
# from run_mae_pretraining import get_model


# rng = np.random.default_rng(seed=42)


# ckpt = "./ckpts/exp2_dmodel_32_subset_size_1000_fps_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_maxstep_1_celllambda_0.5_lr_0.005_checkpoint-6000.pth"

# device = 'cuda'

# checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
# ckpt_args = checkpoint['args']
# print(ckpt_args)
# model = get_model(ckpt_args)
# model.load_state_dict(checkpoint['model'])
# model.to(device)
# model.eval()

# union_marker_to_index = {
#     marker: idx for idx, marker in enumerate(ckpt_args.union_marker_list)
# }

# print(model.is_fake_mask)

# path = '/project/kimgroup_immune_health/data/pan_panel/simulation2/dev/'

# for filename in os.listdir(path):
#     data, labels, marker_list = read_file(path+filename)

#     marker_indices = []
#     for marker in marker_list:
#         if marker in ckpt_args.union_marker_list:
#             marker_indices.append(union_marker_to_index[marker])

#     data = data.unsqueeze(0).to(device, non_blocking=True).to(torch.float)
#     with torch.no_grad():
#         with torch.amp.autocast('cuda'):
#             data = model.forward_all_marker_inference(data, [marker_indices])
#     break