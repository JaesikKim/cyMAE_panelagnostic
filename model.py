
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import itertools

from timm.models.layers import trunc_normal_ as __call_trunc_normal_

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_utils import FlashAttnBlock, Block, get_sinusoid_encoding_table
    
class DataAugmentation(object):
    def __init__(self, subset_size, device, seed, fps=True):
        self.subset_size = subset_size
        self.fps = fps
        # Generator
        self.rng = np.random.default_rng(seed=seed)
        self.device = device

    def undersample_above_target(self, labels_arr, target_count=None):
        unique_labels, counts = np.unique(labels_arr, return_counts=True)
        if target_count is None:
            target_count = int(np.min(counts))
            # print(target_count)
        selected_idx = []
        for lbl, cnt in zip(unique_labels, counts):

            idx = np.where(labels_arr == lbl)[0]
            if cnt > target_count:
                pick = self.rng.choice(idx, size=target_count, replace=False)
            else:
                pick = idx
            selected_idx.append(pick)
        selected_idx = np.concatenate(selected_idx)
        return selected_idx
    
    def farthest_point_sample(self, cells, n_samples):
        B, C, M = cells.shape
        device = cells.device
        dtype = cells.dtype
        
        sampled_cells_list = []
        
        # 각 배치 아이템에 대해 개별적으로 FPS 수행
        for b in range(B):
            batch_cells = cells[b]  # Shape: (C, M)
            
            # 샘플링된 점들의 인덱스를 저장할 텐서
            centroids_indices = torch.zeros(n_samples, device=device, dtype=torch.long)
            
            # 첫 번째 점은 무작위로 선택
            first_idx = torch.randint(0, C, (1,), device=device).item()
            centroids_indices[0] = first_idx
            
            # 선택된 첫 번째 중심점
            centroid = batch_cells[first_idx].unsqueeze(0)  # Shape: (1, M)
            
            # 모든 점들과 첫 번째 중심점 간의 제곱 거리 계산
            dist_to_centroid = torch.sum((batch_cells - centroid) ** 2, dim=1)
            distances = dist_to_centroid
            
            # 나머지 n_samples - 1개의 점 선택
            for i in range(1, n_samples):
                # 현재까지 선택된 중심점들로부터 가장 멀리 떨어진 점의 인덱스 탐색
                next_idx = torch.argmax(distances)
                centroids_indices[i] = next_idx
                
                # 새로 선택된 중심점
                new_centroid = batch_cells[next_idx].unsqueeze(0)  # Shape: (1, M)
                
                # 모든 점들과 새로운 중심점 간의 제곱 거리 계산
                dist_to_new_centroid = torch.sum((batch_cells - new_centroid) ** 2, dim=1)
                
                # 기존 최소 거리와 새로운 중심점까지의 거리 중 더 작은 값으로 distances 업데이트
                distances = torch.minimum(distances, dist_to_new_centroid)
                
            # 최종 선택된 인덱스들을 사용하여 점들을 샘플링
            sampled_cells_list.append(batch_cells[centroids_indices])
            
        # 배치별로 샘플링된 점들을 하나의 텐서로 결합
        sampled_cells = torch.stack(sampled_cells_list, dim=0)
        
        return sampled_cells

    def __call__(self, cells, labels=None): # (B, C_total, M)
        total_cells = cells.shape[1]
        if self.subset_size > total_cells:
            if labels is not None:
                return cells, labels
            else:
                return cells
        else:
            if labels is not None:
                subset_cells = []
                subset_labels = []
                for b in range(cells.shape[0]):
                    # GPU→CPU→numpy
                    lb = labels[b].cpu().numpy()
                    # label balance 위해 undersample_above_target 사용
                    chosen = self.undersample_above_target(lb)
                    sel_cells = cells[b][chosen]
                    sel_labels = labels[b][chosen]
                    subset_cells.append(sel_cells.unsqueeze(0))
                    subset_labels.append(sel_labels.unsqueeze(0))
                subset_cells  = torch.cat(subset_cells,  dim=0)
                subset_labels = torch.cat(subset_labels, dim=0)

                return subset_cells, subset_labels
            elif self.fps:
                return self.farthest_point_sample(cells, self.subset_size)
                
            else:
                indices = self.rng.choice(total_cells, size=self.subset_size, replace=False)
                subset_cells = cells[:, indices]
                return subset_cells



    def __repr__(self):
        repr = "(Augment cells by samping " % str(self.n)
        return repr
    

class MarkerAttentionBlock(nn.Module):
    """
    Self-attention block across markers.
    Input tensor x shape: (B, C, M+1, d_model)
    Each marker does attention across markers.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr, norm_layer, init_values):
        super().__init__()

        # self.block = Block(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
        #         init_values=init_values)
        self.block = FlashAttnBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
                init_values=init_values)

    def forward(self, x):

        B, C, M, d_model = x.shape
        x = x.reshape(B * C, M, d_model)
        x = self.block(x)
        x = x.reshape(B, C, M, d_model)
        return x


def trunc_normal_(tensor, mean=0., std=1.):

    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    

class cyMAEEncoder(nn.Module):
    """ Vision Transformer
    """
    def __init__(self, num_classes=0, embed_dim=32, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # self.patient_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # trunc_normal_(self.patient_query, std=.02)

        self.feature_value_mixer = nn.Linear(embed_dim, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # Transformer blocks
        blocks = []
        for i in range(depth):
            blocks.append(MarkerAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[i], norm_layer, init_values))

        self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        ) if num_classes > 0 else nn.Identity() # nn.Identity()
        # self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}
    
    def get_num_layers(self):
        return len(self.blocks)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, num_classes)
        ) if num_classes > 0 else nn.Identity()
        # self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, marker_embeddings, mask):
        # x: (B, C, M)
        # marker_embeddings: (B, C, M, emb)
        # mask: (B, C, M)

        # input embedding
        # value embeddings, marker embeddings

        value = x.unsqueeze(-1).expand(-1, -1, -1, int(self.embed_dim/2)) # (B, C, M, emb)
        x_token = torch.concat((marker_embeddings, value), axis=3) # x: (B, C, M, 2*emb)
        x_token = self.feature_value_mixer(x_token)

        B, C, M, emb = x_token.shape

        # mask input
        x_unmasked_token = x_token[~mask].reshape(B, C, -1, emb) # ~mask means visible

        # cls token
        cls_token = self.cls_token.expand(B, C, 1, emb)
        x_unmasked_token = torch.cat((cls_token, x_unmasked_token), dim=2) # x: (B, C, 1+M_vis, emb)

        # x_token = x_token + self.pos_embed.type_as(x_token).to(x_token.device).clone().detach()

        for i, blk in enumerate(self.blocks):
            x_unmasked_token = blk(x_unmasked_token)
        x_unmasked_token = self.norm(x_unmasked_token)
        cell_token = x_unmasked_token[:,:,0,:] # (B, C, emb)
        cell_token = self.head(cell_token)

        return cell_token, x_unmasked_token
    

class cyMAEDecoder(nn.Module):

    """ Vision Transformer
    """
    def __init__(self, is_fake_mask=True, is_adv_loss=False, num_classes=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None
                 ):
        super().__init__()
        self.num_classes = num_classes
#         assert num_classes == 3 * patch_size ** 2
        self.embed_dim = embed_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # Transformer blocks
        blocks = []
        for i in range(depth):
            blocks.append(MarkerAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[i], norm_layer, init_values))

        self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() # nn.Linear()

        self.is_fake_mask = is_fake_mask
        self.is_adv_loss = is_adv_loss
        if is_adv_loss:
            self.adv_head = nn.Linear(embed_dim*2, 1)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mask_token'}
    
    def get_num_layers(self):
        return len(self.blocks)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, marker_embeddings, fake_marker_embeddings, mask, is_adv_loss=True):
        # x: (B, C, M_vis, emb)
        # marker_embeddings: (B, C, M, emb)
        # fake_marker_embeddings: (B, C, union_M-M, emb)
        # mask: (B, C, M)

        B, C, M_vis, emb = x.shape
        marker_embeddings_unmasked = torch.cat((torch.zeros(B,C,1,emb, device=x.device), marker_embeddings[~mask].reshape(B,C,-1,emb)), dim=2)
        marker_embeddings_true_masked = marker_embeddings[mask].reshape(B,C,-1,emb)
        marker_embeddings_fake_masked = fake_marker_embeddings
        if self.is_fake_mask:
            marker_embeddings_masked = torch.cat((marker_embeddings_fake_masked, marker_embeddings_true_masked), dim=2)
        else:
            marker_embeddings_masked = marker_embeddings_true_masked
        M_t = marker_embeddings_true_masked.size(2)
        M_f = marker_embeddings_fake_masked.size(2)
        # reconstruction
        x_full = torch.cat([x + marker_embeddings_unmasked, self.mask_token + marker_embeddings_masked], dim=2) # (B, C, 1+M, dim)
        for i,blk in enumerate(self.blocks):
            x_full = blk(x_full)

        if M_t > 0:
            x_full = self.norm(x_full[:, :, -M_t:])
        else:
            x_full = self.norm(x_full)
        x_recon = self.head(x_full) # [B, M_masked, 1] or [B, M_masked, 2]


        if self.is_adv_loss:

            # true/fake mask prediction
            cell_token = x[:,:,0,:]
            cell_unmasked = cell_token.unsqueeze(2).expand(-1, -1, M_vis, -1)  # (B, C, M_vis, emb)
            cell_masked = cell_token.unsqueeze(2).expand(-1, -1, M_t+M_f, -1)  # (B, C, M_f+M_t, emb)

            adv_in_unmasked = torch.cat([cell_unmasked, marker_embeddings_unmasked], dim=-1)
            adv_in_masked = torch.cat([cell_masked, marker_embeddings_masked], dim=-1)
            adv_in = torch.cat([adv_in_unmasked, adv_in_masked], dim=2)  # (B, C, M_vis+M_t+M_f, 2*emb)

            # adv_in = adv_in.view(-1, 2 * emb)       # (B*C*(M_vis+M_t+M_f), 2*emb)
            logits = self.adv_head(adv_in).squeeze(-1)         # (B, C, (M_vis+M_t+M_f))

            # --- BCE 기반 confusion loss 계산 ---
            labels_true = torch.zeros(B, C, M_vis,    device=logits.device)
            labels_fake = torch.ones( B, C, M_t+M_f, device=logits.device)
            labels = torch.cat([labels_true, labels_fake], dim=2)  # (B, C, M_vis+M_t+M_f)

            # 8) flatten & BCEWithLogits
            logits_flat = logits.view(-1)              # (B*C*(M_vis+M_t+M_f),)
            labels_flat = labels.view(-1)              # (B*C*(M_vis+M_t+M_f),)
            bce = F.binary_cross_entropy_with_logits(logits_flat, labels_flat, reduction='mean')

            adv_confusion_loss = -bce
        else:
            adv_confusion_loss = 0

        return x_recon, adv_confusion_loss


class cyMAE_panelagnostic(nn.Module):
    """ Vision Transformer
    """
    def __init__(self,
                 union_marker_list,
                 marker_embeddings=None,
                 fps=True,
                 subset_size=100,
                 masking_alpha=1.0,
                 is_cumul_masking=True,
                 is_fake_mask=False,
                 is_pred_rank=True,
                 is_cell_cls_loss=True,
                 max_step_k=2,
                 cell_lambda=1.0,
                 is_adv_loss=True,
                 encoder_num_classes=0, 
                 encoder_embed_dim=16, 
                 encoder_depth=6,
                 encoder_num_heads=4, 
                 decoder_num_classes=1, 
                 decoder_embed_dim=8, 
                 decoder_depth=2,
                 decoder_num_heads=4, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0., # learnable gamma in attention, learning influence of self-attention modules to the outputs.
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 device='cuda',
                 seed=0
                 ):
        super().__init__()
        self.union_marker_list = union_marker_list
        self.subset_size = subset_size

        self.data_augmentation = DataAugmentation(subset_size, device, seed, fps)
        self.device = device
        self.rng = np.random.default_rng(seed=seed)

        self.encoder = cyMAEEncoder(
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)

        self.decoder = cyMAEDecoder(
            is_fake_mask=is_fake_mask,
            is_adv_loss=is_adv_loss,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        
        self.masking_alpha = masking_alpha
        self.seen_panel = []
        self.marker_prevalence = torch.zeros(len(union_marker_list))

        # special tokens       
        if marker_embeddings is not None:
            # Marker CLS tokens
            self.marker_embeddings = marker_embeddings.to(device)
            self.marker_embeddings_proj = nn.Linear(self.marker_embeddings.size(1), int(encoder_embed_dim/2), bias=True)
        else:
            self.marker_embeddings = nn.Parameter(torch.zeros(len(union_marker_list), int(encoder_embed_dim/2)))
            trunc_normal_(self.marker_embeddings, std=.02)
            self.marker_embeddings_proj = nn.Identity()


        self.apply(self._init_weights)

        self.is_cumul_masking = is_cumul_masking
        self.is_pred_rank = is_pred_rank
        self.is_fake_mask = is_fake_mask
        self.mcm_loss_fn = nn.MSELoss()
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.is_cell_cls_loss = is_cell_cls_loss
        self.max_step_k = max_step_k
        self.cell_lambda = cell_lambda
        self.is_adv_loss = is_adv_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
    
    def update_marker_prevalence(self, marker_indices):
        if marker_indices in self.seen_panel:
            pass
        else:
            self.seen_panel.append(marker_indices)
            self.marker_prevalence[marker_indices] += 1

    def generate_cumulative_masks(self, x: torch.Tensor, marker_indices: list) -> list[torch.Tensor]:
        """
        Return a list of M boolean mask tensors (B, C, M).
        The list represents cumulative masking of 0, 1, 2, ..., M-1 markers,
        unique per cell (C), and fully vectorized.
        """
        B, C, M = x.shape
        device = x.device
        if self.is_cumul_masking == False:
            sampled_prob = self.rng.random(dtype=np.float32)

            m = torch.from_numpy(self.rng.random(M) < sampled_prob).to(device)
            
            if torch.all(m):
                # 무작위 위치 하나를 0으로 바꾼다
                idx = self.rng.integers(0, M)
                m[idx] = 0
            # 만약 m의 모든 요소가 0이라면 (False)
            elif not torch.any(m):
                # 무작위 위치 하나를 1로 바꾼다
                idx = self.rng.integers(0, M)
                m[idx] = 1
                
            # (1, 1, M) 형태로 브로드캐스팅 가능하게 차원 확장
            mask = m.view(1, 1, M).expand(B,C,M)

            return [mask]
        
        # 1. 확률 계산 (변경 없음)
        preval_batch = self.marker_prevalence.to(device)[torch.tensor(marker_indices, device=device)]
        # prevalence의 로그 값을 계산 (수치 안정성을 위해 작은 값 더하기)
        log_preval = torch.log(preval_batch + 1e-8)

        # masking_alpha를 로그 확률에 직접 곱하여 편향의 방향과 크기를 조절
        # 이 값이 Softmax의 logit으로 사용됨
        scaled_log_probs = log_preval * self.masking_alpha

        # Softmax 함수를 적용하여 최종 확률 분포를 계산
        # Softmax(x)는 exp(x)를 취한 후 정규화하는 것과 동일
        probs_batch = torch.softmax(scaled_log_probs, dim=-1)


        # probs_batch = (1.0 / (preval_batch + 1e-8)) ** self.masking_alpha
        # probs_batch /= probs_batch.sum(dim=-1, keepdim=True)

        # 2. Gumbel-Max를 이용한 순열 생성 (변경 없음)
        expanded_probs = probs_batch.unsqueeze(1).expand(B, C, M)
        gumbel_noise = torch.distributions.gumbel.Gumbel(
            torch.tensor(0., device=device), torch.tensor(1., device=device)
        ).sample((B, C, M))
        perturbed_log_probs = torch.log(expanded_probs) + gumbel_noise
        orders = torch.argsort(perturbed_log_probs, dim=-1, descending=True)

        # 3. 누적 마스크 생성 (수정된 부분)
        one_hot_masks = F.one_hot(orders, num_classes=M)
        # cumsum 결과는 1개 ~ M개 마스킹된 상태를 가짐
        cumulative_masks_tensor = torch.cumsum(one_hot_masks, dim=2)
        
        # 차원을 (M, B, C, M)으로 변경하고 boolean으로 변환
        # 이 텐서는 k=0일 때 1개 마스킹, k=M-1일 때 M개 마스킹 상태를 포함
        permuted_masks = cumulative_masks_tensor.permute(2, 0, 1, 3).to(torch.bool)
        
        # 4. 최종 리스트 구성 (0개 ~ M-1개 마스킹으로 조정)
        # 첫 번째 마스크: 0개가 마스킹된 상태 (모두 False)
        zero_mask = torch.zeros((B, C, M), dtype=torch.bool, device=device)
        
        # 최종 마스크 리스트 생성
        # [0개 마스크] + [1개 마스크, 2개 마스크, ..., M-1개 마스크]
        # permuted_masks에서 마지막 상태(M개 마스킹)는 제외
        masks = [zero_mask] + [permuted_masks[k] for k in range(M - 1)]

        return masks


    def process_subset(self, x, marker_indices,
                       isMCM=False, mask=None, is_pred_rank=False):
        B, C, M = x.shape

        # define marker embedding
        if B == 1:
            marker_embeddings = self.marker_embeddings_proj(
                self.marker_embeddings[marker_indices, :]
            ).expand(B, C, -1, -1)
            other_indices = [
                [j for j in range(len(self.union_marker_list))
                 if j not in marker_indices[i]]
                for i in range(B)
            ]
            fake_marker_embeddings = self.marker_embeddings_proj(
                self.marker_embeddings[other_indices, :]
            ).expand(B, C, -1, -1)

        if not isMCM:
            mask =  torch.zeros(B, C, M, dtype=torch.bool, device=x.device)

        # encoder
        cell_token, x_unmasked_token = self.encoder(
            x, marker_embeddings, mask
        )

        # no masking case
        if not isMCM or mask.sum() == 0:
            return cell_token, None, 0.0

        # decoder input projection
        x_unmasked_token = self.encoder_to_decoder(x_unmasked_token)

        if is_pred_rank:
            x_recon, adv_confusion_loss = self.decoder(
                x_unmasked_token,
                marker_embeddings,
                fake_marker_embeddings,
                mask
            )  # [B, C, M, 2]
            recon, rank = x_recon[:, :, 0].squeeze(-1), x_recon[:, :, 1].squeeze(-1)
            return cell_token, recon, rank, adv_confusion_loss
        else:
            x_recon, adv_confusion_loss = self.decoder(
                x_unmasked_token,
                marker_embeddings,
                fake_marker_embeddings,
                mask
            )  # [B, C, M, 1]
            x_recon = x_recon.squeeze(-1)
            return cell_token, x_recon, adv_confusion_loss
        
    def process_cumulative_subsets(self, x, marker_indices, masks):
        tokens = []
        mcm_loss = adv_loss = 0.0
        for mask in masks:
            # bypass internal mask_generation
            cell_token, x_recon, adv = self.process_subset(
                x, marker_indices, isMCM=True,
                mask=mask,
                is_pred_rank=self.is_pred_rank
            )
            mcm_loss += self.compute_mcm_loss(x_recon, x, mask)
            adv_loss += adv
            tokens.append(cell_token)
        mcm_loss /= len(masks)
        adv_loss  /= len(masks)
        return tokens, mcm_loss, adv_loss


    def compute_mcm_loss(self, x_recon, x, mask):
        if x_recon is None:
            return 0.0
        B,C,_ = x.shape
        loss = self.mcm_loss_fn(x_recon, x[mask].reshape(B, C, -1))
        return loss
    
    def compute_cell_cls_ce_loss(self, cell_tokens1, cell_tokens2, temp=1.0):
        """
        Consistency loss for CLS tokens.
        Args:
            cell_tokens1: Tensor (B, C, d_proj)
            cell_tokens2: Tensor (B, C, d_proj)
        Returns:
            cls_loss (scalar Tensor)
        """


        B, C, d_proj = cell_tokens1.shape

        cell_tokens1 = cell_tokens1.view(B*C, d_proj) / temp
        cell_tokens2 = cell_tokens2.view(B*C, d_proj) / temp

        # Compute the log-probabilities for the scaled cell tokens.
        local_log_prob = F.log_softmax(cell_tokens1, dim=-1)
        
        # Compute the probabilities for the original cell tokens.
        global_prob = F.softmax(cell_tokens2, dim=-1)
        
        # Calculate the cross-entropy loss between the two distributions.
        loss = torch.sum(-global_prob * local_log_prob, dim=-1)
        
        # Return the mean loss over all tokens and batches.
        return loss.mean()

    def compute_progressive_cell_cls_loss(
        self,
        tokens_list,          # length L, each (B,C,d)
        counts,               # length L, #unmasked markers per entry
        max_step_k=2,           # 인접으로 간주할 마커 수 차이
        temp=1.0
    ):
        """
        작은 마커 차이(≤max_step)  ->  작은 거리(또는 큰 유사도)를 유도.
        """
        device  = self.device
        L       = len(tokens_list)
        if L < 2:
            return torch.tensor(0.0, device=device)

        loss, n_pairs = 0.0, 0
        for i in range(L):
            for j in range(i+1, min(L, i+1+max_step_k)):
                if counts[j] - counts[i] > max_step_k:
                    break
                loss += self.compute_cell_cls_ce_loss(tokens_list[i], tokens_list[j], temp=temp)
                n_pairs += 1

        return loss / max(n_pairs, 1)

    def compute_monotonic_cell_cls_loss(
        self,
        tokens_list,          # length L, each (B,C,d)
        counts,               # length L, #unmasked markers per entry
        margin_base=0.05      # 최소 margin (거리 단위)
    ):
        """
        i<j  이면   dist(i,j)  ≥  margin_base * (counts[i]-counts[j])
        (여기서 counts 는 '남은 마커 수' → 차이가 클수록 강한 마진)
        """
        device = self.device
        L      = len(tokens_list)
        if L < 3:
            return torch.tensor(0.0, device=device)

        # flatten
        flats = [ t.reshape(-1, t.size(-1)).to(device) for t in tokens_list ]   # (N,d)
        # pairwise mean Euclidean distance
        dist = torch.zeros((L,L), device=device)
        for i in range(L):
            for j in range(i+1, L):
                dist_ij = F.pairwise_distance(flats[i], flats[j], p=2).mean()
                dist[i,j] = dist[j,i] = dist_ij

        # hinge-rank 위배 항목만 패널티
        loss, n_terms = 0.0, 0
        for i in range(L):
            for j in range(i+1, L):
                desired_margin = margin_base * (counts[i]-counts[j])
                violation = F.relu(desired_margin - dist[i,j])   # 위배량
                loss += violation
                n_terms += 1
        return loss / max(n_terms,1)


    def forward(self, x, marker_indices):
        # x: (B, C, M)
        # marker_indices: list of marker indices list (size=B)

        x = self.data_augmentation(x)
        self.update_marker_prevalence(marker_indices)

        masks = self.generate_cumulative_masks(x, marker_indices)

        tokens, mcm_loss, adv_loss = self.process_cumulative_subsets(
            x, marker_indices, masks
        )

        cell_cls_local_loss = 0.0
        cell_cls_mono_loss = 0.0
        if self.is_cell_cls_loss:
            counts = [int((~m).sum().item()/m.size(1)) for m in masks]  # 마커 ‘남은’ 개수
            cell_cls_local_loss  = self.compute_progressive_cell_cls_loss(tokens, counts, max_step_k=self.max_step_k)
            cell_cls_mono_loss   = self.compute_monotonic_cell_cls_loss(tokens, counts)

        # 최종 손실
        print(mcm_loss, self.cell_lambda * cell_cls_local_loss, self.cell_lambda * cell_cls_mono_loss)
        return mcm_loss + self.cell_lambda * cell_cls_local_loss + self.cell_lambda * cell_cls_mono_loss

    def forward_inference(self, x, marker_indices):
        """
        Inference-time forward: splits cells into chunks, encodes, and returns both
        per-cell embeddings and pooled sample embeddings.
        """
        # x: (B, C_total, M)
        global_views = []
        total_cells = x.shape[1]
        chunk_size = self.subset_size

        # split into subsets
        if total_cells > chunk_size:
            for i in range(0, total_cells, chunk_size):
                subset_cells = x[:, i : i + chunk_size]
                global_views.append(subset_cells)
        else:
            global_views.append(x)

        cell_embeddings = []
        pooled_embeddings = []
        for x_sub in global_views:
            B, C, M = x_sub.shape

            # define marker embedding
            marker_embed = self.marker_embeddings_proj(
                self.marker_embeddings[marker_indices, :]
            ).expand(B, C, -1, -1)  # (B, C, M, emb)

            # no masking
            mask = torch.zeros(B, C, M, dtype=torch.bool, device=self.device)

            # encoder
            cell_token, x_unmasked_token = self.encoder(
                x_sub, marker_embed, mask
            )

            # pooled token
            pooled_token = torch.mean(x_unmasked_token, dim=2)
            cell_embeddings.append(cell_token.squeeze(0))
            pooled_embeddings.append(pooled_token.squeeze(0))

        cell_embeddings = torch.cat(cell_embeddings, dim=0)
        pooled_embeddings = torch.cat(pooled_embeddings, dim=0)
        return cell_embeddings, pooled_embeddings
    
    def forward_all_marker_inference(self, x, marker_indices):
        # x: (B, C_total, M)
        global_views = []
        total_cells = x.shape[1]
        chunk_size = self.subset_size

        # split into subsets
        if total_cells > chunk_size:
            for i in range(0, total_cells, chunk_size):
                subset_cells = x[:, i : i + chunk_size]
                global_views.append(subset_cells)
        else:
            global_views.append(x)


        x_recons = []
        masks = []
        for x_sub in global_views:
            B, C, M = x_sub.shape

            marker_embeddings = self.marker_embeddings_proj(
                self.marker_embeddings[marker_indices, :]
            ).expand(B, C, -1, -1)

            other_indices = [
                [j for j in range(len(self.union_marker_list))
                 if j not in marker_indices[i]]
                for i in range(B)
            ]
            fake_marker_embeddings = self.marker_embeddings_proj(
                self.marker_embeddings[other_indices, :]
            ).expand(B, C, -1, -1)

            # no masking
            mask = torch.zeros(B, C, M, dtype=torch.bool, device=self.device)

            # encoder
            _, x_unmasked_token = self.encoder(
                x_sub, marker_embeddings, mask
            )

            # decoder input projection
            x_unmasked_token = self.encoder_to_decoder(x_unmasked_token)

            x_recon, _ = self.decoder(
                x_unmasked_token,
                marker_embeddings,
                fake_marker_embeddings,
                mask
            )  # [B, C, M, 1]
            print(x_recon.shape)
            x_recon = x_recon.squeeze(-1)

            print(x_recon.shape)
            x_recons.append(x_recon)
            masks.append(mask)
        x_recons = torch.cat(x_recons, dim=1)
        return x_recons


        for x_adjusted, _, _ in global_subsets:
            B, C, M = x.shape
            _, _, x_recon, mask = self.process_subset(x_adjusted, isMCM=True, isRandom=isRandom, mask_ratio=mask_ratio)
            x_recons.append(x_recon)
            masks.append(mask)
        x_recons = torch.cat(x_recons, dim=1)
        masks = torch.cat(masks, dim=1)
        return masks, x_recons



    

    def forward_linear_probing_train(self, x, marker_indices, labels):
        # x: (B, C, M)
        # marker_indices: list of marker indices list (size=B)
        # labels : list (size=B) of list (size=C)
        # self.update_marker_prevalence(marker_indices)
        # self.reset_marker_prevalnace()
        x, labels = self.data_augmentation(x, labels)
        masks = self.generate_cumulative_masks(x, marker_indices)

        loss = 0
        for i,mask in enumerate(masks):
            if len(masks)-i > 10:
                y_pred, _, _ = self.process_subset(
                    x, marker_indices, isMCM=False,
                    mask=mask,
                )

                loss += self.ce_loss_fn(y_pred.view(-1, y_pred.size(2)), labels.view(-1))
                # _, preds = torch.max(y_pred, dim=2)
                # corrects = (preds == labels).sum().item()
                # total = labels.numel()
                # print(corrects / total)
                # break
        # loss /= len(masks)
        loss /= (len(masks)-10)
        # cell_embeddings, y_preds = self.forward_linear_probing_inference(x, marker_indices)
        # y_prob = torch.nn.functional.softmax(y_preds).detach().cpu().numpy()

        # row_sums = y_prob.sum(axis=1, keepdims=True)
        # y_prob = y_prob / row_sums


        # y_logits = y_preds.detach().cpu().numpy()
        # y_pred_idx = np.argmax(y_prob, axis=1)   # 예측 인덱스

        # # 실제 레이블 리스트 (strings)
        # y_true_idx = labels[0].cpu().numpy()     # e.g. ['B cell', 'T cell CD4 Naive', ...]
        # # --- 2) 레이블 인코딩 (mapping 이용)

        # # 문자열 레이블 → 정수 인덱스
        # # --- 3) 평가지표 계산
        # from sklearn.metrics import balanced_accuracy_score
        # bal_acc = balanced_accuracy_score(y_true_idx, y_pred_idx)
        # print("bal_Acc", bal_acc)
        return loss
    

    def forward_linear_probing_inference(self, x, marker_indices):
        """
        Inference-time forward: splits cells into chunks, encodes, and returns class prediction
        """
        # x: (B, C_total, M)
        global_views = []
        total_cells = x.shape[1]
        chunk_size = self.subset_size

        # split into subsets
        if total_cells > chunk_size:
            for i in range(0, total_cells, chunk_size):
                subset_cells = x[:, i : i + chunk_size]
                global_views.append(subset_cells)
        else:
            global_views.append(x)

        cell_embeddings = []
        y_preds = []
        for x_sub in global_views:
            B, C, M = x_sub.shape

            # define marker embedding
            marker_embed = self.marker_embeddings_proj(
                self.marker_embeddings[marker_indices, :]
            ).expand(B, C, -1, -1)  # (B, C, M, emb)

            # no masking
            mask = torch.zeros(B, C, M, dtype=torch.bool, device=x.device)

            # encoder
            y_pred, embeddings = self.encoder(
                x_sub, marker_embed, mask
            )
            cell_embeddings.append(embeddings[:,:,0,:].squeeze(0))
            y_preds.append(y_pred.squeeze(0))
        cell_embeddings = torch.cat(cell_embeddings, dim=0)
        y_preds = torch.cat(y_preds, dim=0)

        return cell_embeddings, y_preds



    
    # def forward_mask_inference_for_plot(self, x, isRandom):
    #     assert x.shape[1] < self.global_subset_size
    #     _, _, x_recon, mask = self.process_subset(x, isMCM=True, isRandom=isRandom)

    #     return mask, x_recon


    # def forward_de_novo_inference(self, x, n_de_novo):
    #     B, C, M = x.shape

    #     # define marker embedding
    #     marker_embeddings = self.marker_embeddings_proj(self.marker_embeddings).expand(B, C, -1, -1) # (B, C, M+n_de_novo, emb)

    #     # mask generation
    #     mask = self.mask_generation(x.shape, isMCM=False, isRandom=False)
    #     mask[:,:,-n_de_novo:] = True


    #     # encoder : filter mask and add marker emb, pos emb
    #     _, _, x_unmasked_token = self.encoder(x, marker_embeddings, mask)
        
    #     # decoder : add mask, marker emb/pos emb
    #     x_unmasked_token = self.encoder_to_decoder(x_unmasked_token)
    #     x_recon = self.decoder(x_unmasked_token, marker_embeddings, mask) # [B, C, M+n_de_novo, 1]
    #     return mask, x_recon
