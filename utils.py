# --------------------------------------------------------
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import io
import os
import math
import time
import json
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path

import torch
import torch.distributed as dist
import random

_DIST_AVAILABLE = dist.is_available() and dist.is_initialized()

def is_dist_avail_and_initialized():
    """분산 학습이 가능하고 초기화되었는지 확인"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        # NCCL 백엔드의 경우, 초기화되지 않았어도 get_rank()가 오류를 발생시킬 수 있음
        # 이를 방지하기 위해 환경 변수 확인
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            return True
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        분산 환경에서 모든 프로세스의 값을 동기화합니다.
        `all_reduce`를 사용하여 모든 GPU의 total과 count를 합산합니다.
        """
        if not is_dist_avail_and_initialized():
            return
        
        # device='cuda'는 DDP 환경에서 GPU를 사용한다고 가정합니다.
        # CPU only 분산 학습 시에는 device를 적절히 변경해야 합니다.
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        자신이 관리하는 모든 meter에 대해 동기화를 수행합니다.
        """
        if not is_dist_avail_and_initialized():
            return
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
     if isinstance(parameters, torch.Tensor):
         parameters = [parameters]
     parameters = [p for p in parameters if p.grad is not None]
     norm_type = float(norm_type)
     if len(parameters) == 0:
         return torch.tensor(0.)
     device = parameters[0].grad.device
     if math.isinf(norm_type):
         total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
     else:
         total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
     return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, output_dir, model_name, epoch, model, optimizer, loss_scaler, model_ema=None):
    epoch_name = str(epoch+1) if isinstance(epoch, int) else epoch
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('%s_checkpoint-%s.pth' % (model_name, epoch_name))]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            torch.save(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag='%s_checkpoint-%s.pth' % (model_name, epoch_name), client_state=client_state)


def auto_load_model(args, model, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        print("No loss_scaler, no resume")



def confusion(preds, targets):
    from sklearn.metrics import confusion_matrix
    preds = torch.max(preds,1)[1]
    return confusion_matrix(targets.view(-1).numpy(), preds.view(-1).numpy())

def baccuracy(preds, targets):
    from sklearn.metrics import balanced_accuracy_score
    preds = torch.max(preds,1)[1]
    return balanced_accuracy_score(targets.view(-1).numpy(), preds.view(-1).numpy())

def ARI(preds, targets):
    from sklearn.metrics import adjusted_rand_score
    preds = torch.max(preds,1)[1]
    return adjusted_rand_score(targets.view(-1).numpy(), preds.view(-1).numpy())

def NMI(preds, targets):
    from sklearn.metrics import normalized_mutual_info_score
    preds = torch.max(preds,1)[1]
    return normalized_mutual_info_score(targets.view(-1).numpy(), preds.view(-1).numpy())


def create_ds_config(args):
    pass
#     args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
#     with open(args.deepspeed_config, mode="w") as writer:
#         ds_config = {
#             "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
#             "train_micro_batch_size_per_gpu": args.batch_size,
#             "steps_per_print": 1000,
#             "optimizer": {
#                 "type": "Adam",
#                 "adam_w_mode": True,
#                 "params": {
#                     "lr": args.lr,
#                     "weight_decay": args.weight_decay,
#                     "bias_correction": True,
#                     "betas": [
#                         0.9,
#                         0.999
#                     ],
#                     "eps": 1e-8
#                 }
#             },
#             "fp16": {
#                 "enabled": True,
#                 "loss_scale": 0,
#                 "initial_scale_power": 7,
#                 "loss_scale_window": 128
#             }
#         }

#         writer.write(json.dumps(ds_config, indent=2))


def torch_interp(x, xp, fp):
    """
    1D linear interpolation function.
    
    Args:
        x: Query tensor (1D) where interpolation is performed.
        xp: 1D tensor containing the x-coordinates of the data points (must be in increasing order).
        fp: 1D tensor containing the corresponding y-coordinates for xp.
    
    Returns:
        A tensor of interpolated values at positions x.
    """
    # Find the indices in xp for each value in x.
    inds = torch.searchsorted(xp, x, right=False)
    inds = inds.clamp(1, xp.numel() - 1)
    
    x0 = xp[inds - 1]
    x1 = xp[inds]
    y0 = fp[inds - 1]
    y1 = fp[inds]
    
    # Compute the linear interpolation: y = y0 + slope * (x - x0)
    slope = (y1 - y0) / (x1 - x0)
    y = y0 + slope * (x - x0)
    
    # Handle the boundary: if x equals the last element in xp.
    y = torch.where(x == xp[-1], fp[-1].expand_as(y), y)
    return y