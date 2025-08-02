
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path

from optim_factory import create_optimizer
from datasets import CyTOFDataset
from engine_for_pretraining import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from model import cyMAE_panelagnostic

torch.set_num_threads(1)

def get_args():
    parser = argparse.ArgumentParser('cyMAE_panelagnostic pre-training script', add_help=False)
    parser.add_argument('--exp', default='exp1', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    parser.add_argument('--d_model', default=16, type=int)

    parser.add_argument('--union_marker_list', nargs='+', default=["CD3","CD4","CD8a","CD11c","CD14","CD16","CD19","CD20","CD25","CD27","CD28","CD38","CD45","CD56","CD57","CD66b","CD123","CD127","CD161","CD183","CD185","CD194","CD196","CD197","CD294","CD45RA","CD45RO","HLA-DR","IgD","TCRgd"], type=str,
                        help='Union marker list over all panels')
    parser.add_argument('--subset_size', default=100, type=int,
                        help='cell subset size for sampling cells')

    parser.add_argument('--noise_std', default=0.0, type=float,
                        help='std of noise during training')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--fps', action='store_true',
                        help='Use fps')
    parser.add_argument('--no_fps', action='store_false', dest='fps',
                        help='No use fps')
    parser.set_defaults(fps=True)

    parser.add_argument('--is_cumul_masking', action='store_true',
                        help='Use cumulative masking strategy')
    parser.add_argument('--no_cumul_masking', action='store_false', dest='is_cumul_masking',
                        help='No use cumulative masking strategy')
    parser.set_defaults(is_cumul_masking=True)

    parser.add_argument('--esm_embedding', action='store_true',
                        help='Use esm embeddings for initial protein embedding')
    parser.add_argument('--no_esm_embedding', action='store_false', dest='esm_embedding',
                        help='No use esm embeddings')
    parser.set_defaults(esm_embedding=False)

    parser.add_argument('--is_fake_mask', action='store_true',
                        help='Use fake mask token for decoder input')
    parser.add_argument('--no_fake_mask', action='store_false', dest='is_fake_mask',
                        help='No use ake mask token for decoder input')
    parser.set_defaults(is_fake_mask=True)
                        
    parser.add_argument('--is_pred_rank', action='store_true',
                        help='Use cell cls loss')
    parser.add_argument('--no_pred_rank', action='store_false', dest='is_pred_rank',
                        help='No use cell cls loss')
    parser.set_defaults(is_pred_rank=True)

    parser.add_argument('--is_cell_cls_loss', action='store_true',
                        help='Use cell cls loss')
    parser.add_argument('--no_cell_cls_loss', action='store_false', dest='is_cell_cls_loss',
                        help='No use cell cls loss')
    parser.set_defaults(is_cell_cls_loss=True)

    parser.add_argument('--is_adv_loss', action='store_true',
                        help='Use confusion loss')
    parser.add_argument('--no_adv_loss', action='store_false', dest='is_adv_loss',
                        help='No use confusion loss')
    parser.set_defaults(is_adv_loss=True)


    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_const', dest='mode',
                       const='train', help='Run in training mode')
    # group.add_argument('--inference', action='store_const', dest='mode',
    #                    const='inference', help='Run in inference mode')
    group.add_argument('--linear_probing_train', action='store_const', dest='mode',
                       const='linear_probing_train', help='Run in linear probing training mode')
    # group.add_argument('--linear_probing_inference', action='store_const', dest='mode',
    #                    const='linear_probing_inference', help='Run in linear probing inference mode')
    parser.set_defaults(mode='train')


    
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 1.5e-3)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    
    parser.add_argument('--masking_alpha', type=float, default=1.0,
                        help='alpha for masking prob bias')
    parser.add_argument('--cell_lambda', type=float, default=1.0,
                        help='loss weight for cell cls loss')
    parser.add_argument('--max_step_k', type=int, default=0,
                        help='max step k for cell cls loss')

    # Dataset parameters
    parser.add_argument('--data_path', default='/project/kimgroup_immune_health/data/pan_panel/simulation/dev/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./ckpts/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs/',
                        help='path where to tensorboard log')
    
    parser.add_argument('--ckpt', default='',
                        help='ckpt path for inference')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # linear probing params
    parser.add_argument('--linear_probing_ckpt', default='./ckpts/dmodel_16_no_pred_rank_no_adv_loss/cyMAE_panelagnostic_maskingalpha_0.0_celllambda_0.0_lr_0.005_checkpoint-5000.pth',
                        help='ckpt path for linear probing training')
    parser.add_argument('--num_classes', default=46, type=int,
                        help='num of classes for linear probing')

    # parser.add_argument("--local_rank", type=int, default=0, help="local rank for DistributedDataParallel")

    return parser.parse_args()

def get_model(args):
    if args.esm_embedding:
        marker_embeddings = torch.stack([torch.load(f'./esm_embeddings/{marker}.pt') for marker in args.union_marker_list])
        marker_embeddings.requires_grad_(False)
    else:
        marker_embeddings = None
    model = cyMAE_panelagnostic(
        union_marker_list=args.union_marker_list,
        marker_embeddings=marker_embeddings,
        fps=args.fps,
        subset_size=args.subset_size, 
        masking_alpha=args.masking_alpha,
        is_cumul_masking=args.is_cumul_masking,
        is_pred_rank=args.is_pred_rank,
        is_cell_cls_loss=args.is_cell_cls_loss,
        max_step_k=args.max_step_k,
        cell_lambda=args.cell_lambda,
        is_adv_loss=args.is_adv_loss,
        encoder_embed_dim=args.d_model,
        encoder_depth=6,
        encoder_num_heads=4,
        encoder_num_classes=0 if args.mode == 'train' else args.num_classes,
        decoder_num_classes=2 if args.is_pred_rank else 1, 
        decoder_embed_dim=int(args.d_model/2), 
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        device=args.device,
        seed=args.seed
    )
    return model

def main(args):
    # 분산 환경 초기화 (train 모드일 때만 사용하도록 분기)
    if args.mode == 'train' or args.mode == 'linear_probing_train':
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:  # inference 모드에서는 단일 GPU 사용 (기본적으로 GPU 0)
        device = torch.device("cuda:0")
        local_rank = 0  # inference 시에는 분산 환경이 아니므로 0으로 고정

    args.device = device  # 모델 생성 시 올바른 device 전달

    # rank 0에서 출력 디렉토리 생성 (train 모드에서 분산 환경일 경우만)
    if args.mode == 'train' and args.output_dir and torch.distributed.get_rank() == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 시드 고정 (inference 모드에서는 단일 GPU이므로 local_rank 영향이 없음)
    seed = args.seed + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    print(args)

    if args.mode == 'train':
        # 데이터셋 로드
        dataset_train = CyTOFDataset(args.union_marker_list, args.data_path, args.seed)
        args.pretraining_samples = dataset_train.filenames
        print(dataset_train.filenames)

        # 분산 샘플러 적용
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            worker_init_fn=utils.seed_worker,
            collate_fn=dataset_train.custom_collate_fn
        )

        # 모델 생성 및 DDP 래핑
        model = get_model(args)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        n_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        print('number of params: {} M'.format(n_parameters / 1e6))
        
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size

        total_batch_size = args.batch_size * torch.distributed.get_world_size()
        print("LR = %.8f" % args.lr)
        print("Batch size (per GPU) = %d" % args.batch_size)
        print("Total batch size = %d" % total_batch_size)
        print("Number of training steps = %d" % num_training_steps_per_epoch)
        print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

        optimizer = create_optimizer(args, model.module, filter_bias_and_bn=True)
        loss_scaler = NativeScaler()

        print("Use step level LR & WD scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        print("Max LR = %.7f, Min LR = %.7f" % (max(lr_schedule_values), min(lr_schedule_values)))
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

        utils.auto_load_model(args=args, model=model.module, optimizer=optimizer, loss_scaler=loss_scaler)
        if args.esm_embedding:
            model_name = f"cyMAE_panelagnostic_maskingalpha_{args.masking_alpha}_maxstep_{args.max_step_k}_celllambda_{args.cell_lambda}_lr_{args.lr}_esm"
        else:
            model_name = f"cyMAE_panelagnostic_maskingalpha_{args.masking_alpha}_maxstep_{args.max_step_k}_celllambda_{args.cell_lambda}_lr_{args.lr}"
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)  # epoch마다 샘플러 시드 변경
            train_stats = train_one_epoch(
                args, model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, 
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
            )
            # rank 0에서만 체크포인트 저장 및 로깅 수행
            if torch.distributed.get_rank() == 0:
                base_dir_name = f'{args.exp}_dmodel_{args.d_model}_subset_size_{args.subset_size}'
                if args.fps:
                    base_dir_name = base_dir_name + '_fps'

                if not args.is_cumul_masking:
                    base_dir_name = base_dir_name + '_no_cumul_masking'

                if not args.is_pred_rank:
                    base_dir_name = base_dir_name + '_no_pred_rank'
                
                if not args.is_cell_cls_loss:
                    base_dir_name = base_dir_name + '_no_cell_cls'

                if not args.is_adv_loss:
                    base_dir_name = base_dir_name + '_no_adv_loss'

                if not args.is_fake_mask:
                    base_dir_name = base_dir_name + '_no_fake_mask'

                output_dir = Path(args.output_dir) / f'{base_dir_name}'
                os.makedirs(output_dir, exist_ok=True)

                if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                    utils.save_model(
                        args=args, output_dir=output_dir, model_name=model_name, model=model.module, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch, 'n_parameters': n_parameters}
                with open(os.path.join(output_dir, model_name+'_log.txt'), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if torch.distributed.get_rank() == 0:
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))

    # elif args.mode == 'inference':
    #     # inference 모드에서는 단일 GPU 사용
    #     # (분산 관련 초기화와 분산 샘플러를 사용하지 않고, 기본적으로 GPU:0에 모델을 올림)
    #     args.cohorts = ['AALC', 'DORA', 'HSIH', 'IHCV', 'ISPY', 'LRAD', 'MESSI', 'MS', 'PREPRO', 'Sarcoidosis']
    #     args.num_samples_per_cohort = 1000
    #     checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    #     ckpt_args = checkpoint['args']
    #     print("Loaded ckpt args:")
    #     print(ckpt_args)

    #     model = get_model(ckpt_args)
    #     model.load_state_dict(checkpoint['model'])
    #     model.to(device)
    #     # DDP 제거 (단일 GPU 사용이므로 래핑하지 않음)

    #     dataset_test = CyTOFDataset(args.union_marker_list, args.data_path, args.cohorts, args.num_samples_per_cohort, is_perm=False, seed=args.seed)
    #     args.batch_size = 1

    #     # 분산 샘플러 대신 SequentialSampler 사용 (또는 sampler 인자 생략 가능)
    #     test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    #     data_loader_test = torch.utils.data.DataLoader(
    #         dataset_test,
    #         batch_size=args.batch_size,
    #         sampler=test_sampler,
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=False,
    #         worker_init_fn=utils.seed_worker,
    #         collate_fn=dataset_test.custom_collate_fn
    #     )

    #     original_data, cell_embeddings, pooled_embeddings, panel_labels, cell_labels, _ = inference_one_epoch(args, model, data_loader_test, device)
        
    #     # inference 결과 저장 (단일 GPU 환경이므로 항상 저장 실행)
    #     train_output_dir = Path(args.output_dir, 'train')
    #     train_output_dir.mkdir(parents=True, exist_ok=True)
    #     test_output_dir = Path(args.output_dir, 'test')
    #     test_output_dir.mkdir(parents=True, exist_ok=True)
    #     inference_filenames = dataset_test.filenames
    #     for filename, c_ori, c_emb, c_pool_emb, panel, c_label in zip(inference_filenames, original_data, cell_embeddings, pooled_embeddings, panel_labels, cell_labels):
    #         if filename in ckpt_args.pretraining_samples:
    #             output_path = train_output_dir / f'{filename}.pt'
    #         else:
    #             output_path = test_output_dir / f'{filename}.pt'
    #         output_dict = {
    #             'original': c_ori,
    #             'cell_embeddings': c_emb,
    #             'pooled_embeddings': c_pool_emb,
    #             'cell_type': c_label,
    #             'panel': panel
    #         }
    #         torch.save(output_dict, output_path)

    elif args.mode == 'linear_probing_train':
        # linear probing 모드도 동일하게 분산 환경 적용
        checkpoint = torch.load(args.linear_probing_ckpt, map_location='cpu', weights_only=False)
        ckpt_args = checkpoint['args']
        print("Loaded ckpt args:")
        print(ckpt_args)

        model = get_model(ckpt_args)
        model.load_state_dict(checkpoint['model'])
        for param in model.parameters():
            param.requires_grad = False

        model.encoder.reset_classifier(args.num_classes)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        model_name = args.linear_probing_ckpt.split('/')[-1][:-4] + "_linear_probing"

        ckpt_args.lr = args.lr

        ckpt_args.data_path = args.data_path
        ckpt_args.output_dir = args.output_dir
        ckpt_args.num_classes = args.num_classes
        ckpt_args.epochs = args.epochs
        ckpt_args.batch_size = args.batch_size
        ckpt_args.save_ckpt_freq = args.save_ckpt_freq
        ckpt_args.mode = args.mode
        args = ckpt_args

        dataset_train = CyTOFDataset(args.union_marker_list, args.data_path, args.seed)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=False)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=utils.seed_worker,
            collate_fn=dataset_train.custom_collate_fn
        )

        n_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        print('number of params: {} M'.format(n_parameters / 1e6))

        total_batch_size = args.batch_size
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size
        print("LR = %.8f" % args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Number of training steps = %d" % num_training_steps_per_epoch)
        print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

        optimizer = create_optimizer(args, model.module, filter_bias_and_bn=True)
        loss_scaler = NativeScaler()
        print("Use step level LR & WD scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        print("Max LR = %.7f, Min LR = %.7f" % (max(lr_schedule_values), min(lr_schedule_values)))
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))



        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                args, model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
            )

            if torch.distributed.get_rank() == 0:
                base_dir_name = f'{args.exp}_classifier_dmodel_{args.d_model}_subset_size_{args.subset_size}'
                if args.fps:
                    base_dir_name = base_dir_name + '_fps'

                if not args.is_pred_rank:
                    base_dir_name = base_dir_name + '_no_pred_rank'
                
                if not args.is_cell_cls_loss:
                    base_dir_name = base_dir_name + '_no_cell_cls'

                if not args.is_adv_loss:
                    base_dir_name = base_dir_name + '_no_adv_loss'

                if not args.is_fake_mask:
                    base_dir_name = base_dir_name + '_no_fake_mask'
                
                output_dir = Path(args.output_dir) / f'{base_dir_name}'
                os.makedirs(output_dir, exist_ok=True)

                if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                    utils.save_model(
                        args=args, output_dir=output_dir, model_name=model_name, model=model.module, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch)
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch, 'n_parameters': n_parameters}
                with open(os.path.join(output_dir, model_name+'_log.txt'), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
            
        if torch.distributed.get_rank() == 0:
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))

    # elif args.mode == 'linear_probing_inference':
    #     # inference 모드에서는 단일 GPU 사용
    #     # (분산 관련 초기화와 분산 샘플러를 사용하지 않고, 기본적으로 GPU:0에 모델을 올림)
    #     args.cohorts = ['AALC', 'DORA', 'HSIH', 'IHCV', 'ISPY', 'LRAD', 'MESSI', 'MS', 'PREPRO', 'Sarcoidosis']
    #     args.num_samples_per_cohort = 1000
    #     checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    #     ckpt_args = checkpoint['args']
    #     print("Loaded ckpt args:")
    #     print(ckpt_args)

    #     model = get_model(ckpt_args)
    #     model.load_state_dict(checkpoint['model'])
    #     model.to(device)
    #     # DDP 제거 (단일 GPU 사용이므로 래핑하지 않음)

    #     dataset_test = CyTOFDataset(args.union_marker_list, args.data_path, args.cohorts, args.num_samples_per_cohort, args.seed)
    #     args.batch_size = 1

    #     # 분산 샘플러 대신 SequentialSampler 사용 (또는 sampler 인자 생략 가능)
    #     test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    #     data_loader_test = torch.utils.data.DataLoader(
    #         dataset_test,
    #         batch_size=args.batch_size,
    #         sampler=test_sampler,
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=False,
    #         worker_init_fn=utils.seed_worker,
    #         collate_fn=dataset_test.custom_collate_fn
    #     )

    #     cell_embeddings, pooled_embeddings, cell_labels = inference_one_epoch(args, model, data_loader_test, device)
        
    #     # inference 결과 저장 (단일 GPU 환경이므로 항상 저장 실행)
    #     train_output_dir = Path(args.output_dir, 'train')
    #     train_output_dir.mkdir(parents=True, exist_ok=True)
    #     test_output_dir = Path(args.output_dir, 'test')
    #     test_output_dir.mkdir(parents=True, exist_ok=True)
    #     inference_filenames = dataset_test.filenames
    #     for filename, c_emb, c_pool_emb, c_label in zip(inference_filenames, cell_embeddings, pooled_embeddings, cell_labels):
    #         if filename in ckpt_args.pretraining_samples:
    #             output_path = train_output_dir / f'{filename}.pt'
    #         else:
    #             output_path = test_output_dir / f'{filename}.pt'
    #         output_dict = {
    #             'cell_embeddings': c_emb,
    #             'pooled_embeddings': c_pool_emb,
    #             'cell_type': c_label
    #         }
    #         torch.save(output_dict, output_path)

if __name__ == '__main__':
    opts = get_args()
    main(opts)