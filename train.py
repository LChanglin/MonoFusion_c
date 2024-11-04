from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import datetime
import torch.distributed as dist

import dataloader.dataset as datasets
from model.monofusion import MonoFusion
from utils.trainer import TTCTrainer

time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
out_dir = "./log/%s_selfcon_ttc"%(time_stamp)

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                    help='where to save the training log and models')
parser.add_argument('--stage', default='chairs', type=str,
                    help='training stage on different datasets')
parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                    help='validation datasets')
parser.add_argument('--max_flow', default=400, type=int,
                    help='exclude very large motions during training')
parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                    help='image size for training')
parser.add_argument('--padding_factor', default=16, type=int,
                    help='the input should be divisible by padding_factor, otherwise do padding or resizing')

# evaluation
parser.add_argument('--eval', action='store_true',
                    help='evaluation after training done')
parser.add_argument('--save_eval_to_file', action='store_true')
parser.add_argument('--evaluate_matched_unmatched', action='store_true')
parser.add_argument('--val_things_clean_only', action='store_true')
parser.add_argument('--with_speed_metric', action='store_true',
                    help='with speed methic when evaluation')

# training
parser.add_argument('--load_flow_param', action='store_true')
parser.add_argument('--epoch', default=20, type=int)
parser.add_argument('--lr', default=4e-4, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--grad_clip', default=1.0, type=float)
parser.add_argument('--num_steps', default=100000, type=int)
parser.add_argument('--seed', default=326, type=int)
parser.add_argument('--summary_freq', default=100, type=int)
parser.add_argument('--val_freq', default=10000, type=int)
parser.add_argument('--save_ckpt_freq', default=10000, type=int)
parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

# resume pretrained model or resume training
parser.add_argument('--resume', default=None, type=str,
                    help='resume from pretrained model or resume from unexpectedly terminated training')
parser.add_argument('--save', default=None, type=str,
                    help='resume from pretrained model or resume from unexpectedly terminated training')
parser.add_argument('--strict_resume', action='store_true',
                    help='strict resume while loading pretrained weights')
parser.add_argument('--no_resume_optimizer', action='store_true')

# model: learnable parameters
parser.add_argument('--num_scales', default=1, type=int,
                    help='feature scales: 1/8 or 1/8 + 1/4')
parser.add_argument('--num_head', default=1, type=int)
parser.add_argument('--feature_channels', default=128, type=int)
parser.add_argument('--upsample_factor', default=8, type=int)
parser.add_argument('-- ', default=1, type=int)
parser.add_argument('--ffn_dim_expansion', default=4, type=int)
parser.add_argument('--num_transformer_layers', default=6, type=int)
parser.add_argument('--reg_refine', action='store_true',
                    help='optional task-specific local regression refinement')
parser.add_argument('--parallel', action='store_true',
                    help='optional task-specific local regression refinement')
parser.add_argument('--load_opt', action='store_true',
                    help='optional task-specific local regression refinement')

# model: parameter-free
parser.add_argument('--attn_type', default='swin', type=str,
                    help='attention function')
parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                    help='number of splits in attention')
parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                    help='correlation radius for matching, -1 indicates global matching')
parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                    help='self-attention radius for propagation, -1 indicates global attention')
parser.add_argument('--num_reg_refine', default=1, type=int,
                    help='number of additional local regression refinement')

# loss
parser.add_argument('--gamma', default=0.9, type=float,
                    help='exponential weighting')

# predict on sintel and kitti test set for submission
parser.add_argument('--kittidataset', default='/mnt/pool2/lcl/data/data_scene_flow/training/', type=str)
parser.add_argument('--drivingdataset', default='/mnt/pool2/lcl/data/Driving/', type=str)
parser.add_argument('--submission', action='store_true',
                    help='submission to sintel or kitti test sets')
parser.add_argument('--output_path', default='output', type=str,
                    help='where to save the prediction results')
parser.add_argument('--save_vis_flow', action='store_true',
                    help='visualize flow prediction as .png image')
parser.add_argument('--no_save_flo', action='store_true',
                    help='not save flow as .flo if only visualization is needed')

# inference on images or videos
parser.add_argument('--inference_dir', default=None, type=str)
parser.add_argument('--inference_video', default=None, type=str)
parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                    help='can specify the inference size for the input to the network')
parser.add_argument('--save_flo_flow', action='store_true')
parser.add_argument('--pred_bidir_flow', action='store_true',
                    help='predict bidirectional flow')
parser.add_argument('--pred_bwd_flow', action='store_true',
                    help='predict backward flow only')
parser.add_argument('--fwd_bwd_check', action='store_true',
                    help='forward backward consistency check with bidirection flow')
parser.add_argument('--save_video', action='store_true')
parser.add_argument('--concat_flow_img', action='store_true')

# distributed training
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

# misc
parser.add_argument('--count_time', action='store_true',
                    help='measure the inference time')

parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

if args.parallel:
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])# + 1
    # if local_rank>=1:
    #     local_rank += 1
    print(local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    parallel = True
else:
    torch.cuda.set_device(2)
    device = torch.device("cuda", 2)

    parallel = False

def main():

    print(device)
    model = MonoFusion( num_scales=args.num_scales,
                    feature_channels=args.feature_channels,
                    upsample_factor=args.upsample_factor,
                    num_head=args.num_head,
                    ffn_dim_expansion=args.ffn_dim_expansion,
                    num_transformer_layers=args.num_transformer_layers,
                    reg_refine=args.reg_refine,
                    train=True).cuda()
    
    max_lr = args.lr
    ini_lr = max_lr / 25
    min_lr = ini_lr / 1e4

    optimizer = torch.optim.AdamW([{"params":model.parameters(), "max_lr":max_lr, "initial_lr":ini_lr, "min_lr":min_lr}],\
                                     lr=max_lr, weight_decay=args.weight_decay)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=args.weight_decay)

    model_loaded = None
    epoch = 0

    if not args.load_flow_param:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.load_opt:
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'net' in checkpoint:
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    else:
        from utils.unimatch.unimatch import UniMatch
        model_loaded = UniMatch(num_scales=args.num_scales,
                                feature_channels=args.feature_channels,
                                upsample_factor=args.upsample_factor,
                                num_head=args.num_head,
                                ffn_dim_expansion=args.ffn_dim_expansion,
                                num_transformer_layers=args.num_transformer_layers,
                                reg_refine=True,
                                task='flow')

        checkpoint = torch.load(args.resume, map_location='cpu')
        model_loaded.load_state_dict(checkpoint['model'], strict=args.strict_resume)
        
        

        model.cnet.load_state_dict(model_loaded.backbone.state_dict())

        
        for key in model.state_dict().keys():
            if 'featnet' in key:
                model.featnet.transformer.load_state_dict(model_loaded.transformer.state_dict())
                if 'flownet' in key:
                    model.flownet.feature_flow_attn.load_state_dict(model_loaded.feature_flow_attn.state_dict())
                else:
                    print('aaa')
                    model.scalenet.decoder.feature_flow_attn.load_state_dict(model_loaded.feature_flow_attn.state_dict())
                    model.corrnet.feature_flow_attn.load_state_dict(model_loaded.feature_flow_attn.state_dict())
                break 
        # for key in model.state_dict().keys():
        #     if 'upsampler' in key:
        #         print('bbb')
        #         model.corrnet.upsampler.load_state_dict(model_loaded.upsampler.state_dict())
        #         break



        for key in model.scalenet.decoder.state_dict().keys():
            if 'refine' in key:
                model.scalenet.refine.load_state_dict(model_loaded.refine.state_dict())
                break
        print('corr model loaded')
        # for key in model.scalenet.encoder.state_dict().keys():
        #     if 'feature_scale_attn' in key:
        #         model.scalenet.encoder.feature_scale_attn.load_state_dict(model.flownet.feature_flow_attn.state_dict())
        #         break

    if parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], \
                        output_device=local_rank, find_unused_parameters=True)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], \
        #                 output_device=local_rank)
        if torch.distributed.get_rank()==0:
                # torch.distributed.barrier()
                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)
                    loss_txt = out_dir + '/0.txt'
                    file = open(loss_txt,'w')
                    file.close()
        # else:
        #     torch.distributed.barrier()
    else:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir) 
            loss_txt = out_dir + '/0.txt'
            file = open(loss_txt,'w')
            file.close()

    # num_params = sum(p.numel() for p in model.parameters())
    # print('Number of params:', num_params)

    # num_params = sum(p.numel() for p in model_loaded.parameters())
    # print('Number of ori params:', num_params)

    #dataset_kitti = ScaleDataset(args.kittidataset, shape = [384, 512], dataset='KITTI')
    start = time.time()
    print('Start Loading ...')
    dataset = datasets.fetch_dataloader(args)
    
    print('Done ', time.time()-start)
    
    # dataset = torch.utils.data.ConcatDataset([dataset_driving]*1 + [dataset_kitti]*100)
    print("Learning rate: ", optimizer.state_dict()['param_groups'][0]['lr'])

    trainer = TTCTrainer(model=model, dataset=dataset, optimizer=optimizer, args=args, 
                        start_epoch=epoch, device=device, model_path=args.resume, save_path=args.save, 
                        test_model=model_loaded, parallel=parallel, time_stamp=time_stamp, max_lr=max_lr, crop_size=args.image_size)

    trainer.train()


if __name__ == "__main__":
    main()