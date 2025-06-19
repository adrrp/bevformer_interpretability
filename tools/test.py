# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os
import argparse
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

#from mmdet3d.apis import single_gpu_test
from projects.mmdet3d_plugin.apis.test import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.apis.test import multi_gpu_test
from projects.mmdet3d_plugin.metrics import IntersectionOverUnion
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np 
import mmcv
import cv2
from nuscenes import NuScenes
import types
import matplotlib.pyplot as plt
from captum.attr import Saliency, IntegratedGradients
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--save-raw-bev',
        help='directory where to save raw BEV features extracted from encoder as .npy files')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES'] 
        #print("CLASSES: ", checkpoint['meta']['CLASSES']) # Print: CLASSES:  ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    print("PRINT MODEL:")
    print(model)

    #x defecto
    """
    if not distributed:
        print("USING SINGLE GPU TEST")
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        print("USING DISTRIBUTED TEST")
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        #outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
        #Gpu Collect disabled by default
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    """

    """#Bev feature extraction
    map_enable = True
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        outputs = []
        dataset = data_loader.dataset
        
        bev_feats = []
        def grab_bev(module, inp, outp):
            bev_feats.append(outp.detach().cpu())
        enc = model.module.pts_bbox_head.transformer.encoder
        enc.register_forward_hook(grab_bev)

        if map_enable:
            num_map_class = 4
            semantic_map_iou_val = IntersectionOverUnion(num_map_class).cuda()

        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):

            with torch.no_grad():
                in_data = {i: j for i, j in data.items() if 'img' in i}
                result = model(return_loss=False, rescale=True, **in_data)

            batch_size = len(result)
            
            
            if result[0]['pts_bbox'] != None:
                outputs.extend([dict(pts_bbox=result[0]['pts_bbox'])])

            if result[0]['seg_preds'] is not None:
                pred = result[0]['seg_preds']
                #Hacemos one hot encoding
                max_idx = torch.argmax(pred, dim=1, keepdim=True)
                one_hot = pred.new_full(pred.shape, 0)
                one_hot.scatter_(1, max_idx, 1)
                
                num_cls = pred.shape[1]
                indices = torch.arange(0, num_cls).reshape(-1, 1, 1).to(pred.device)
                pred_semantic_indices = torch.sum(one_hot * indices, axis=1).int()
                target_semantic_indices = data['semantic_indices'][0].cuda()
                semantic_map_iou_val(pred_semantic_indices, target_semantic_indices)

            for _ in range(batch_size):
                prog_bar.update()



        if map_enable:
            import prettytable as pt
            scores = semantic_map_iou_val.compute()
            mIoU = sum(scores[1:]) / (len(scores) - 1)
            tb = pt.PrettyTable()
            tb.field_names = ['Validation num', 'Divider', 'Pred Crossing', 'Boundary', 'mIoU']
            tb.add_row([len(dataset), round(scores[1:].cpu().numpy()[0], 4),
                        round(scores[1:].cpu().numpy()[1], 4), round(scores[1:].cpu().numpy()[2], 4),
                        round(mIoU.cpu().numpy().item(), 4)])
            print('\n')
            print(tb)

        if args.save_raw_bev:
            DATAROOT = './data/nuscenes'
            nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
            os.makedirs(args.save_raw_bev, exist_ok=True)
            ds = dataset

            for idx, bev in enumerate(bev_feats):
                info = ds.data_infos[idx]
                scene = nusc.get('scene', info['scene_token'])
                log   = nusc.get('log', scene['log_token'])
                scene_name = log['logfile']

                scene_dir = osp.join(args.save_raw_bev, scene_name)
                os.makedirs(scene_dir, exist_ok=True)

                prefix = scene_name[:5]
                fname = f'{prefix}{idx:04d}.npy'
                path = osp.join(scene_dir, fname)
                np.save(path, bev.numpy())

            print(f"[INFO] Saved {len(bev_feats)} raw BEV features to {args.save_raw_bev}")
    
    """


    
    # Traduccio single gpu test
    """
    map_enable = True
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        outputs = []
        dataset = data_loader.dataset

        if map_enable:
            num_map_class = 4
            semantic_map_iou_val = IntersectionOverUnion(num_map_class).cuda()

        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):

            with torch.no_grad():
                in_data = {i: j for i, j in data.items() if 'img' in i}
                result = model(return_loss=False, rescale=True, **in_data)

            batch_size = len(result)
            
            
            if result[0]['pts_bbox'] != None:
                outputs.extend([dict(pts_bbox=result[0]['pts_bbox'])])

            if result[0]['seg_preds'] is not None:
                pred = result[0]['seg_preds']
                #Hacemos one hot encoding
                max_idx = torch.argmax(pred, dim=1, keepdim=True)               # Per cada pixel pilla l'idx de la clase amb major prob
                one_hot = pred.new_full(pred.shape, 0)                          # Crea un tensor de 0s amb igual forma que pred
                one_hot.scatter_(1, max_idx, 1)                                 # dim=1 (dimensio de clase), a la dimesio de clase aplica segons max_idx el valor 1
                
                num_cls = pred.shape[1]                                                 #Numero de clases
                indices = torch.arange(0, num_cls).reshape(-1, 1, 1).to(pred.device)    # indices tensor (C,1,1) on primera dimensió te idxs de les clases (0...C-1)
                pred_semantic_indices = torch.sum(one_hot * indices, axis=1).int()      # Multiplica en eix, acaba indicant quina clase ha predit cada pixel
                target_semantic_indices = data['semantic_indices'][0].cuda()            # Etiquetes reals de segmentació
                semantic_map_iou_val(pred_semantic_indices, target_semantic_indices)    # Calcula IoU

            for _ in range(batch_size):
                prog_bar.update()

        if map_enable:
            import prettytable as pt
            scores = semantic_map_iou_val.compute()
            mIoU = sum(scores[1:]) / (len(scores) - 1)
            tb = pt.PrettyTable()
            tb.field_names = ['Validation num', 'Divider', 'Pred Crossing', 'Boundary', 'mIoU']
            tb.add_row([len(dataset), round(scores[1:].cpu().numpy()[0], 4),
                        round(scores[1:].cpu().numpy()[1], 4), round(scores[1:].cpu().numpy()[2], 4),
                        round(mIoU.cpu().numpy().item(), 4)])
            print('\n')
            print(tb)
    """

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            assert False
            #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs, **eval_kwargs))
    

if __name__ == '__main__':
    main()
