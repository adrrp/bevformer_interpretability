import os
import argparse
import torch
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
import gc
import prettytable as pt
from scipy.spatial.distance import cdist
from saliency_techniques.util import parse_args, init_model_data
import torch.nn.functional as F
import torchvision
from mmcv.ops import box_iou_rotated
from torch.distributions.normal import Normal


def render_saliency_from_npy(npy_path, output_dir_img, args):
    print(f"Carregant saliencys desde: {npy_path}")
    saliencys = np.load(npy_path, allow_pickle=True)

    # Expandir cada mapa para que tenga shape (1, H, W) como espera save_saliency_images
    saliencys_expanded = [[cam[np.newaxis, ...] for cam in frame] for frame in saliencys]

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        plugin_dir = getattr(cfg, 'plugin_dir', os.path.dirname(args.config))
        _module_path = '.'.join(os.path.normpath(plugin_dir).split(os.sep))
        importlib.import_module(_module_path)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    os.makedirs(output_dir_img, exist_ok=True)
    save_saliency_images(saliencys_expanded, data_loader, cfg.img_norm_cfg, output_dir=output_dir_img)
    print(f"✅ Imatges guardadas a: {output_dir_img}")


def generate_random_saliency(num_frames, num_cams, h, w, mean=0.5, std=0.15, seed=42, output_path="saliency_techniques/npy-maps/random-saliency/random_saliency.npy"):
    print("Generating Random Saliency Map")
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    saliencys = []
    for _ in range(num_frames):
        frame_saliency = []
        for _ in range(num_cams):
            sal_map = np.random.normal(loc=mean, scale=std, size=(h, w)).astype(np.float32)
            sal_map = np.clip(sal_map, 0, 1)

            frame_saliency.append(sal_map)
        saliencys.append(frame_saliency)

    saliencys = np.array(saliencys)
    np.save(output_path, saliencys)
    print(f"Random map saved in: {output_path}")


def compute_frame_nds_full_diff(gt_boxes: torch.Tensor,
                                     pred_boxes: torch.Tensor,
                                     pred_scores: torch.Tensor,
                                     dist_ths: list = [0.5, 1.0, 2.0, 4.0],
                                     dist_th_tp: float = 2.0,
                                     mean_ap_weight: float = 4.0,
                                     alpha: float = 10.0,
                                     sigma: float = 0.05,
                                     N_pr_points: int = 100):

    device = pred_scores.device
    dtype = pred_scores.dtype
    eps = 1e-6
    torch_pi = torch.acos(torch.zeros(1)).item() * 2 #Equivalent a pi

    N_gt = gt_boxes.shape[0]
    N_pred = pred_boxes.shape[0]

    ctr_gt = gt_boxes[:, :3]       # [N_gt, 3]
    ctr_pr = pred_boxes[:, :3]     # [N_pred, 3]
    dim_gt = gt_boxes[:, 3:6]      # [N_gt, 3]
    dim_pr = pred_boxes[:, 3:6]    # [N_pred, 3]
    yaw_gt = gt_boxes[:, 6]        # [N_gt]
    yaw_pr = pred_boxes[:, 6]      # [N_pred]
    vel_gt = gt_boxes[:, 7:9]      # [N_gt, 2]
    vel_pr = pred_boxes[:, 7:9]    # [N_pred, 2]

    # =========================
    # Calcul soft AP
    # =========================
    dists = torch.cdist(ctr_gt, ctr_pr)  # [N_gt, N_pred]
    scores = pred_scores.view(1, -1).expand(N_gt, -1)  # [N_gt, N_pred]

    score_threshs = torch.linspace(0, 1, N_pr_points, device=device)

    softAPs = []

    for d_th in dist_ths:
        # Matching
        match_dist = torch.sigmoid(alpha * (d_th - dists))  # [N_gt, N_pred]

        """Gaussian PDF
        score_diffs = (score_threshs.view(-1, 1, 1) - scores) / sigma  # [N_threshs, N_gt, N_pred]
        F_vals = torch.exp(-0.5 * score_diffs ** 2) / (sigma * (2 * torch_pi) ** 0.5)
        F_vals = F_vals.sum(dim=0) / N_pr_points  # [N_gt, N_pred]
        """
        #Gaussian CDF
        normal = Normal(0, 1)
        score_diffs = (scores - score_threshs.view(-1, 1, 1)) / sigma  # [T, N_gt, N_pred]
        F_vals = normal.cdf(score_diffs)
        F_vals = F_vals.mean(dim=0)

        TP_contrib = (F_vals * match_dist)  # [N_gt, N_pred]
        soft_tp = TP_contrib.sum(dim=1) / (match_dist.sum(dim=1) + eps)  # [N_gt]
        softAP = soft_tp.mean()
        softAPs.append(softAP)

    mAP_frame = torch.stack(softAPs).mean() 

    # =========================================
    # TP Metrics
    # =========================================
    matches_tp = torch.sigmoid(alpha * (dist_th_tp - dists))  # [N_gt, N_pred]
    max_match, idx = matches_tp.max(dim=1)  # best pred por GT
    w_tp = max_match.view(-1, 1)  # confiança de matching

    #mATE
    ate = (ctr_gt - ctr_pr[idx]).norm(dim=1)  # distancia 3D
    mATE = (w_tp.squeeze() * ate).sum() / (w_tp.sum() + eps)


    #mASE
    boxes1 = torch.stack([ctr_gt[:,0], ctr_gt[:,1],
                          dim_gt[:,0], dim_gt[:,1], -yaw_gt], dim=1)
    boxes2 = torch.stack([ctr_pr[idx,0], ctr_pr[idx,1],
                          dim_pr[idx,0], dim_pr[idx,1], -yaw_pr[idx]], dim=1)
    iou2d = box_iou_rotated(boxes1.unsqueeze(0), boxes2.unsqueeze(0)).squeeze(0)
    ase = 1 - torch.diag(iou2d)
    mASE = (w_tp.squeeze() * ase).sum() / (w_tp.sum() + eps)

    #mAOE
    yaw_diff = torch.abs(((yaw_pr[idx] - yaw_gt + torch_pi) % (2 * torch_pi)) - torch_pi)
    mAOE = (w_tp.squeeze() * yaw_diff).sum() / (w_tp.sum() + eps)

    # AVE
    vel_diff = (vel_gt - vel_pr[idx]).norm(dim=1)
    mAVE = (w_tp.squeeze() * vel_diff).sum() / (w_tp.sum() + eps)

    # mAAE
    mAAE = torch.zeros((), device=device, dtype=dtype)

    # NDS
    w_ap = mean_ap_weight
    w_oth = 1.0
    Z = w_ap + 4 * w_oth 

    nds = (
        w_ap * mAP_frame
      + w_oth * (1 - mATE)
      + w_oth * (1 - mASE)
      + w_oth * (1 - mAOE)
      + w_oth * (1 - mAVE)
    ) / Z

    return {
        'softAP': mAP_frame,
        'mATE': mATE,
        'mASE': mASE,
        'mAOE': mAOE,
        'mAVE': mAVE,
        'mAAE': mAAE,
        'NDS': nds
    }


def run_saliency_inference(model, data_loader, saliency_type, target_class, technique="gradients", its=50):
    num_map_class = 4
    attributions_list = []

    class StatefulForwardFn:
        def __init__(self, dataset):
            self.prev = {'prev_bev': None, 'scene_token': None, 'prev_pos': None, 'prev_angle': None}
            self.outputs = []
            self.semantic_map_iou_val = IntersectionOverUnion(num_map_class).cuda()
            self.dataset = dataset
            self.nds = []

        def __call__(self, inputs, frame_idx, img_meta, target_class, data):
            if img_meta['scene_token'] != self.prev['scene_token']:
                self.prev['prev_bev'] = None

            curr_pos = np.array(img_meta['can_bus'][:3])
            curr_angle = img_meta['can_bus'][-1]

            if self.prev['prev_bev'] is not None:
                img_meta['can_bus'][:3] = curr_pos - self.prev['prev_pos']
                img_meta['can_bus'][-1] = curr_angle - self.prev['prev_angle']
            else:
                img_meta['can_bus'][:3] = 0
                img_meta['can_bus'][-1] = 0

            bev_feat, result = model.module.simple_test([img_meta], inputs, self.prev['prev_bev'], True)
            self.prev.update({'scene_token': img_meta['scene_token'], 'prev_pos': curr_pos, 'prev_angle': curr_angle, 'prev_bev': bev_feat.detach()})

            if result[0]['seg_preds'] is not None:
                pred = result[0]['seg_preds']
                max_idx = torch.argmax(pred, dim=1, keepdim=True)
                one_hot = pred.new_full(pred.shape, 0)
                one_hot.scatter_(1, max_idx, 1)
                indices = torch.arange(0, pred.shape[1]).reshape(-1, 1, 1).to(pred.device)
                pred_semantic_indices = torch.sum(one_hot * indices, axis=1).int()
                target_semantic_indices = data['semantic_indices'][0].cuda()
                self.semantic_map_iou_val(pred_semantic_indices, target_semantic_indices)

            if result[0]['pts_bbox'] is not None:
                raw = result[0]['pts_bbox']
                boxes_3d = raw['boxes_3d'].to('cpu')
                boxes_3d.tensor = boxes_3d.tensor.detach()
                self.outputs.append({'pts_bbox': {'boxes_3d': boxes_3d, 'labels_3d': raw['labels_3d'].cpu().detach(), 'scores_3d': raw['scores_3d'].cpu().detach()}})

            if saliency_type == 'segm':
                if target_class >= 0:
                    seg_preds = result[0]['seg_preds']
                    if seg_preds is None:
                        return torch.tensor([0.], device=inputs.device)
                    return seg_preds[0, target_class].sum().unsqueeze(0)
                else:
                    #Saliency general de detection
                    #Versió diferenciable quasi equivalent de one hot de quan evaluem
                    pred = result[0]['seg_preds']  # (1, C, H, W)
                    target = data['semantic_indices'][0].to(pred.device).long()  # (1, H, W)
                    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])  # (1, H, W, C)
                    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
                    probs = torch.softmax(pred, dim=1)  # (1, C, H, W)
                    intersection = (probs * target_one_hot).sum(dim=(2, 3))
                    union = (probs + target_one_hot - probs * target_one_hot).sum(dim=(2, 3))
                    soft_iou = (intersection + 1e-6) / (union + 1e-6)
                    return soft_iou.mean(dim=1)


            elif saliency_type == 'bbox':
                if target_class >= 0:
                    pred = result[0]['pts_bbox']
                    labels = pred['labels_3d']
                    scores = pred['scores_3d']
                    mask = labels == target_class
                    class_scores = scores[mask]
                    return class_scores.sum().unsqueeze(0)
                else:
                    pred = result[0]['pts_bbox']
                    pb = pred['boxes_3d'].tensor        # [N_pred, 7], float32
                    ps = pred['scores_3d']              # [N_pred]

                    device = ps.device

                    info = self.dataset.data_infos[frame_idx]
                    gt_boxes_np = info['gt_boxes']      # numpy array [N_gt, 7]
                    gt_velocity_np = info['gt_velocity']

                    gt_boxes = np.concatenate((gt_boxes_np, gt_velocity_np), axis=1)

                    diff_m = compute_frame_nds_full_diff(
                        gt_boxes=torch.as_tensor(gt_boxes, device=device, dtype=pb.dtype),
                        pred_boxes=pb, 
                        pred_scores=ps
                    )

                    return diff_m['NDS'].unsqueeze(0)

            else:
                raise ValueError(f"Unsupported saliency_type {saliency_type}")
            
        def get_segm_iou(self):
            return self.semantic_map_iou_val.compute().cpu()

    forward_fn = StatefulForwardFn(dataset=data_loader.dataset)

    if technique == "gradients":
        saliency = Saliency(forward_fn)
    elif technique == "ig":
        saliency = IntegratedGradients(forward_fn)

    prog_bar = mmcv.ProgressBar(len(data_loader.dataset)) 

    for i, data in enumerate(data_loader):
        img_meta = data['img_metas'][0].data[0][0]
        inputs = data['img'][0].data[0].cuda()

        if technique == "gradients":
            attributions = saliency.attribute(inputs, additional_forward_args=(i, img_meta, target_class, data))
        elif technique == "ig":        
            attributions = saliency.attribute(inputs, additional_forward_args=(i, img_meta, target_class, data), n_steps=its, internal_batch_size=1)

        attributions_np = attributions.detach().cpu().squeeze(0).numpy() #El squeeze es per treure el batch size
        attributions_list.append(attributions_np)

        torch.cuda.empty_cache()
        gc.collect()
        prog_bar.update()

    segm_iou = forward_fn.get_segm_iou()
    return forward_fn.outputs, attributions_list, segm_iou

def save_saliency_images(saliency_maps, data_loader, img_norm_cfg, output_dir="outputs/saliency", filter=0.01):
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating images...")
    prog_bar = mmcv.ProgressBar(len(saliency_maps)) 

    data_iter = iter(data_loader)

    for i, attributions in enumerate(saliency_maps):
        data = next(data_iter)
        cams = data['img'][0].data[0]

        overlays = [None] * 6

        for cam_id in range(cams.shape[1]):
            # 1. Saliency com L2
            sal_map = attributions[cam_id]  # shape: (3, H, W)
            sal_map = np.linalg.norm(sal_map, axis=0)  # (H, W)

            # 2. Norm unitaria
            sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-6)

            # 3. Imatge original (no normalitzat)
            mean = np.array(img_norm_cfg['mean']) / 255.0  # (3,)
            std = np.array(img_norm_cfg['std']) / 255.0    # (3,)

            # Obtener imagen original
            img_tensor = cams[0, cam_id]  # (3, H, W)
            img_np = img_tensor.cpu().permute(1, 2, 0).numpy()  # (H, W, 3)

            # Revertir norm
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)

            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 4. Aplicar colormap
            heatmap = cv2.applyColorMap((sal_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # 5. Superposició condicional
            mask = sal_map > filter
            mask_3ch = np.stack([mask] * 3, axis=-1)
            overlay = img_bgr.copy()
            overlay[mask_3ch] = (
                img_bgr[mask_3ch] * 0.6 + heatmap[mask_3ch] * 0.4
            ).astype(np.uint8)

            overlays[cam_id] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


        # 7. Concatenar 
        fila1 = np.concatenate([overlays[2], overlays[0], overlays[1]], axis=1)
        fila2 = np.concatenate([overlays[4], overlays[3], overlays[5]], axis=1)
        final_image = np.concatenate([fila1, fila2], axis=0)

        # 8. Guardar 
        out_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        prog_bar.update()


def save_saliency_images_ranked(saliency_maps, data_loader, img_norm_cfg, output_dir="outputs/saliency", filter=0.01):
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating images (ranking-based coloring)...")
    prog_bar = mmcv.ProgressBar(len(saliency_maps)) 

    data_iter = iter(data_loader)

    for i, attributions in enumerate(saliency_maps):
        data = next(data_iter)
        cams = data['img'][0].data[0]
        overlays = [None] * 6

        for cam_id in range(cams.shape[1]):
            sal_map = attributions[cam_id]
            sal_map = np.linalg.norm(sal_map, axis=0)  # (H, W)

            # 1. Ranking
            flat = sal_map.flatten()
            sorted_idx = np.argsort(flat)
            ranks = np.empty_like(sorted_idx)
            ranks[sorted_idx] = np.arange(len(flat))
            rank_map = ranks.reshape(sal_map.shape).astype(np.float32)
            rank_map /= (rank_map.max() + 1e-6)  # Normalizar a [0, 1]

            # 2. Revertir norm
            mean = np.array(img_norm_cfg['mean']) / 255.0
            std = np.array(img_norm_cfg['std']) / 255.0
            img_tensor = cams[0, cam_id]
            img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 3. Color
            colormap = cv2.applyColorMap((rank_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # 4. Superposició condicional
            mask = rank_map > filter
            mask_3ch = np.stack([mask] * 3, axis=-1)
            overlay = img_bgr.copy()
            overlay[mask_3ch] = (
                img_bgr[mask_3ch] * 0.6 + colormap[mask_3ch] * 0.4
            ).astype(np.uint8)

            overlays[cam_id] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        fila1 = np.concatenate([overlays[2], overlays[0], overlays[1]], axis=1)
        fila2 = np.concatenate([overlays[4], overlays[3], overlays[5]], axis=1)
        final_image = np.concatenate([fila1, fila2], axis=0)

        out_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        prog_bar.update()



def main_std(args, saliency_type = "bbox", target_class=-1, end_output="ig-global-bbox-base", technique="gradients"):
    cfg, model, dataset, data_loader = init_model_data(args)
    outputs, attributions_list, scores = run_saliency_inference(model, data_loader, saliency_type, target_class, technique)
    save_saliency_images(attributions_list, data_loader, cfg.img_norm_cfg, output_dir=f"outputs/{end_output}", filter=0.01)
    
    #Guardem en .npy:
    
    output_base_dir = f"saliency_techniques/npy-maps/{end_output}"
    print("Guardando Mapa Saliency .npy en ", output_base_dir)

    num_cams = attributions_list[0].shape[0]
    h, w = attributions_list[0].shape[-2:]
    saliencys = [
        [np.zeros((h, w), dtype=np.float32) for _ in range(num_cams)]
        for _ in range(len(attributions_list))
    ]
    for frame_idx, attrib in enumerate(attributions_list):
        for cam_id in range(num_cams):
            sal_map = attrib[cam_id]  # shape: (3, H, W)
            sal_map = np.linalg.norm(sal_map, axis=0)  #Norm L2
            min_val = sal_map.min()
            max_val = sal_map.max()
            norm_sal = (sal_map - min_val) / (max_val - min_val + 1e-6)
            saliencys[frame_idx][cam_id] = norm_sal

    os.makedirs(output_base_dir, exist_ok=True)
    np.save(os.path.join(output_base_dir, f"global_saliency.npy"), np.array(saliencys))
    

    
    mIoU = sum(scores[1:]) / (len(scores) - 1)
    tb = pt.PrettyTable()
    tb.field_names = ['Validation num', 'Divider', 'Pred Crossing', 'Boundary', 'mIoU']
    tb.add_row([len(dataset), round(scores[1:].cpu().numpy()[0], 4),
                round(scores[1:].cpu().numpy()[1], 4), round(scores[1:].cpu().numpy()[2], 4),
                round(mIoU.cpu().numpy().item(), 4)])
    print('\n')
    print(tb)

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for k in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(k, None)
    
    #print("Eval kwargs pre: ", eval_kwargs)
    eval_kwargs.update(dict(metric='segm', jsonfile_prefix=osp.join('test', args.config.split('/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))))
    #print("Eval kwargs post: ", eval_kwargs)
    evl = dataset.evaluate(outputs, **eval_kwargs)
    print("Keys : ", evl.keys())

    print("mIoU: ", mIoU)
    print("NDS: ", evl["pts_bbox_NuScenes/NDS"])
    


def main_save_all_saliency(technique="gradients"):
    print("Starting")
    args = parse_args()
    cfg, model, dataset, data_loader = init_model_data(args)

    segclass = ["none", "divider", "ped_crossing", "boundary"]
    bboxclass = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    output_base_dir = "saliency_techniques/npy-maps/saliency-base"
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"Generando saliency maps para {len(bboxclass)} bbox classes y {len(segclass) - 1} segm classes...")

    # BBOX
    for i, cls_name in enumerate(bboxclass):
        print(f"\n[BBox] Clase {i}: {cls_name}")
        _, attributions_list, _ = run_saliency_inference(model, data_loader, "bbox", i, technique)

        class_dir = os.path.join(output_base_dir, f"bbox_class_{i}_{cls_name}")
        os.makedirs(class_dir, exist_ok=True)
        
        for j, attrib in enumerate(attributions_list):
            np.save(os.path.join(class_dir, f"frame_{j:03d}.npy"), attrib)
        
        del attributions_list
        torch.cuda.empty_cache()
        gc.collect()

    # SEGM
    for i in range(1, len(segclass)):  # omitimos la clase 0 ("none")
        cls_name = segclass[i]
        print(f"\n[Segm] Clase {i}: {cls_name}")
        _, attributions_list, _ = run_saliency_inference(model, data_loader, "segm", i, technique)

        class_dir = os.path.join(output_base_dir, f"segm_class_{i}_{cls_name}")
        os.makedirs(class_dir, exist_ok=True)
        
        for j, attrib in enumerate(attributions_list):
            np.save(os.path.join(class_dir, f"frame_{j:03d}.npy"), attrib)
        
        del attributions_list
        torch.cuda.empty_cache()
        gc.collect()

def main_global_saliency(args, technique="gradients"):
    print("Starting")
    cfg, model, dataset, data_loader = init_model_data(args)

    for i in range(25):
        print("LES CLASES TRAILER, CONSTRUCTION VEHICLE I BARRIER NO APAREIXEN MAI EN EL ESTANDARD 2 ESCENES, POT DONAR ERRORS O INCOHERENCIES SALIENCY?")


    segclass = ["none", "divider", "ped_crossing", "boundary"] #
    bboxclass = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'] #

    output_dir = "saliency_techniques/npy-maps/ig-global-sum-base"
    output_dir_img = "outputs/ig-global-sum-base"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_img, exist_ok=True)

    print(f"Generando saliency global para {len(bboxclass)} clases bbox + {len(segclass) - 1} clases segm...")


    num_cams = None
    h, w = None, None
    saliencys = None

    all_classes = [('bbox', i, name) for i, name in enumerate(bboxclass)] + \
                  [('segm', i, name) for i, name in enumerate(segclass) if i > 0]

    for sal_type, idx, name in all_classes:
        print(f"\n[{sal_type.upper()}] Clase {idx}: {name}")
        _, attributions_list, _ = run_saliency_inference(model, data_loader, sal_type, idx, technique, its=25)

        if saliencys is None:
            num_cams = attributions_list[0].shape[0]
            h, w = attributions_list[0].shape[-2:]
            saliencys = [
                [np.zeros((h, w), dtype=np.float32) for _ in range(num_cams)]
                for _ in range(len(attributions_list))
            ]

        for frame_idx, attrib in enumerate(attributions_list):
            for cam_id in range(num_cams):
                sal_map = attrib[cam_id]  # shape: (3, H, W)
                sal_map = np.linalg.norm(sal_map, axis=0)
                min_val = sal_map.min()
                max_val = sal_map.max()
                norm_sal = (sal_map - min_val) / (max_val - min_val + 1e-6)
                saliencys[frame_idx][cam_id] += norm_sal

        del attributions_list
        torch.cuda.empty_cache()
        gc.collect()

    npy_path = os.path.join(output_dir, "global_saliency.npy")
    np.save(npy_path, np.array(saliencys))
    print(f"\n✅ Saliency global guardado en: {npy_path}")

    # Guardar imágenes
    saliencys_expanded = [[cam[np.newaxis, ...] for cam in frame] for frame in saliencys]
    save_saliency_images(saliencys_expanded, data_loader, cfg.img_norm_cfg, output_dir=output_dir_img)



def main_render_saliency():
    args = parse_args()
    render_saliency_from_npy("saliency_techniques/npy-maps/ig-global-bbox-base/global_saliency.npy", "outputs/saliency-decompressed", args)



if __name__ == '__main__':
    
    #IG tarda 25m en generar saliency map per bbox global small
    args = parse_args()
    main_std(args, saliency_type="bbox", target_class=-1, end_output="cdf-ig-global-bbox-base", technique="ig")
    
